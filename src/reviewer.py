from langgraph.graph import StateGraph, START
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Send
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from typing import Tuple, List
from .state import VideoState
import subprocess, time
import tempfile
import os
import random
import re
import json
from datetime import datetime

def safe_llm_invoke(llm, messages, max_retries=5, base_delay=3):
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e).lower() or "timeout" in str(e).lower():
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                print(f"Rate limit hit, attempt: {attempt+1}/{max_retries}, delaying for {delay:.2f} secs")
                time.sleep(delay)
            else:
                raise e
            
    raise Exception("Too many rate limit retires, retying")

def query_docs_rag(llm, logs: str, code: str, k: int) -> Tuple[str, str]:
    if not logs:
        return "No error logs, and docs as code is running successfully"
    
    prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at analyzing Manim error logs and extracting key information."),
            ("human", """
Analyze these Manim execution logs and extract the KEY ERROR INFORMATION for documentation search.

EXECUTION LOGS:
{logs}

----------------------------
CODE CONTEXT:
```python
{code}
```

Extract and return ONLY:
1. The main error type (AttributeError, TypeError, ImportError, etc.)
2. The specific Manim class/function causing the issue (important)
3. Key missing attributes or incorrect usage patterns
4. Suggested changes only if it is given in the logs.

Return a SHORT summary focusing on:
- Error type
- Manim objects involved (e.g., "MathTex", "ThreeDScene", "Create")
- What went wrong (e.g., "missing method", "invalid color", "wrong syntax")

Examples of good summaries:
- "AttributeError: MathTex object missing set_color method, invalid color GOLD used"
- "ThreeDScene: 2D Text objects not added with add_fixed_in_frame_mobjects"
- "ImportError: Cannot import Arc from manim.mobject.geometry"

Return ONLY the summary, no explanations.
""")
        ])
    

    messages = prompt.format_messages(
        logs=logs,
        code=code
    )

    try:
        response = safe_llm_invoke(llm, messages)
        summary = response.content.strip()

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store_path = "./chroma_docs_db"

        vector_store = Chroma(
            collection_name="docs",
            embedding_function=embeddings,
            persist_directory=vector_store_path
        )

        docs = vector_store.similarity_search(summary, k=k)

        docs_content = ""
        for i, doc in enumerate(docs, 1):
            docs_content += f"--- Documentation {i} ---\n{doc.page_content}\n\n"

        print(f"----Retirved {len(docs)} docs from the query")
        return (docs_content, summary)
    
    except Exception as e:
        print(f"Error while performing RAG in docs in reviewer: {e}")
        return ("Documentation retrival failed", logs)



class CodeReviewerAgent:
    """Reviews and fixes the manim code through iterative cycles"""

    def __init__(self, llm: ChatAnthropic, llm2: ChatOpenAI, max_cycles: int =5): #yaha
        self.llm = llm
        self.llm2 = llm2
        self.max_cycles = max_cycles


    def review_and_fix_code(self, code: str, segment_id: int, animation_prompt: str, duration: float) -> Tuple[str, bool]:

        current_code = code
        error_history = []

        for cycle in range(self.max_cycles):
            print(f"----Code Review cycle {cycle + 1} for Segment {segment_id}")

            success, logs = self._execute_code(current_code, segment_id)
            
            if success:
                docs_content = "Code validated successfully - no documentation needed"
                error_summary = "No errors"
            else:
                docs_content, error_summary = query_docs_rag(
                    llm=self.llm2, 
                    logs=logs, 
                    code=current_code, 
                    k=3
                )
                error_history.append({
                    "cycle": cycle+1,
                    "error": error_summary,
                    "code": current_code
                })
                       

            if success:
                print(f"----Segment {segment_id} ready to run")
                return (current_code, True)
            
            current_code = self._generate_fix(
                current_code, 
                logs, 
                animation_prompt, 
                duration, 
                segment_id,
                error_summary,
                docs_content,
                error_history
            )

        print(f"----Segment {segment_id} failed after {self.max_cycles} cycles")
        return (current_code, False)
    
    def _execute_code(self, code: str, segment_id: int) -> Tuple[bool, str]:
        """Execute Manim code and capture errors"""

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_file = f.name

            syntax_check = subprocess.run(
                ["python", "-m", "py_compile", temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )

            if syntax_check.returncode != 0:
                os.unlink(temp_file)
                return (False, f"Python syntax error: {syntax_check.stderr}")
            

            scene_pattern = r'class\s+(\w+)\s*\(\s*(?:Scene|ThreeDScene)\s*\)'
            scene_names = re.findall(scene_pattern, code)

            if not scene_names:
                os.unlink(temp_file)
                return (False, "No valid class found, must inherit from Scene or ThreeDScene")
            
            logs = "ERROR in scene validation: \n\n"
            success = True

            for scene in scene_names:
                command = [
                    "manim",
                    "-ql",
                    "--dry_run",
                    temp_file,
                    scene
                ]

                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=os.environ.copy()
                )

                scene_log = f"""
                SCENE: {scene}
                --------------------
                STDOUT:
                {process.stdout}
                --------------------
                STDERR:
                {process.stderr}
                --------------------
                """

                logs += scene_log

                if process.returncode != 0:
                    success = False
                    logs += f"\n\nERROR: Scene {scene} validation failed with exit code {process.returncode}\n"

            os.unlink(temp_file)

            if success:
                return (True, "Success, All scenes are validated successfully")
            else:
                return (False, logs)
                
                
        except subprocess.TimeoutExpired:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            print("Timeout error in _execute_code")
            return (False, "Code validation timed out (timeout error)")
        except Exception as e:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            print(f"Error in _execute_code: {e}")
            return (False, f"Validation error: {e}")
        
    def _generate_fix(self, code: str, logs: str, animation_prompt: str, duration: float, segment_id: int, error_summary: str, docs_content: str, error_history: List[dict]) -> str:

        history_text = ""
        if error_history:
            history_text = "\n---PREVIOUS ERRORS IN THIS SEGMENT START---\n\n"
            for entry in error_history[-5:]:
                history_text += f"CYCLE {entry["cycle"]}: {entry["error"]}\n"
            history_text += "\n---END OF PREVIOUS ERRORS ---\n"

        if len(logs) > 1000:
            truncated_logs = logs[:500] + "\n\n... (middle section truncated) ...\n\n" + logs[-500:]
        else:
            truncated_logs = logs


        prompt = ChatPromptTemplate.from_messages([("system", "You are a expert in fixing manim code with bugs, Return ONLY the fixed manim python code"),
                ("human", """
Fix manim code given the execution logs which tell about the bugs and problems in the code:
YOUR JOB IS TO SEE THE PROBLEMS IN THE CODE REFERING TO THE PROMPT BELOW AND JUST CHANGE THE PROBLEMATIC LINES OF CODE
DO NOT TRY TO CHANGE THE ENTIRE CODE UNLESS NECESSARY. JUST CHANGE THE PROBLEMATIC LINES. RETURN THE ENTIRE CORRECTED CODE ONLY.

CURRENT CODE:
```python
{code}
```
------------------------------------

CURRENT ERROR SUMMARY:
{error_summary}

------------------------------------
Check the previous errors, don't make the same mistakes again.
{previous_errors}


EXECUTION LOGS START--------------
{logs}
EXECUTION LOGS END----------------

NECESSARY MANIM COMMUNITY DOCS FOR REFERENCE (MAY OR MAY NOT BE RELEVANT):
DOCUMENTATION START---------------
{docs}
DOCUMENTATION END-----------------

CONTEXT:
- animation prompt: {animation_prompt}
- Required duration: {duration} seconds
- Segment ID: {segment_id}

REQUIREMENTS:
1. Fix the code based of the logs given above. RETURN ONLY THE CORRECTED CODE, Refer to the documentation
2. Explain the bug/issue and fix you made in a short comment below your corrected code. Do not return any other explainations or comments in the code, just simple code
3. The Timing of the animation should match EXACTLY {duration} seconds. Use self.wait() if needed.
4. Do not change the animation scenes or colours unless neccessary
5. The video rendered by the code should have no elements that overlap
6. Use modules and functions that are part of the manim community library
7. Make sure that Class name is Segment{segment_id} and the class in inhertited from Scene or ThreeDScene Class only.
8. Refer (if neccessary) to the comment about the previous issue/bug and the fix at the end of the code block if it exists.
9. Check the previous errors and do not make the same mistakes again.
10. The background colour should be pure black (#000000)
11. Use ONLY valid Manim Community colors: BLUE, RED, GREEN, YELLOW, PURPLE_A, PURPLE_B, PURPLE_C, ORANGE, WHITE, PINK, GRAY
   - NO: GOLD, TEAL, CYAN, MAGENTA, MAROON, PASTEL_*, VIVID_*, BRIGHT_*

If logs are empty, return the EXACT code unchanged.

DO NOT USE external resources and dependencies like svg's, images or other libraries (other than the one's already used in code)
DO NOT CHANGE/MENTION MINOR THINGS like missing or wrong comments in code, CHANGE THE CODE that is crucial to the functioning of the code.

""")])
        
        messages = prompt.format_messages(
        code=code,
        error_summary=error_summary,
        previous_errors= history_text,  
        logs=truncated_logs,  
        docs=docs_content,  
        animation_prompt=animation_prompt,
        duration=duration,
        segment_id=segment_id
    )

        response = safe_llm_invoke(self.llm, messages)

        manim_code = response.content
        if "```python" in manim_code:
            manim_code = manim_code.split("```python")[1].split("```")[0].strip()
            return manim_code
        elif "```" in manim_code:
            manim_code = manim_code.split("```")[1].split("```")[0].strip()
            return manim_code
        
        return manim_code.strip()
    
def code_reviewer_node(state: VideoState, config) -> dict:

    reviewer = CodeReviewerAgent(llm=config["configurable"]["review_llm"],llm2= config["configurable"]["summary_llm"],max_cycles=5) 

    updated_segments = []
    segments_needing_regen = []

    for segment in state.segments:
        if segment.manim_script:
            print(f"----Reviewing code for segment {segment.segment_id}")

            fixed_code, success = reviewer.review_and_fix_code(
                code=segment.manim_script,
                segment_id=segment.segment_id,
                animation_prompt=segment.animation_prompt,
                duration=segment.audio_duration_sec
            )

            segment.manim_script = fixed_code

            if not success:
                print(f"----Segment {segment.segment_id} failed in reviewer, sending for regen")
                segments_needing_regen.append(segment)
            else:
                print(f"----Segment {segment.segment_id} validation successful")

        updated_segments.append(segment)

    return {
        "segments": updated_segments,
        "segments_needing_regeneration": segments_needing_regen
    }

def route_after_review(state: VideoState):

    segments_needing_regen = state.segments_needing_regeneration

    if segments_needing_regen:
        print(f"----Sending {len(segments_needing_regen)} segments back to regeneration")
        return "manim_generation"
    else:
        print("----All segments validated successfully, proceeding to renderer")
        return "manim_renderer"

