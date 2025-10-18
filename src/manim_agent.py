from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from typing import List
import time
import subprocess
from .state import VideoSegment, VideoState, ManimScript
from langchain_core.runnables.config import RunnableConfig
import os, uuid
import shutil
import random, time
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def safe_llm_invoke(llm, messages, max_retries=5, base_delay=2):
    """Retry LLM calls with exponential backoff + jitter if rate limit or timeout occurs."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e) or "timeout" in str(e).lower():
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit hit (attempt {attempt+1}/{max_retries}), retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise e
    raise Exception("Too many rate-limit retries, aborting.")

def query_manim_rag(query: str, k: int) -> str:
    try:
        chroma_dir = "./chroma_manim_db"

        if not os.path.exists(chroma_dir):
            print(f"Warning: No database found at {chroma_dir}")
            return ""
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        vector_store = Chroma(
            collection_name="manim_code",
            embedding_function=embeddings,
            persist_directory=chroma_dir
        )

        results = vector_store.similarity_search_with_score(query=query, k=k)

        if not results:
            print("no results found from rag")
            return ""
        
        examples_text = ""
        
        for i, (doc, score) in enumerate(results, 1):
            examples_text += f"--- Example {i} (from {doc.metadata.get('file', 'unknown')}, similarity: {score:.2f}) ---\n"
            examples_text += f"```python\n{doc.page_content}\n```\n\n"        

        return examples_text

    except Exception as e:
        print(f"Error with RAG query: {e}")
        return ""


def manim_orchestrator(state: VideoState) -> List[Send]:
    print("Starting manim orchestrator")
    
    segments_needing_regen = state.segments_needing_regeneration
    
    if segments_needing_regen:
        print(f"----Regenerating {len(segments_needing_regen)} failed segments")
        return [
            Send("manim_worker", {
                "segment": segment,
                "manim_dir": "video_files/manim_script",
                "video_dir": "video_files/video"
            })
            for segment in segments_needing_regen
        ]
    else:
        print(f"----Processing all {len(state.segments)} segments")
        return [
            Send("manim_worker", {
                "segment": segment,
                "manim_dir": "video_files/manim_script",
                "video_dir": "video_files/video"
            })
            for segment in state.segments
        ]

def manim_worker(data: dict, config: RunnableConfig) -> dict:

    segment = data["segment"]
    manim_dir = Path(data["manim_dir"])
    video_dir = Path(data["video_dir"])

    manim_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    llm = config["configurable"]["manim_llm"]

    print(f"----Worker generating manim script for segement {segment.segment_id}")

    if segment.segment_id > 0:
        delay = int(segment.segment_id) + 1
        print(f"----Waiting {delay:.1f}s to avoid rate limits...")
        time.sleep(delay)

    try:
        rag_examples = query_manim_rag(segment.animation_prompt, k=2)

        structured_llm = llm.with_structured_output(ManimScript)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Manim (Mathematical Animation Engine) programmer.
Generate complete, working Manim Community Edition code that creates engaging educational animations.
Always include proper imports and ensure timing exactly matches the required duration"""),
        ("human", """Generate a complete Manim Python script for this animation pseudocode:

KEEP THE CODE SHORT AND SIMPLE, DO NOT GIVE BIG CODE. KEEP SHORT, SIMPLE CODE
Animation Pseudocode: {animation_prompt}
Required Duration: {duration} seconds
Segment ID: {segment_id}
         
Below are some example animation manim scripts by 3Blue1Brown for reference
Please note that these script use the 3b1b version of Manim, not ManimCommunity, so the functions, tools, etc may be different or have different names.
You have to write code in Manim Community edition, not 3b1b version.
Some of the code in examples may be old and depreciated, be aware of that while writing your own code.
Refer to the code for animations, animation styles, colours, visual style, etc.
         
<START OF MANIM CODE EXAMPLES>

{examples}
         
</END OF MANIM CODE EXAMPLES>


USE THE PSEUDOCODE AND CREATE THE CORRESPONDING ANIMATION USING MANIM COMMUNITY EDITION.
KEEP TRACKS OF THE FUNCTIONS AND TOOLS YOU USE, MAKE SURE THEY ARE IN THE MANIM COMMUNITY EDITION LIBRARY.
CHECK THE CODE FOR POSSIBLE BUGS BEFORE RESPONDING, THE CODE SHOULD BE BUG FREE.         

Requirements:
1. Use Manim Community Edition (from manim import *)
2. Create a Scene class called Segment{segment_id}
3. MUST use self.wait() (if necessary) to reach EXACTLY {duration} seconds total runtime.
4. Use clear, educational animations (Write, Create, FadeIn, Transform, etc.)
5. The video rendered by the code should have no elements that overlap
6. Use vibrant colors and clear text and include proper timing comments
7. Match the 3Blue1Brown visual style, refer to example code.
8. Use ONLY these colors: BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE, WHITE, PINK
   (NO CYAN, GOLD, TEAL, MAGENTA, MAROON - they cause errors)
9. Use Scene class ONLY (never MovingCameraScene)
10. For 3D scenes: Use ThreeDScene, not Scene, (although try to use Scene instead of ThreeDScene whenever possible,
          if the animation strictly requires 3D animation, then only use ThreeDScene)        
        
# Example code for reference:
# self.play(SomeAnimation, run_time=X) 
# self.wait(Y)
Total of X + Y + ... should equal {duration} seconds STRICTLY.
         
Animation is planned so that it runs for the given duration, use self.wait() only when the animation timing does not reach the given duration.
NO COMMENTS, NO EXPLAINATIONS, JUST RETURN ONLY PYTHON CODE.
Return the complete working code with all imports. Don't give any explainations or text, just give code.""")
        ])

        messages = prompt.format_messages(
            animation_prompt = segment.animation_prompt,
            duration = segment.audio_duration_sec,
            examples = rag_examples,
            segment_id=segment.segment_id
        )

        response = safe_llm_invoke(structured_llm, messages)

        manim_code = response.completed_code
        if "```python" in manim_code:
            manim_code = manim_code.split("```python")[1].split("```")[0].strip()
        elif "```" in manim_code:
            manim_code = manim_code.split("```")[1].split("```")[0].strip()

        segment.manim_script = manim_code
        script_path = manim_dir / f"segment_{segment.segment_id}.py"

        with open(script_path, "w") as f:
            f.write(manim_code)

        print(f"----Segment {segment.segment_id}: Manim script saved, now rendering")

        video_path = video_dir / f"segment_{segment.segment_id}.mp4"

        #see this shii::
        unique_tex_dir = manim_dir / f"tex_temp_{uuid.uuid4()}"
        unique_tex_dir.mkdir(exist_ok=True)

        env = os.environ.copy()
        env["MANIMCE_TEX_DIR"] = str(unique_tex_dir)
        env["MANIM_DISABLE_CACHING"] = "true"

        render_cmd = [
            "manim",
            str(script_path.absolute()),
            response.class_name,
            "-qm",
            "--format", "mp4",
            "-o", str(video_path.absolute()),
            "--disable_caching"
        ]

        result = subprocess.run(
            render_cmd,
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )

        if result.returncode != 0:
            error_msg = result.stderr
            print(f"----Manim render error: {error_msg}")

            #just tyring to find the output in default manim locations:
            default_locations = [
                Path("media/videos") / script_path.stem / "1080p60" / f"{response.class_name}.mp4",
                Path("media/videos") / script_path.stem / "720p30" / f"{response.class_name}.mp4",
            ]

            for default_path in default_locations:
                if default_path.exists():
                    shutil.move(str(default_path), str(video_path))
                    print(f"----video was found in default location, moved to {str(video_path)}")
                    break

        if video_path.exists():
            segment.video_path = str(video_path)
            print(f"----Segment {segment.segment_id} rendered successfully")
        else:
            raise Exception(f"Video file not created at {str(video_path)}")
        
        shutil.rmtree(unique_tex_dir, ignore_errors=True)
        
        return {"segments": [segment], "segments_needing_regeneration": []}
    except subprocess.TimeoutExpired:
        print(f"----ERROR: Segment {segment.segment_id}: Rendering Timeout")
        return {"segments": [segment], "segments_needing_regeneration": []}
    
    except Exception as e:
        print(f"----Error rendering segment: {e} ")
        return {"segments": [segment], "segments_needing_regeneration": []}
    
def create_manim_graph():

    graph = StateGraph(VideoState)

    graph.add_node("manim_worker", manim_worker)

    graph.add_conditional_edges(START, manim_orchestrator)
    graph.add_edge("manim_worker", END)

    return graph.compile()


