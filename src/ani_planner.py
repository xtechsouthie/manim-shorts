from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from typing import List
from .state import VideoSegment, VideoState, OutputSchema
from langchain_core.runnables.config import RunnableConfig
import os

def animation_planner_orchestrator(state: VideoState) -> List[Send]:
    print(f"Starting animation planner orchestrator for {len(state.segments)} segments")
    return [Send("animation_planner_worker", {"segment": segment}) for segment in state.segments]

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

def animation_planner_worker(data: dict, config: RunnableConfig) -> dict:

    segment = data["segment"]

    code_examples = query_manim_rag(query=segment.text, k=3)

    llm = config["configurable"]["animation_llm"]

    print(f"----Worker planning animation for the segment {segment.segment_id}")

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at creating Manim library animation descriptions/pseudocode for animations in educational videos."),
            ("human", """Create a detailed Manim pseudocode for this segment:

                Narration: {text}
                Duration: {duration} seconds
             
                Below are some of the code examples from 3Blue1Brown.
                You can refer to them for making the animation pseudocode.
                NOTE: The code provided may or may not match this particular animation narration.
                So use the code as reference only after checking that it is matching this particular animation narration
             
                <START OF CODE EXAMPLES>
             
                {examples}
             
                <END OF CODE EXAMPLES>
                
                NOTE: YOU ONLY HAVE TO PROVIDE PSEUDOCODE BASED ON THE NARRATION GIVEN ABOVE.
                The video style should match to that of the youtube channel 3Blue1brown by
                Grant Sanderson. The video animation prompt/pseudocode should be something that
                Manim library can make good animations of, like colourful graphs, diagrams, 3d/2d plots and curves,
                mathematical expressions, symbols, equations and written words, etc.

                Provide a clear, specific animation pseudocode that:
                1. Matches the narration content, The pseudocode should strictly supplement or match the Narration text
                2. Can be created with Manim (mathematical animations library)
                3. Is visually engaging and educational
                4. Can be completed in {duration} seconds
                5. Uses Manim's capabilities: graphs, equations, geometric shapes, transformations
                6. They should sync with the narration provided.

                Include:
                - What objects to show (text, shapes, graphs, equations, diagrams)
                - What animations to use (FadeIn, Transform, Create, Write, etc.)
                - Color scheme (use vibrant colors)
                - Key visual moments that sync with narration
             
                pseudocode instructions:
                1. The pseudocode should be detailed, describing every animation to last detail
                2. The coding agent can refer to the pseudocode and create exact animations using the manim library
                3. Make sure that the functions, tools that you use are available in the manim library.

                Be specific and concise.
                """)
        ])

        messages = prompt.format_messages(
            text = segment.text,
            duration = segment.audio_duration_sec,
            examples = code_examples,
        )

        response = llm.invoke(messages)
        segment.animation_prompt = response.content

        print(f"----The animation prompt of segment {segment.segment_id} is created.")
        return {"segments": [segment]}
    except Exception as e:
        print(f"Error creating segment: {e}")
        return {"segments": [segment]}
    
def create_animation_planner_graph():
    graph = StateGraph(state_schema=VideoState, output_schema=OutputSchema)
    try:

        graph.add_node("animation_planner_worker", animation_planner_worker)

        graph.add_conditional_edges(START, animation_planner_orchestrator)
        graph.add_edge("animation_planner_worker", END)

        return graph.compile()
    except Exception as e:
        print(f"Error in animation planner: {e}")
        return graph.compile()





