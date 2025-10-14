from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from typing import List
from .state import VideoSegment, VideoState, OutputSchema
from langchain_core.runnables.config import RunnableConfig

def animation_planner_orchestrator(state: VideoState) -> List[Send]:
    print("Starting animation planner orchestrator")
    return [Send("animation_planner_worker", {"segment": segment}) for segment in state.segments]

def animation_planner_worker(data: dict, config: RunnableConfig) -> dict:

    segment = data["segment"]


    llm = config["configurable"]["animation_llm"]

    print(f"----Worker planning animation for the segment {segment.segment_id}")

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at creating Manim library animation descriptions/pseudocode for animations in educational videos."),
            ("human", """Create a detailed Manim pseudocode for this segment:

                Narration: {text}
                Duration: {duration} seconds
                MAXIMUM LENGTH OF PSEUDOCODE: 200 WORDS. MANAGE THE PSEUDOCODE WITHIN THE LIMIT

                The video style should match to that of the youtube channel 3Blue1brown by
                Grant Sanderson. The video animation prompt/pseudocode should be something that
                Manim library can make good animations of, like colourful graphs, diagrams, 3d/2d plots and curves,
                mathematical expressions, symbols, equations and written words, etc.

                Provide a clear, specific animation pseudocode that:
                1. Matches the narration content, The video prompt should strictly supplement or match the Narration text
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





