from langgraph.types import Send
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from typing import List
from .state import VideoSegment, VideoState

def animation_planner_orchestrator(state: VideoState) -> List[Send]:
    print("Starting animation planner orchestrator")
    return [
        Send("animation_planner_worker", {"segment": segment, "state": state})
        for segment in state.segments
    ]

def animation_planner_worker(data: dict, config: dict) -> VideoSegment:

    segment = data["segment"]
    state = data["state"]
    topic = state.topic

    llm = config["configurable"]["animation_llm"]

    print(f"----Worker planning animation for the segment {segment.segment_id}")

    try:
        prompt = ChatPromptTemplate.from_messages([
            "system", "You are an expert at creating Manim library animation descriptions/prompts for educational videos."
            "human", """Create a detailed Manim animation prompt for this segment:

                Narration: {text}
                Duration: {duration} seconds
                Topic: {topic}

                The video style should match to that of the youtube channel 3Blue1brown by
                Grant Sanderson. The video animation prompt/description should be something that
                Manim library can make good animations of, like colourful graphs, diagrams, 3d/2d plots and curves,
                mathematical expressions, symbols, equations and written words, etc.

                Provide a clear, specific animation description that:
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

                Be specific and concise.
                """
        ])

        messages = prompt.format_messages(
            text = segment.text,
            duration = segment.audio_duration_sec,
            topic = topic
        )

        response = llm.invoke(messages)
        segment.animation_prompt = response.content

        print(f"----The animation prompt of segment {segment.segment_id} is created.")
        return segment
    except Exception as e:
        print(f"Error creating segment: {e}")
        return segment
    
def create_animation_planner_graph():
    graph = StateGraph(VideoState)

    graph.add_node("orchestrator", animation_planner_orchestrator)
    graph.add_node("animation_planner_worker", animation_planner_worker)

    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges("orchestrator", lambda x: x)
    graph.set_finish_point("animation_planner_worker")

    return graph.compile()



