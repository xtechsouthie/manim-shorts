from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from src.state import VideoState
from src.scripts import scriptwriter_agent
from src.audio import create_audio_graph
from src.ani_planner import create_animation_planner_graph
from src.manim_agent import create_manim_graph
from src.composer import video_composer
from IPython.display import Image, display

load_dotenv()

def create_workflow():

    workflow = StateGraph(VideoState)

    workflow.add_node("scriptwriter", scriptwriter_agent)
    workflow.add_node("audio_generation", create_audio_graph())
    workflow.add_node("animation_planning", create_animation_planner_graph())
    workflow.add_node("manim_generation", create_manim_graph())
    workflow.add_node("composer", video_composer)

    workflow.add_edge(START, "scriptwriter")

    workflow.add_edge("scriptwriter", "audio_generation")
    workflow.add_edge("scriptwriter", "animation_planning")

    workflow.add_edge("audio_generation", "manim_generation")
    workflow.add_edge("animation_planning", "manim_generation")

    workflow.add_edge("manim_generation", "composer")
    workflow.add_edge("composer", END)

    return workflow.compile()

def main():
    print("-" * 30)
    print("3Blue1Brown style educational video generation")
    print("-" * 30)

    video_topic = input("\nEnter the video topic you want to explore: ").strip()

    if not video_topic:
        print("Topic cannot be empty")
        return
    
    print(f"Creating video on topic: {video_topic}")
    print("-" * 30)

    try:
        openai_llm = ChatOpenAI(model="gpt-5-mini", temperature=0.6)
        claude_llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.6, max_tokens_to_sample=4096)

        app = create_workflow()

        #display(Image(app.get_graph().draw_mermaid_png()))

        initial_state = VideoState(
            topic=video_topic,
            full_script="",
            segments=[],
            final_video_path="",
            current_segment_id=0      
        )

        print(initial_state)
        print("Starting the video generation pipeline")

        result = app.invoke(
            initial_state,
            config={
                "configurable": {
                    "script_llm": openai_llm,
                    "animation_llm": openai_llm,
                    "manim_llm": claude_llm,
                }
            }
        )

        if hasattr(result, 'error') and result.error:
            print(f"Error with generating results: {result.error}")
            return
        
        print("\n" + "="*60)
        print("VIDEO GENERATION COMPLETE")
        print("=" * 60)
        print(f"\n Final video path: {result.final_video_path}")


    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

              
    

