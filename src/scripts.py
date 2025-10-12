from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from .state import VideoState, ScriptOutput, VideoSegment
from langchain_core.runnables.config import RunnableConfig

#change this to use chatprompttemplate as they do in docs

def scriptwriter_agent(state: VideoState, config: RunnableConfig) -> VideoState:
    print("Running scriptwriter\n")

    llm = config["configurable"]["script_llm"]

    prompt = f"""Create a 2 minute educational video script about: {state.topic}

    The Video script should be like the videos by youtube channel 3Blue1Brown by Grant Sanderson.
    Make the Script simple and explain concepts clearly with geometrical intuition if possible, Include mathematical expressions and derivations if applicable.

    Requirements:
    1. Total duration should be around 120-150 seconds (around 2 minutes) (300-350 words maximum)
    2. Split into 2-5 segments (each segment = one clear concept that can be explained by one animation)
    3. Each segment should be 30-60 seconds long, with maximum of 5 segments, ideally keep 3-4 segments
    4. Write in simple, narrative style suitable for voice narration
    5. Make sure that the script does not have silly explainations/visualizations (like suppose you are on hill or moon etc.), the Manim library
    that is used to make the animations is only good at animating colourful graphs, diagrams, 3d/2d plots and curves,
    mathematical expressions, symbols and written words. So the script should be such that we could make animations
    using the manim library for that script.
    6. Each segment should be explainable with ONE clear animation
    7. When you are giving the script, give it like a human would read it. Since the text goes to a text to speech agent,
        give the script like a human would read it. Like instead of f(x), give "f of x". or instead of  x^2 = 4 give "x squared equals 4".
    8. Don't make it too complex, EXPLAIN IN LAYMAN TERMS

    Make it engaging and educational
    """

    structured_llm = llm.with_structured_output(ScriptOutput)

    try:
        response = structured_llm.invoke([HumanMessage(content=prompt)])

        state.full_script = response.full_script
        print(response.full_script)
        state.segments = []

        for seg in response.segments:
            segment = VideoSegment(
                segment_id=seg.segment_id,
                text=seg.script,
                planned_duration=seg.duration_sec,
                audio_path="",
                audio_duration_sec=0.0,
                animation_prompt="",
                video_path="",
                manim_script=""
            )
            state.segments.append(segment)

        return state
    except Exception as e:
        state.error = f"ScriptWriter error: {e}"
        print(f"Error in scriptwriter: {e}\n")
        return state
    


    