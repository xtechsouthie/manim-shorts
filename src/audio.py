from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from pathlib import Path
from .state import VideoSegment, VideoState
from pydub import AudioSegment

def audio_orchestrator(state: VideoState) -> list[Send]:
    print(f"Running audio orchestrator for creating audio for {len(state.segments)} segments\n")

    return [
        Send("audio_worker", {"segment": segment, "config": state})
        for segment in state.segments
    ]

def audio_worker(seg: dict) -> VideoSegment:
    segment = seg["segment"]
    state = seg["state"]

    print(f"----Worker processing segement ID: {segment.segment_id}")

    instructions = """Voice Affect: Calm, composed, and reassuring; project quiet authority and confidence.
    Tone: Sincere, empathetic, and gently authoritative—express genuine apology while conveying competence.
    Pacing: Steady and moderate; unhurried enough to communicate care, yet efficient enough to demonstrate professionalism.
    Emotion: Genuine empathy and understanding; speak with warmth, especially during apologies ("I'm very sorry for any disruption...").
    Pronunciation: Clear and precise, emphasizing key reassurances ("smoothly," "quickly," "promptly") to reinforce confidence.
    Pauses: Brief pauses after offering assistance or requesting details, highlighting willingness to listen and support.    
    """

    try:
        client = OpenAI()
        audio_dir = Path("video_files/audio")
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_path = audio_dir / f"segment_{segment.segment_id}.mp3"

        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="sage",
            input=segment.text,
            instructions=instructions,
        ) as response:
            response.stream_to_file(audio_path)

        audio = AudioSegment.from_mp3(str(audio_path))
        duration = len(audio) / 1000.0

        segment.audio_path = str(audio_path)
        segment.audio_duration_sec = duration

        print(f"----Segment {segment.segment_id} audio generated with duration of {duration:.2f} seconds")
        return segment
    except Exception as e:
        print(f"----Error in segment: {e}")
        return segment
    
def create_audio_graph():

    graph = StateGraph(VideoState)

    graph.add_node("orchestrator", audio_orchestrator)
    graph.add_node("audio_worker", audio_worker)
    
    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges("orchestrator", lambda x: x,)
    graph.set_finish_point("aggregator")

    return graph.compile()