from pydantic import BaseModel, Field
from typing import List, Dict, Annotated
import operator

class VideoSegment(BaseModel):
    segment_id: int
    text: str
    planned_duration: float
    audio_path: str
    audio_duration_sec: float
    animation_prompt: str
    video_path: str
    manim_script: str

class VideoState(BaseModel):
    topic: str
    full_script: str
    segments: Annotated[List[VideoSegment], operator.add]
    final_video_path: str
    error: str = None
    current_segment_id: int

class ScriptSegment(BaseModel):
    segment_id: int = Field(description="The ID of the segment created")
    script: str = Field(description="The script created for the particular segement")
    duration_sec: float = Field(description="The duration of that script segment in seconds")

class ScriptOutput(BaseModel):
    """A stuctured output of the script"""
    full_script: str = Field(description="The full script for the short video")
    segments: List[ScriptSegment] = Field(description="List of script segments.")





