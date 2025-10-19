from pydantic import BaseModel, Field
from typing import List, Dict, Annotated, Optional
import operator


class VideoSegment(BaseModel):
    segment_id: int
    text: str
    planned_duration: float
    audio_path: str = ""
    audio_duration_sec: float = 0.0
    animation_prompt: str = ""
    video_path: str = ""
    manim_script: str = ""

def merge_segments_reducer(existing: List[VideoSegment], new: List[VideoSegment]) -> List[VideoSegment]:

    segment_dict = {seg.segment_id: seg for seg in existing}
    
    for new_seg in new:
        if new_seg.segment_id in segment_dict:
            old_seg = segment_dict[new_seg.segment_id]
            
            if new_seg.audio_path:
                old_seg.audio_path = new_seg.audio_path
                old_seg.audio_duration_sec = new_seg.audio_duration_sec
            
            if new_seg.animation_prompt:
                old_seg.animation_prompt = new_seg.animation_prompt
            
            if new_seg.manim_script:
                old_seg.manim_script = new_seg.manim_script
            
            if new_seg.video_path:
                old_seg.video_path = new_seg.video_path
        else:
            segment_dict[new_seg.segment_id] = new_seg
    
    return sorted(segment_dict.values(), key=lambda x: x.segment_id)

class VideoState(BaseModel):
    topic: str
    full_script: str
    segments: Annotated[List[VideoSegment], merge_segments_reducer]
    final_video_path: str
    error: Annotated[Optional[str], operator.add] = None 
    current_segment_id: int
    segments_needing_regeneration: Annotated[List[VideoSegment], operator.add] = []

class ScriptSegment(BaseModel):
    segment_id: int = Field(description="The ID of the segment created")
    script: str = Field(description="The script created for the particular segement")
    duration_sec: float = Field(description="The duration of that script segment in seconds")

class ScriptOutput(BaseModel):
    """A stuctured output of the script"""
    full_script: str = Field(description="The full script for the short video")
    segments: List[ScriptSegment] = Field(description="List of script segments.")

class ManimScript(BaseModel):
    """Structured output for manim code generation"""
    class_name: str = Field(description="Name of the Manim scene class")
    completed_code: str = Field(description="Complete python manim code with imports")
    estimated_duration: float = Field(description="Estimated durations in seconds")
    animations_used: str = Field(description="List of Manim animations used")

class OutputSchema(BaseModel):
    segments: List[VideoSegment]
    error: Optional[str]





