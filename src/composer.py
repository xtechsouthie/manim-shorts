from pathlib import Path
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
from .state import VideoState
import subprocess
import os
import uuid
import shutil
import re

def render_manim_scripts(state: VideoState) -> VideoState:

    print("Starting manim scripts rendering for all segments....")

    manim_dir = Path("video_files/manim_script")
    video_dir = Path("video_files/video")
    
    for segment in state.segments:
        if not segment.manim_script:
            print(f"----Skipping segment {segment.segment_id}: No script")
            continue
        
        script_path = manim_dir / f"segment_{segment.segment_id}.py"
        
        # Save the reviewed script
        with open(script_path, "w") as f:
            f.write(segment.manim_script)
        
        print(f"----Rendering segment {segment.segment_id}...")
        
        try:
            video_path = video_dir / f"segment_{segment.segment_id}.mp4"
            video_dir.mkdir(parents=True, exist_ok=True)
            
            unique_tex_dir = manim_dir / f"tex_temp_{uuid.uuid4()}"
            unique_tex_dir.mkdir(exist_ok=True)
            
            env = os.environ.copy()
            env["MANIMCE_TEX_DIR"] = str(unique_tex_dir)
            env["MANIM_DISABLE_CACHING"] = "true"
            
            # Extract class name from script
            class_match = re.search(r'class\s+(\w+)\s*\(', segment.manim_script)
            class_name = class_match.group(1) if class_match else f"Segment{segment.segment_id}"
            
            render_cmd = [
                "manim",
                str(script_path.absolute()),
                class_name,
                "-qh",
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
                print(f"----Render error for segment {segment.segment_id}: {result.stderr}")
                
                # Check default locations
                default_locations = [
                    Path("media/videos") / script_path.stem / "1080p60" / f"{class_name}.mp4",
                    Path("media/videos") / script_path.stem / "720p30" / f"{class_name}.mp4",
                ]
                
                for default_path in default_locations:
                    if default_path.exists():
                        shutil.move(str(default_path), str(video_path))
                        print(f"----Found video in default location, moved to {video_path}")
                        break
            
            if video_path.exists():
                segment.video_path = str(video_path)
                print(f"----Segment {segment.segment_id} rendered successfully")
            else:
                print(f"----ERROR: Video not created for segment {segment.segment_id}")
            
            shutil.rmtree(unique_tex_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"----Error rendering segment {segment.segment_id}: {e}")
    
    return state

def video_composer(state: VideoState) -> VideoState:
    print("Starting the Video composer, merging the final audio and video")

    try:
        merged_clips = []

        for segment in state.segments:
            if not segment.video_path or not segment.audio_path:
                print(f"----Skipping Incomplete segment {segment.segment_id}")
                continue

            if not Path(segment.video_path).exists():
                print(f"----Video for segment {segment.segment_id} not found")
                continue

            if not Path(segment.audio_path).exists():
                print(f"----Audio for segmnet {segment.segment_id} not found")
                continue

            print(f"----Merging segment {segment.segment_id}")

            video_clip = VideoFileClip(segment.video_path)
            audio_clip = AudioFileClip(segment.audio_path)

            video_with_audio = video_clip.with_audio(audio_clip)

            if abs(video_clip.duration - audio_clip.duration) > 0.1:
                if video_clip.duration < audio_clip.duration:
                    print(f"----Extending video to match audio: {audio_clip.duration:.1f}s")
                    video_with_audio = video_with_audio.with_duration(audio_clip.duration)
                else:
                    print(f"----Trimming video to match audio: {audio_clip.duration:.1f}s")
                    video_with_audio = video_with_audio.subclipped(0, audio_clip.duration)

            merged_clips.append(video_with_audio)
            print(f"----Merged the segment {segment.segment_id}, duration: {audio_clip.duration:.2f}s")

        if not merged_clips:
            raise Exception("No vaild clips to merge, All segments maybe incomplete")
        
        final_clip = concatenate_videoclips(merged_clips, method="compose")

        output_dir = Path("video_files")
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / "final_video.mp4"

        print(f"----Video file concatenated, writing final video to {str(final_path)}")
        final_clip.write_videofile(
            str(final_path),
            codec='libx264',
            audio_codec='aac',
            fps=30,
            preset='medium',
            audio_bitrate='192k',
            bitrate="3000k",  
            threads=8,
        )

        state.final_video_path = str(final_path)

        total_duration = sum(seg.audio_duration_sec for seg in state.segments if seg.audio_path)
        print(f"FINAL VIDEO CREATED: {final_path}")
        print(f"----Total duration: {total_duration:.1f}s")
        print(f"----Total segments: {len(merged_clips)}")

        for clip in merged_clips:
            clip.close()
        final_clip.close()

        return state

    except Exception as e:
        state.error = f"Final composer error: {e}"
        print(f"ERROR with final composer: {e}")
        return state
