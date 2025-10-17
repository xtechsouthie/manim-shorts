from pathlib import Path
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
from .state import VideoState

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
            fps=24,
            preset='veryfast',
            bitrate="2000k",  
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
