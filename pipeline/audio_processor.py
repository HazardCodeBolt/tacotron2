import os
import subprocess
from pathlib import Path
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip
from pydub import AudioSegment

class AudioProcessor:
    def __init__(self, output_dir="processed_audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def extract_audio(self, video_path):
        """Step 1: Get the audio file from video"""
        video_path = Path(video_path)
        audio_path = self.output_dir / f"{video_path.stem}.wav"
        
        print(f"Extracting audio from {video_path.name}...")
        video = VideoFileClip(str(video_path))
        video.audio.write_audiofile(str(audio_path), logger=None)
        return audio_path

    def standardize_audio(self, audio_path):
        """
        Step 2: Convert to 22.05 KHz sample rate
        Step 3: Standardize into Flac format
        Step 4: Convert to Mono Audio
        """
        audio_path = Path(audio_path)
        output_path = self.output_dir / f"{audio_path.stem}_standard.flac"
        
        print(f"Standardizing audio: {audio_path.name} -> 22.05kHz, Mono, FLAC")
        
        # Load audio
        audio = AudioSegment.from_file(str(audio_path))
        
        # Convert to mono
        audio = audio.set_channels(1)
        
        # Set sample rate
        audio = audio.set_frame_rate(16000)
        
        # Export as FLAC
        audio.export(str(output_path), format="flac")
        
        return output_path

if __name__ == "__main__":
    # processor = AudioProcessor()
    # test_video = "downloads/instagram/test.mp4"
    # if os.path.exists(test_video):
    #     wav = processor.extract_audio(test_video)
    #     flac = processor.standardize_audio(wav)
    pass
