import os
import subprocess
from pathlib import Path

def create_mock_video(output_path):
    """Creates a 5-second mock video with a sine wave audio track."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate a 5-second video with a test tone
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=5:size=640x480:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
        "-c:v", "libx264", "-c:a", "aac", "-shortest",
        str(output_path)
    ]
    subprocess.run(cmd, check=True)
    print(f"Created mock video: {output_path}")

if __name__ == "__main__":
    # Create a mock video in the downloads folder to simulate a successful download
    create_mock_video("downloads/malakofficial.1/mock_video.mp4")
