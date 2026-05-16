"""
Voice Dataset Pipeline: process videos from a folder → extract audio, extract vocals (Demucs/GPU), standardize, split by silence into parts, transcribe each part, export XLSX/JSON.
Vocals are isolated using Demucs (GPU when available) before speaker separation. No denoise step. Final step clips audio into parts based on silence; each part gets its own row in voice_dataset.xlsx (part_name, audio_file, transcription, speaker, pitch, energy, emotion, segments).
Eval: 5% of outputs at each step copied to eval/01_extracted_audio, eval/02_vocals, eval/03_standardized, eval/04_parts.

================================================================================
CLI USAGE
================================================================================

  python pipeline.py <videos_folder> [-n MAX_VIDEOS] [--no-recursive]

  Required:
    videos_folder    Path to a folder containing video files (.mp4, .mov, .mkv, .avi, .webm).
                     By default, videos in all subfolders are included.

  Optional:
    -n, --max-videos N   Maximum number of videos to process. Default: 50. Use 0 for no limit.
    --no-recursive       Only process videos directly inside videos_folder, not in subfolders.

  Output:
    voice_dataset.xlsx   Main output: one row per audio part (part_name, audio_file, transcription, ...).
    voice_dataset.json  Same data as JSON (includes segment timestamps).
    processed_audio/     Extracted WAV, vocals, standardized FLAC.
    cleaned_audio/      Final part clips (FLAC) referenced in the dataset.
    eval/               Sample outputs at each pipeline step (5% of files).

  Examples:

    # Process up to 50 videos from dataset and all subfolders (default)
    python pipeline.py dataset

    # Process one video only (quick test)
    python pipeline.py dataset -n 1

    # Process all videos in dataset and subfolders, no limit
    python pipeline.py dataset -n 0

    # Process only videos in the given folder (ignore subfolders)
    python pipeline.py dataset --no-recursive

    # Process one video from a specific subfolder
    python pipeline.py dataset/saeed_2day -n 1 --no-recursive

================================================================================
Usage (Python):
    from pipeline import VoiceDatasetPipeline
    pipeline = VoiceDatasetPipeline()
    pipeline.run("dataset", max_videos=0, recursive=True)
"""
import os
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from audio_processor import AudioProcessor
from speaker_manager import SpeakerManager
from cleaner import AudioCleaner
from annotator import AudioAnnotator

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

# Eval: 5% of outputs per step (order matches pipeline: extract → vocals → standardize → parts by silence)
EVAL_STEP_FOLDERS = [
    "01_extracted_audio",
    "02_vocals",
    "03_standardized",
    "04_parts",
]
EVAL_FRACTION = 0.05


def _eval_save_indices(total_count, fraction=EVAL_FRACTION):
    """Indices to save for eval: fraction of total, spread across the dataset."""
    n_save = max(1, int(round(total_count * fraction)))
    return set(int(round(x)) for x in np.linspace(0, total_count - 1, n_save))


def get_videos_from_folder(folder, max_count=None, recursive=False):
    """
    Return list of video file paths from a folder.
    recursive: if True, include videos in all subfolders.
    max_count: max number of paths to return (None or 0 = no limit).
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    paths = []
    if recursive:
        for f in sorted(folder.rglob("*")):
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
                paths.append(str(f))
                if max_count and len(paths) >= max_count:
                    break
    else:
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
                paths.append(str(f))
                if max_count and len(paths) >= max_count:
                    break
    return paths


class VoiceDatasetPipeline:
    def __init__(self, config=None):
        self.config = config or {}
        self.processor = AudioProcessor()
        self.speaker_mgr = SpeakerManager()
        self.cleaner = AudioCleaner()
        self.annotator = AudioAnnotator()
        self.results = []

    def _save_eval_copy(self, src_path, step_folder, base_name):
        """Copy a pipeline output into the eval folder for that step."""
        eval_dir = Path(self.config.get("eval_dir", "eval"))
        step_dir = eval_dir / step_folder
        step_dir.mkdir(parents=True, exist_ok=True)
        src = Path(src_path)
        ext = src.suffix or ".wav"
        dest = step_dir / f"{base_name}{ext}"
        try:
            shutil.copy2(src_path, dest)
        except Exception as e:
            print(f"Eval copy failed for {step_folder}/{base_name}: {e}")

    def run(self, videos_folder, max_videos=0, recursive=True):
        """
        Run the voice dataset pipeline on videos from a local folder.
        videos_folder: path to a folder (e.g. "dataset") containing video files or subfolders with videos.
        max_videos: max number of videos to process (0 = no limit, process all).
        recursive: if True (default), include all videos in subfolders of videos_folder.
        Saves 5% of outputs at each step to eval/01_... through eval/04_parts (spread across the dataset).
        """
        folder = Path(videos_folder)
        if not folder.is_dir():
            raise ValueError(f"videos_folder is not a directory: {videos_folder}")

        limit = max_videos if max_videos and max_videos > 0 else None
        video_files = get_videos_from_folder(folder, max_count=limit, recursive=recursive)
        if not video_files:
            print(f"No video files found in {folder} (looking for {VIDEO_EXTENSIONS})")
            self.export_dataset()
            return

        # Which video indices to save for eval (5%, spread across dataset)
        save_indices = _eval_save_indices(len(video_files))
        eval_dir = Path(self.config.get("eval_dir", "eval"))
        for step in EVAL_STEP_FOLDERS:
            (eval_dir / step).mkdir(parents=True, exist_ok=True)
        print(f"--- Processing {len(video_files)} videos from {videos_folder} (eval: {len(save_indices)} per step in {eval_dir}) ---")

        for video_idx, video_path in enumerate(video_files):
            try:
                video_stem = Path(video_path).stem
                do_eval = video_idx in save_indices

                # 1. Extract audio from video (WAV)
                raw_audio = self.processor.extract_audio(video_path)
                if do_eval:
                    self._save_eval_copy(raw_audio, EVAL_STEP_FOLDERS[0], f"{video_stem}_extracted")

                # 2. Extract vocals (Demucs on raw WAV, before any FLAC conversion)
                vocals_audio = self.cleaner.extract_vocals(raw_audio)
                if do_eval:
                    self._save_eval_copy(vocals_audio, EVAL_STEP_FOLDERS[1], f"{video_stem}_vocals")

                # Copy vocals to a unique path so standardize produces one FLAC per video (Demucs output is .../vocals.wav)
                vocals_path = Path(vocals_audio)
                unique_vocals = self.processor.output_dir / f"{video_stem}_vocals.wav"
                if vocals_path.resolve() != unique_vocals.resolve():
                    shutil.copy2(str(vocals_path), str(unique_vocals))
                    vocals_audio = str(unique_vocals)

                # 3. Standardize vocals (16 kHz, mono, FLAC)
                standard_audio = self.processor.standardize_audio(vocals_audio)
                if do_eval:
                    self._save_eval_copy(standard_audio, EVAL_STEP_FOLDERS[2], f"{video_stem}_standardized")

                # 4-5. Speaker Identification & Separation (on standardized vocals)
                speaker_segments = self.speaker_mgr.identify_and_separate(standard_audio)

                for seg_idx, (speaker_id, speaker_audio) in enumerate(speaker_segments):
                    # 6. Clip to parts based on silence (no denoise)
                    part_base = f"{video_stem}_spk{speaker_id}_{seg_idx}"
                    parts = self.cleaner.split_on_silence_to_parts(speaker_audio, base_name=part_base)

                    for part_path, part_name in parts:
                        if do_eval:
                            self._save_eval_copy(part_path, EVAL_STEP_FOLDERS[3], f"{part_base}_{part_name}")

                        # 7-8. Annotation per part
                        text, segments = self.annotator.transcribe_and_normalize(part_path)
                        features = self.annotator.extract_prosody_and_emotion(part_path)

                        # Store one row per part in CSV (part_name, audio_file, transcription, ...)
                        self.results.append({
                            "video_source": video_path,
                            "audio_file": part_path,
                            "part_name": part_name,
                            "speaker": speaker_id,
                            "transcription": text,
                            "normalized_text": text.lower(),
                            "pitch": features['pitch'],
                            "energy": features['energy'],
                            "emotion": features['emotion'],
                            "segments": segments
                        })
            except Exception as e:
                print(f"Error processing {video_path}: {e}")

        # 16. Export Dataset
        self.export_dataset()

    def export_dataset(self, output_file="voice_dataset.xlsx"):
        print(f"--- Exporting Dataset to {output_file} ---")
        df = pd.DataFrame(self.results)
        df.to_excel(output_file, index=False, engine="openpyxl")
        
        # Also export as JSON for richer segment data
        with open("voice_dataset.json", "w") as f:
            json.dump(self.results, f, indent=4)
        
        print("Pipeline Complete!")

if __name__ == "__main__":
    # CLI: python pipeline.py <videos_folder> [-n N] [--no-recursive]
    # See module docstring at top of file for full usage and examples.
    import argparse
    parser = argparse.ArgumentParser(
        description="Run Voice Dataset Pipeline: extract audio from videos, isolate vocals, split by silence, transcribe (Whisper), export voice_dataset.xlsx.",
        epilog="Examples:  pipeline.py dataset -n 1   (test 1 video)   |   pipeline.py dataset -n 0   (all videos)",
    )
    parser.add_argument("videos_folder", help="Folder containing video files (.mp4, .mov, .mkv, .avi, .webm)")
    parser.add_argument("-n", "--max-videos", type=int, default=50, help="Max videos to process; 0 = no limit (default: 50)")
    parser.add_argument("--no-recursive", action="store_true", help="Do not search subfolders; only use videos in videos_folder")
    args = parser.parse_args()
    pipeline = VoiceDatasetPipeline()
    pipeline.run(
        args.videos_folder,
        max_videos=args.max_videos if args.max_videos > 0 else 0,
        recursive=not args.no_recursive,
    )
