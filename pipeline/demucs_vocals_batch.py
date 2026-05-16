"""
Batch Demucs vocals extraction for FLAC files in a folder.
Usage: python demucs_vocals_batch.py <input_folder> [--output OUTPUT] [--recursive]

Output: All vocals FLAC files are saved in a single flat folder (no nested per-file subfolders).
"""
import tempfile
from pathlib import Path

import librosa
import soundfile as sf

from cleaner import extract_vocals_demucs, _get_device


def extract_vocals_to_flat_flac(audio_path, output_folder, device=None):
    """
    Extract vocals using Demucs and save as FLAC in a single flat folder.
    Unlike extract_vocals_demucs, all outputs go directly to output_folder as
    <stem>_vocals.flac (no nested demucs_vocals/htdemucs/<stem>/ subfolders).
    """
    audio_path = Path(audio_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    flat_flac = output_folder / f"{audio_path.stem}_vocals.flac"
    if flat_flac.exists():
        return str(flat_flac)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir)
        vocals_wav = extract_vocals_demucs(
            audio_path,
            tmp_out,
            device=device,
        )
        vocals_wav = Path(vocals_wav)
        if not vocals_wav.exists() or vocals_wav == audio_path:
            return str(audio_path)
        data, sr = librosa.load(str(vocals_wav), sr=None, mono=True)
        sf.write(str(flat_flac), data, sr)
    return str(flat_flac)


def get_flac_files(folder, recursive=False):
    """Return list of FLAC file paths from folder."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    if recursive:
        return sorted(folder.rglob("*.flac"))
    return sorted(folder.glob("*.flac"))


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run Demucs on all FLAC files in a folder and extract vocals only."
    )
    parser.add_argument(
        "input_folder",
        default="cleaned_audio",
        nargs="?",
        help="Folder containing FLAC files (default: cleaned_audio)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output folder for vocals (default: <input_folder>/vocals_only)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process FLAC files in subfolders too",
    )
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output) if args.output else input_folder / "vocals_only"
    output_folder.mkdir(parents=True, exist_ok=True)

    flac_files = get_flac_files(input_folder, recursive=args.recursive)
    if not flac_files:
        print(f"No FLAC files found in {input_folder}")
        return

    device = _get_device()
    print(f"Processing {len(flac_files)} FLAC files (device={device}) -> {output_folder}")

    for i, flac_path in enumerate(flac_files):
        print(f"[{i+1}/{len(flac_files)}] {flac_path.name}")
        try:
            vocals_path = extract_vocals_to_flat_flac(
                flac_path,
                output_folder,
                device=device,
            )
            print(f"  -> {Path(vocals_path).name}")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
