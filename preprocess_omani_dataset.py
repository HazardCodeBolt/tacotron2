"""
Preprocessing pipeline for dataset_new_omani:
  1. Demucs (htdemucs) — separate vocals from music/background
  2. Silence trimming  — remove leading/trailing silence with librosa
  3. Save to dataset_new_omani_processed/ mirroring the original structure

Usage:
    python preprocess_omani_dataset.py [--input DIR] [--output DIR]
                                       [--sr 22050] [--top-db 30]
                                       [--device auto]

Requires:
    pip install demucs soundfile librosa
"""

import argparse
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

DEFAULT_INPUT  = "dataset_new_omani/clean_flac"
DEFAULT_OUTPUT = "dataset_new_omani_processed/clean_flac"
DEFAULT_SR     = 22050
DEFAULT_TOP_DB = 30
DEFAULT_DEVICE = "auto"


def _resolve_device(device):
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_model(device):
    from demucs.pretrained import get_model
    model = get_model("htdemucs")
    model.eval()
    model.to(device)
    return model


def _process_file(flac_path, out_path, model, device, sr, top_db):
    import torch
    import torchaudio
    import soundfile as sf
    import librosa

    # ── 1. Load ──────────────────────────────────────────────────────────────
    wav, orig_sr = torchaudio.load(str(flac_path))   # (C, T)
    if orig_sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, orig_sr, model.samplerate)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)                        # demucs needs stereo

    # ── 2. Demucs vocal separation ────────────────────────────────────────────
    from demucs.apply import apply_model
    wav_in = wav.unsqueeze(0).to(device)              # (1, 2, T)
    with torch.no_grad():
        sources = apply_model(model, wav_in, device=device)  # (1, n_src, 2, T)

    vocal_idx = model.sources.index("vocals")
    vocals = sources[0, vocal_idx].mean(dim=0)        # mono (T,)

    if model.samplerate != sr:
        vocals = torchaudio.functional.resample(vocals, model.samplerate, sr)

    audio_np = vocals.cpu().numpy()

    # ── 3. Silence trim ───────────────────────────────────────────────────────
    trimmed, _ = librosa.effects.trim(audio_np, top_db=top_db)
    if trimmed.size == 0:
        trimmed = audio_np                            # keep original if fully silent

    # ── 4. Save ───────────────────────────────────────────────────────────────
    sf.write(str(out_path), trimmed, sr, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser(description="Demucs + silence-trim for dataset_new_omani")
    parser.add_argument("--input",   default=DEFAULT_INPUT,  help="Folder of raw .flac files")
    parser.add_argument("--output",  default=DEFAULT_OUTPUT, help="Destination folder")
    parser.add_argument("--sr",      type=int,   default=DEFAULT_SR,     help="Target sample rate")
    parser.add_argument("--top-db",  type=float, default=DEFAULT_TOP_DB, help="Silence threshold dB")
    parser.add_argument("--device",  default=DEFAULT_DEVICE, help="auto | cpu | cuda")
    args = parser.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    flac_files = sorted(in_dir.glob("*.flac"))
    if not flac_files:
        print(f"No .flac files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    device = _resolve_device(args.device)
    print(f"Input  : {in_dir}  ({len(flac_files)} files)")
    print(f"Output : {out_dir}")
    print(f"SR     : {args.sr} Hz  |  top_db={args.top_db}  |  device={device}")
    print("Loading demucs model ...")
    model = _load_model(device)
    print(f"Model samplerate: {model.samplerate} Hz\n")

    ok = skipped = errors = 0
    total = len(flac_files)

    for i, flac_path in enumerate(flac_files, 1):
        out_path = out_dir / flac_path.name
        if out_path.exists():
            print(f"[{i:4d}/{total}] -> {flac_path.name}  skipped")
            skipped += 1
            continue
        try:
            _process_file(flac_path, out_path, model, device, args.sr, args.top_db)
            print(f"[{i:4d}/{total}] ok {flac_path.name}")
            ok += 1
        except Exception as e:
            print(f"[{i:4d}/{total}] ERROR {flac_path.name}  {e}")
            errors += 1

    print(f"\nDone — ok={ok}  skipped={skipped}  errors={errors}")

    src_xlsx = in_dir.parent / "transcriptions.xlsx"
    dst_xlsx = out_dir.parent / "transcriptions.xlsx"
    if src_xlsx.exists() and not dst_xlsx.exists():
        dst_xlsx.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_xlsx, dst_xlsx)
        print(f"Copied transcriptions.xlsx -> {dst_xlsx}")


if __name__ == "__main__":
    main()
