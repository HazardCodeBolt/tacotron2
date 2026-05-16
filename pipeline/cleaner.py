import os
import shutil
import subprocess
import sys
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Demucs model name (htdemucs = default, htdemucs_ft = finer quality, slower)
DEMUCS_MODEL = "htdemucs"

# SpeechBrain MetricGAN+ runs at 16 kHz
AI_DENOISER_SR = 16000

_ai_denoiser_model = None


def _get_device():
    """Use GPU if available, else CPU."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_ai_denoiser():
    """Lazy-load SpeechBrain MetricGAN+ for AI-based denoising (GPU if available)."""
    global _ai_denoiser_model
    if _ai_denoiser_model is not None:
        return _ai_denoiser_model
    try:
        # Fix for torchaudio 2.9+ which removed list_audio_backends (SpeechBrain compatibility)
        import torchaudio
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: [""]  # type: ignore
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
        device = _get_device()
        run_opts = {"device": device}
        _ai_denoiser_model = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir="pretrained_models/metricgan-plus-voicebank",
            run_opts=run_opts,
        )
        return _ai_denoiser_model
    except Exception as e:
        print(f"AI denoiser (SpeechBrain) not available: {e}")
        return None


def _denoise_ai(audio_path, output_path):
    """
    Denoise with SpeechBrain MetricGAN+ (16 kHz model). Resamples to 16k, enhances, resamples back.
    Returns True if successful, False to fall back to noisereduce.
    """
    model = _get_ai_denoiser()
    if model is None:
        return False
    try:
        import torch
        data, rate = librosa.load(audio_path, sr=None, mono=True)
        # Resample to 16 kHz for the model
        if rate != AI_DENOISER_SR:
            data_16k = librosa.resample(data, orig_sr=rate, target_sr=AI_DENOISER_SR)
        else:
            data_16k = data
        # Model expects (batch, samples); 16k mono; move to model device
        device = next(model.parameters()).device
        noisy = torch.from_numpy(data_16k).float().unsqueeze(0).to(device)
        length_sec = noisy.shape[1] / AI_DENOISER_SR
        with torch.no_grad():
            enhanced = model.enhance_batch(noisy, lengths=torch.tensor([length_sec], device=device))
        enhanced_np = enhanced.squeeze(0).cpu().numpy()
        # Resample back to original rate if needed
        if rate != AI_DENOISER_SR:
            enhanced_np = librosa.resample(enhanced_np, orig_sr=AI_DENOISER_SR, target_sr=rate)
        sf.write(str(output_path), enhanced_np, rate)
        return True
    except Exception as e:
        print(f"AI denoise failed: {e}")
        return False


def extract_vocals_demucs(audio_path, output_dir, model=DEMUCS_MODEL, device=None):
    """
    Extract vocals only from audio using Demucs. Prefer GPU (-d cuda), fall back to CPU if GPU fails.
    Uses system TEMP dir with short paths to avoid Windows long-path and space issues.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda"
    demucs_out_root = output_dir / "demucs_vocals"
    demucs_out_root.mkdir(parents=True, exist_ok=True)
    vocals_wav = demucs_out_root / model / audio_path.stem / "vocals.wav"
    if vocals_wav.exists():
        return str(vocals_wav)

    # Use system TEMP with short names so Demucs sees no long paths or spaces (Windows-friendly)
    temp_dir = Path(os.environ.get("TEMP", os.environ.get("TMP", ".")))
    temp_dir.mkdir(parents=True, exist_ok=True)
    short_in = temp_dir / "dm_in.wav"
    short_out = temp_dir / "dm_out"
    short_out.mkdir(parents=True, exist_ok=True)

    # Always convert to WAV in TEMP (avoids FLAC/long-path issues)
    try:
        data, sr = librosa.load(str(audio_path), sr=None, mono=False)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        else:
            data = data.T
        sf.write(str(short_in), data, sr)
    except Exception as e:
        print(f"WAV conversion failed: {e}, using original path.")
        short_in = audio_path.resolve()
        short_out = demucs_out_root
        # Output will be in place
        short_out_str = str(demucs_out_root)
        copy_result = False
    else:
        short_out_str = str(short_out)
        copy_result = True

    for try_device in (device, "cpu"):
        cmd = [
            sys.executable, "-m", "demucs",
            "--two-stems", "vocals",
            "-o", short_out_str,
            "-n", model,
            "-d", try_device,
            "--float32",
            "--segment", "7",
            str(Path(short_in).resolve()),
        ]
        print(f"Extracting vocals (Demucs, device={try_device}): {audio_path.name}...")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600,
                cwd=os.getcwd(),
            )
            if result.returncode != 0:
                err = (result.stderr or "") + (result.stdout or "")
                err_lines = [l for l in err.splitlines() if "Error" in l or "error" in l or "Exception" in l or "Traceback" in l]
                err_clean = "\n".join(err_lines) if err_lines else err[-1200:]
                if err_clean:
                    print(f"Demucs error: {err_clean}")
                raise subprocess.CalledProcessError(result.returncode, cmd, output=err)
            break
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            if try_device == "cpu":
                print(f"Demucs failed ({e}), using original audio.")
                if short_in.exists() and short_in != audio_path:
                    try:
                        short_in.unlink()
                    except OSError:
                        pass
                return str(audio_path)
            print(f"Demucs GPU failed, retrying on CPU...")

    # Copy result to expected path if we used TEMP
    if copy_result:
        temp_vocals = short_out / model / "dm_in" / "vocals.wav"
        if temp_vocals.exists():
            vocals_wav.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(temp_vocals), str(vocals_wav))
            try:
                shutil.rmtree(short_out, ignore_errors=True)
                if short_in.exists():
                    short_in.unlink()
            except OSError:
                pass
    if not vocals_wav.exists():
        print(f"Demucs did not produce vocals file, using original audio.")
        return str(audio_path)
    return str(vocals_wav)


class AudioCleaner:
    def __init__(self, output_dir="cleaned_audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def remove_noise(self, audio_path):
        """Step 7: Remove noise with AI (SpeechBrain MetricGAN+); fallback to noisereduce."""
        ap = Path(audio_path)
        out_path = self.output_dir / f"denoised_{ap.name}"
        print(f"Denoising {ap.name} (AI)...")
        if _denoise_ai(audio_path, out_path):
            return str(out_path)
        print(f"Using fallback (noisereduce) for {ap.name}...")
        data, rate = librosa.load(audio_path, sr=None)
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
        sf.write(str(out_path), reduced_noise, rate)
        return str(out_path)

    def extract_vocals(self, audio_path, model=DEMUCS_MODEL):
        """Step 8: Extract vocals only (UVR-style, GPU via Demucs)."""
        return extract_vocals_demucs(
            audio_path,
            self.output_dir,
            model=model,
            device=_get_device(),
        )

    def remove_music(self, audio_path):
        """Step 8 (legacy): Same as extract_vocals – isolate vocals using Demucs."""
        return self.extract_vocals(audio_path)

    def remove_silence(self, audio_path):
        """Step 9: Remove Silence (concatenate non-silent chunks into one file)."""
        print(f"Removing silence from {Path(audio_path).name}...")
        audio = AudioSegment.from_file(audio_path)
        chunks = split_on_silence(
            audio, 
            min_silence_len=500, 
            silence_thresh=audio.dBFS-16, 
            keep_silence=100
        )
        
        combined = AudioSegment.empty()
        for chunk in chunks:
            combined += chunk
            
        out_path = self.output_dir / f"nosilence_{Path(audio_path).name}"
        combined.export(str(out_path), format="flac")
        return str(out_path)

    def split_on_silence_to_parts(self, audio_path, base_name=None, min_silence_len=500, silence_thresh_db=-16, keep_silence=100, min_part_len_ms=300):
        """
        Split audio into separate parts based on silence. Each part is saved as its own file.
        Returns list of (part_path, part_name) e.g. ("path/to/part_0.flac", "part_0").
        Parts shorter than min_part_len_ms are skipped.
        """
        audio_path = Path(audio_path)
        base_name = base_name or audio_path.stem
        print(f"Splitting {audio_path.name} on silence into parts...")
        audio = AudioSegment.from_file(audio_path)
        thresh = audio.dBFS + silence_thresh_db if silence_thresh_db else audio.dBFS - 16
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=thresh,
            keep_silence=keep_silence,
        )
        parts = []
        for i, chunk in enumerate(chunks):
            if len(chunk) < min_part_len_ms:
                continue
            part_name = f"part_{i}"
            out_path = self.output_dir / f"{base_name}_{part_name}.flac"
            chunk.export(str(out_path), format="flac")
            parts.append((str(out_path), part_name))
        print(f"  -> {len(parts)} parts")
        return parts

if __name__ == "__main__":
    pass
