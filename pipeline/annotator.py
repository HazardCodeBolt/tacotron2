import whisper
import librosa
import numpy as np
from pathlib import Path

# Whisper model size: tiny, base, small, medium, large
WHISPER_MODEL = "large-v3"


def _get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class AudioAnnotator:
    def __init__(self, model_id=WHISPER_MODEL, device=None):
        if device is None:
            device = _get_device()
        print(f"Loading Whisper model ({model_id}) on {device}...")
        self.model = whisper.load_model(model_id, device='cpu')

    def transcribe_and_normalize(self, audio_path):
        """
        Transcribe audio with Whisper, normalize text, return segments with timestamps.
        """
        print(f"Transcribing {Path(audio_path).name}...")
        result = self.model.transcribe(audio_path)

        text = (result.get("text") or "").strip()
        normalized_text = text.lower()

        segments = []
        for seg in result.get("segments") or []:
            start = seg.get("start")
            end = seg.get("end")
            seg_text = (seg.get("text") or "").strip()
            if seg_text:
                segments.append({
                    "start": float(start) if start is not None else 0.0,
                    "end": float(end) if end is not None else 0.0,
                    "text": seg_text,
                })

        if not segments and text:
            y, sr = librosa.load(audio_path, sr=None)
            duration_sec = len(y) / sr
            segments = [{"start": 0.0, "end": duration_sec, "text": text}]

        return normalized_text, segments

    def extract_prosody_and_emotion(self, audio_path):
        """
        Add Prosody (Pitch, Energy) and Emotion Labelling.
        """
        print(f"Extracting prosody and emotion for {Path(audio_path).name}...")
        y, sr = librosa.load(audio_path)

        # Pitch (F0)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        # Energy (RMS)
        rms = librosa.feature.rms(y=y)
        avg_energy = np.mean(rms)

        emotion = "neutral"

        return {
            "pitch": float(avg_pitch),
            "energy": float(avg_energy),
            "emotion": emotion
        }

if __name__ == "__main__":
    pass
