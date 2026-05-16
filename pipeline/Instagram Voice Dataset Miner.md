# Instagram Voice Dataset Miner

This pipeline automates the collection and processing of Instagram videos to create high-quality datasets for voice model training.

## Features (The 16 Steps)

1.  **Ingestion**: Downloads videos and audio from Instagram profiles.
2.  **Resampling**: Converts all audio to 22.05 kHz.
3.  **Standardization**: Exports to high-fidelity FLAC format.
4.  **Channel Mix**: Converts stereo audio to Mono.
5.  **Diarization**: Identifies distinct speakers in the audio.
6.  **Separation**: Splits audio into speaker-specific segments.
7.  **Denoising**: Removes environmental and background noise.
8.  **Music Removal**: Isolates vocals from background music.
9.  **Trimming**: Automatically removes silent intervals.
10. **Transcription**: High-accuracy speech-to-text using Whisper.
11. **Normalization**: Cleans and formats the transcription text.
12. **Prosody Analysis**: Extracts pitch and energy metrics.
13. **Emotion Labeling**: Classifies the emotional tone of speech.
14. **Timing**: Provides precise start and end times for every segment.
15. **Export**: Generates a structured CSV/JSON dataset.

## Installation

```bash
pip install instaloader moviepy pydub noisereduce librosa transformers torch whisper pandas
```

## Usage

1. Configure your Instagram target in `pipeline.py`.
2. Run the orchestrator:
   ```bash
   python3 pipeline.py
   ```

## Output Structure

- `downloads/`: Raw video files from Instagram.
- `processed_audio/`: Intermediate standardized audio.
- `speaker_data/`: Audio split by speaker identity.
- `cleaned_audio/`: Final, denoised, and trimmed audio files.
- `voice_dataset.csv`: The final dataset with all metadata and transcriptions.
