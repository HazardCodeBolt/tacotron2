# Pipeline Design: Instagram Voice Dataset Miner

This document outlines the architecture and data flow for the automated pipeline designed to extract, process, and label voice data from Instagram accounts for voice model training.

## System Architecture

The pipeline is structured as a series of modular components, each responsible for a specific stage of the 16-step process. This modularity ensures that each step can be individually tested, updated, or replaced without affecting the entire system.

| Component | Responsibility | Tools/Libraries |
| :--- | :--- | :--- |
| **Ingestor** | Fetches videos and metadata from specified Instagram accounts. | `instaloader` |
| **Audio Engine** | Extracts audio, converts to FLAC, resamples to 22.05kHz, and converts to Mono. | `moviepy`, `ffmpeg` |
| **Source Separator** | Removes background music and isolates vocals from the audio stream. | `demucs` |
| **Diarizer** | Identifies distinct speakers and segments the audio by speaker identity. | `pyannote.audio` |
| **Refiner** | Removes ambient noise and trims silence from the segmented audio. | `noisereduce`, `pydub` |
| **Transcriber** | Generates text transcriptions and provides precise timing for speech. | `whisper` |
| **Analyzer** | Extracts prosodic features (pitch, energy) and labels emotional content. | `librosa`, `transformers` |
| **Exporter** | Normalizes text and packages the final dataset for training. | `pandas`, `json` |

## Data Flow Diagram

The data flows linearly through the system, with each step enriching the metadata or refining the audio content.

1.  **Input**: Instagram Username.
2.  **Download**: Video files (.mp4) are saved to a temporary workspace.
3.  **Extraction**: Audio is pulled from videos and converted to the target format (22.05kHz, Mono, FLAC).
4.  **Cleaning**: `demucs` removes music, and `noisereduce` handles environmental noise.
5.  **Segmentation**: `pyannote` identifies speakers, and the audio is split into individual files per utterance.
6.  **Labeling**: `whisper` transcribes the speech, while `librosa` and emotion models add metadata.
7.  **Output**: A structured dataset folder containing cleaned audio files and a `metadata.csv` file.

## Technical Considerations

The pipeline will prioritize **accuracy** over speed, utilizing high-fidelity models like Whisper and Demucs. To handle the computational load, the system will process videos in batches. All intermediate files will be managed in a structured directory hierarchy to prevent data loss during processing.
