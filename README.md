---
title: Omani TTS
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.29.1"
app_file: app.py
pinned: false
---

# Omani Dialect TTS Engine

A neural Text-to-Speech system for the Omani Arabic dialect, built with Tacotron 2 + HiFi-GAN.

## Usage

Enter Arabic text and click **TRANSMIT VOICE** to synthesize speech.

## Models

- **Tacotron 2** — fine-tuned on Omani dialect speech (`speaker_omani_epoch_0360.pth`)
- **HiFi-GAN** — Arabic-trained vocoder (`hifigan-asc.pth`)
- Falls back to WaveRNN or Griffin-Lim if HiFi-GAN is unavailable
