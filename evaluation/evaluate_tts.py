# -*- coding: utf-8 -*-
"""
TTS Evaluation Pipeline
=======================
Preprocessing per file:
  1. Clip first/last 200 ms
  2. Remove leading/trailing silence (librosa trim)
  3. Spectral-gating denoising (noisereduce)
  4. Second silence trim after denoising

STT engines (run in parallel threads):
  - Munsit  (munsit model, Arabic)
  - Deepgram (nova-3, language=ar)

Metrics: WER and CER vs. ground-truth sentences
Output:  evaluation_results.xlsx  (one sheet per engine + summary sheet)
"""

import os
import sys
import concurrent.futures
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
import requests
import pandas as pd
from pydub import AudioSegment

sys.stdout.reconfigure(encoding="utf-8")

# ── Config ─────────────────────────────────────────────────────────────────────
EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR   = os.path.join(EVAL_DIR, "voices")
CLEAN_DIR    = os.path.join(EVAL_DIR, "voices_clean")
EXCEL_PATH   = os.path.join(EVAL_DIR, "Arabic_TTS_100_Sentences_updated.xlsx")
OUTPUT_EXCEL = os.path.join(EVAL_DIR, "evaluation_results.xlsx")

CLIP_MS  = 200
TRIM_DB  = 30

MUNSIT_URL = "https://api.cntxt.tools/audio/transcribe"
MUNSIT_KEY = ""

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen?model=nova-3&language=ar"
DEEPGRAM_KEY = ""

os.makedirs(CLEAN_DIR, exist_ok=True)


# ── Preprocessing ──────────────────────────────────────────────────────────────

def clip_edges(audio: np.ndarray, sr: int, ms: int = 200) -> np.ndarray:
    n = int(sr * ms / 1000)
    if len(audio) <= 2 * n:
        return audio
    return audio[n:-n]


def remove_silence(audio: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Spectral-gating noise reduction.
    Uses the first 0.5 s as a noise profile when the clip is long enough;
    falls back to stationary-mode estimation otherwise.
    """
    noise_len = int(sr * 0.5)
    if len(audio) > noise_len * 2:
        denoised = nr.reduce_noise(
            y=audio,
            y_noise=audio[:noise_len],
            sr=sr,
            stationary=False,
            prop_decrease=0.9,
        )
    else:
        denoised = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,
            prop_decrease=0.9,
        )
    return denoised.astype(np.float32)


def preprocess(wav_path: str, clean_wav_path: str) -> str:
    """
    Full preprocessing chain.
    Saves cleaned WAV; returns path to MP3 (needed by Munsit).
    Deepgram uses the WAV directly.
    """
    audio, sr = librosa.load(wav_path, sr=None, mono=True)

    audio = clip_edges(audio, sr, ms=CLIP_MS)
    audio = remove_silence(audio, sr, top_db=TRIM_DB)
    audio = denoise(audio, sr)
    audio = remove_silence(audio, sr, top_db=TRIM_DB)

    sf.write(clean_wav_path, audio, sr, subtype="PCM_16")

    mp3_path = clean_wav_path.replace(".wav", ".mp3")
    AudioSegment.from_wav(clean_wav_path).export(mp3_path, format="mp3", bitrate="128k")
    return mp3_path


# ── STT engines ────────────────────────────────────────────────────────────────

def transcribe_munsit(mp3_path: str) -> str:
    headers = {"Authorization": f"Bearer {MUNSIT_KEY}"}
    with open(mp3_path, "rb") as f:
        resp = requests.post(
            MUNSIT_URL,
            headers=headers,
            files={"file": (os.path.basename(mp3_path), f, "audio/mpeg")},
            data={"model": "munsit"},
            timeout=60,
        )
    resp.raise_for_status()
    body = resp.json()
    return (
        (body.get("data") or {}).get("transcription")
        or body.get("transcription")
        or body.get("text")
        or ""
    ).strip()


def transcribe_deepgram(wav_path: str) -> str:
    headers = {
        "Authorization": f"Token {DEEPGRAM_KEY}",
        "Content-Type": "audio/wav",
    }
    with open(wav_path, "rb") as f:
        resp = requests.post(DEEPGRAM_URL, headers=headers, data=f, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    try:
        return body["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
    except (KeyError, IndexError):
        return ""


# ── Metrics ────────────────────────────────────────────────────────────────────

def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            dp[j] = prev[j-1] if a[i-1] == b[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]


def compute_cer(ref: str, hyp: str) -> float:
    r, h = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


def compute_wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_excel(EXCEL_PATH)
    df.columns = ["id", "sentence", "wav_file"]
    df = df.dropna(subset=["sentence", "wav_file"])
    df["wav_file"] = df["wav_file"].astype(str).str.strip()
    df["sentence"] = df["sentence"].astype(str).str.strip()
    print(f"Loaded {len(df)} sentences.\n")

    munsit_rows   = []
    deepgram_rows = []

    for _, row in df.iterrows():
        wav_name   = row["wav_file"]
        wav_path   = os.path.join(VOICES_DIR, wav_name)
        clean_path = os.path.join(CLEAN_DIR, wav_name)
        ref_text   = row["sentence"]

        if not os.path.exists(wav_path):
            print(f"[SKIP] {wav_name} — file not found.")
            continue

        print(f"[{wav_name}] preprocessing ...", end=" ", flush=True)
        try:
            mp3_path = preprocess(wav_path, clean_path)
        except Exception as e:
            print(f"PREPROCESS ERROR: {e}")
            continue

        # Call both STT engines in parallel
        print("transcribing (Munsit + Deepgram) ...", end=" ", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fut_munsit   = pool.submit(transcribe_munsit,   mp3_path)
            fut_deepgram = pool.submit(transcribe_deepgram, clean_path)

            hyp_munsit   = ""
            hyp_deepgram = ""

            try:
                hyp_munsit = fut_munsit.result(timeout=90)
            except Exception as e:
                print(f"\n  [Munsit error] {e}", end="")

            try:
                hyp_deepgram = fut_deepgram.result(timeout=90)
            except Exception as e:
                print(f"\n  [Deepgram error] {e}", end="")

        # Munsit metrics
        m_wer = compute_wer(ref_text, hyp_munsit)   if hyp_munsit   else None
        m_cer = compute_cer(ref_text, hyp_munsit)   if hyp_munsit   else None
        # Deepgram metrics
        d_wer = compute_wer(ref_text, hyp_deepgram) if hyp_deepgram else None
        d_cer = compute_cer(ref_text, hyp_deepgram) if hyp_deepgram else None

        print(f"done")
        print(f"  REF     : {ref_text}")
        if hyp_munsit:
            print(f"  Munsit  : {hyp_munsit}  WER={m_wer:.3f} CER={m_cer:.3f}")
        if hyp_deepgram:
            print(f"  Deepgram: {hyp_deepgram}  WER={d_wer:.3f} CER={d_cer:.3f}")
        print()

        base = {"ID": row["id"], "WAV": wav_name, "Reference": ref_text}

        if hyp_munsit is not None:
            munsit_rows.append({**base,
                "Hypothesis": hyp_munsit,
                "WER": round(m_wer, 4) if m_wer is not None else "",
                "CER": round(m_cer, 4) if m_cer is not None else "",
            })

        if hyp_deepgram is not None:
            deepgram_rows.append({**base,
                "Hypothesis": hyp_deepgram,
                "WER": round(d_wer, 4) if d_wer is not None else "",
                "CER": round(d_cer, 4) if d_cer is not None else "",
            })

    # ── Build summary ──────────────────────────────────────────────────────────
    def stats(rows):
        nums_wer = [r["WER"] for r in rows if isinstance(r["WER"], float)]
        nums_cer = [r["CER"] for r in rows if isinstance(r["CER"], float)]
        return (
            round(sum(nums_wer)/len(nums_wer), 4) if nums_wer else None,
            round(sum(nums_cer)/len(nums_cer), 4) if nums_cer else None,
            len(nums_wer),
        )

    m_avg_wer, m_avg_cer, m_count   = stats(munsit_rows)
    d_avg_wer, d_avg_cer, d_count   = stats(deepgram_rows)

    print("=" * 70)
    print(f"{'Engine':<12} {'Files':>6}  {'Avg WER':>10}  {'Avg CER':>10}")
    print("-" * 70)
    if m_avg_wer is not None:
        print(f"{'Munsit':<12} {m_count:>6}  {m_avg_wer*100:>9.2f}%  {m_avg_cer*100:>9.2f}%")
    if d_avg_wer is not None:
        print(f"{'Deepgram':<12} {d_count:>6}  {d_avg_wer*100:>9.2f}%  {d_avg_cer*100:>9.2f}%")
    print("=" * 70)

    # ── Write Excel with 3 sheets ──────────────────────────────────────────────
    def df_with_summary(rows, avg_wer, avg_cer):
        d = pd.DataFrame(rows)
        summary = pd.DataFrame([{
            "ID": "AVERAGE", "WAV": "", "Reference": "", "Hypothesis": "",
            "WER": avg_wer, "CER": avg_cer,
        }])
        return pd.concat([d, summary], ignore_index=True)

    # Side-by-side comparison sheet
    if munsit_rows and deepgram_rows:
        m_df = pd.DataFrame(munsit_rows).rename(columns={
            "Hypothesis": "Munsit_HYP", "WER": "Munsit_WER", "CER": "Munsit_CER"})
        d_df = pd.DataFrame(deepgram_rows).rename(columns={
            "Hypothesis": "Deepgram_HYP", "WER": "Deepgram_WER", "CER": "Deepgram_CER"})
        cmp_df = m_df.merge(
            d_df[["ID", "Deepgram_HYP", "Deepgram_WER", "Deepgram_CER"]],
            on="ID", how="outer"
        )
        cmp_summary = pd.DataFrame([{
            "ID": "AVERAGE", "WAV": "", "Reference": "",
            "Munsit_HYP": "", "Munsit_WER": m_avg_wer, "Munsit_CER": m_avg_cer,
            "Deepgram_HYP": "", "Deepgram_WER": d_avg_wer, "Deepgram_CER": d_avg_cer,
        }])
        cmp_df = pd.concat([cmp_df, cmp_summary], ignore_index=True)

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        if munsit_rows:
            df_with_summary(munsit_rows, m_avg_wer, m_avg_cer).to_excel(
                writer, sheet_name="Munsit", index=False)
        if deepgram_rows:
            df_with_summary(deepgram_rows, d_avg_wer, d_avg_cer).to_excel(
                writer, sheet_name="Deepgram", index=False)
        if munsit_rows and deepgram_rows:
            cmp_df.to_excel(writer, sheet_name="Comparison", index=False)

    print(f"\nResults saved → {OUTPUT_EXCEL}")
    print(f"Cleaned audio  → {CLEAN_DIR}/")


if __name__ == "__main__":
    main()
