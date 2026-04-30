"""
Gradio TTS UI – Omani Dialect
Mirrors the dark sci-fi aesthetic of ui/templates/index.html.
Diacritizes input Arabic text with Mishkal before TTS synthesis.
"""

import glob
import io
import math
import os
import pathlib
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio
import gradio as gr

# ── repo-root path resolution ─────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "tacotron2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Fix Linux-saved checkpoints on Windows
if not hasattr(pathlib, "PosixPath") or pathlib.PosixPath is not pathlib.WindowsPath:
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[attr-defined]

from commons.dataset import AudioMelConversions, denormalize, normalize
from commons.hyperparams import Tacotron2Config, WaveRNNConfig
from model import Tacotron2
from tokenizer import Tokenizer
from wavernn.wavernn import WaveRNN
from wavernn.hifigan import load_hifigan

# ── device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Mishkal diacritizer ───────────────────────────────────────────────────────
import mishkal.tashkeel as _mt
import threading

_diac_local = threading.local()

def add_diacritics(text: str) -> str:
    if not hasattr(_diac_local, "instance"):
        _diac_local.instance = _mt.TashkeelClass()
    return _diac_local.instance.tashkeel(text).strip()

# ── helpers ───────────────────────────────────────────────────────────────────
def _load_checkpoint(path: str):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(120)
    if size < 20_000:
        if b"git-lfs.github.com" in head or b"version https://git-lfs" in head:
            raise RuntimeError("Git LFS pointer — run `git lfs pull`.")
        raise RuntimeError(f"Checkpoint only {size} bytes — too small.")
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location="cpu")
    if isinstance(ck, dict) and "model_state_dict" in ck:
        return ck.get("config"), ck["model_state_dict"]
    return None, ck


def _build_wavernn(cfg: WaveRNNConfig) -> WaveRNN:
    return WaveRNN(
        upsample_scales=list(cfg.upsample_scales),
        n_classes=cfg.n_classes,
        hop_length=cfg.hop_length,
        n_res_block=cfg.n_res_block,
        n_rnn=cfg.n_rnn,
        n_fc=cfg.n_fc,
        kernel_size=cfg.kernel_size,
        n_freq=cfg.n_mels,
        n_hidden=cfg.n_hidden,
        n_output=cfg.n_output,
    )


# ── load Tacotron 2 ───────────────────────────────────────────────────────────
_CKPT_CANDIDATES = [
    os.path.join(_REPO_ROOT, "tacotron2_epoch_0096.pth"),
    os.path.join(_REPO_ROOT, "checkpoints_omani", "B", "speaker_B_last.pth"),
    os.path.join(_REPO_ROOT, "speaker_B_last.pth"),
    os.path.join(_REPO_ROOT, "..", "speaker_B_last.pth"),
]
_TACOTRON_PATH = next(
    (os.path.normpath(p) for p in _CKPT_CANDIDATES if os.path.isfile(p)), None
)
if _TACOTRON_PATH is None:
    raise FileNotFoundError(
        "Tacotron2 checkpoint not found. Tried:\n"
        + "\n".join(f"  {os.path.normpath(p)}" for p in _CKPT_CANDIDATES)
    )

saved_cfg, state_dict = _load_checkpoint(_TACOTRON_PATH)
taco_config = saved_cfg if saved_cfg is not None else Tacotron2Config()
taco_model = Tacotron2(taco_config).to(DEVICE)
taco_model.load_state_dict(state_dict, strict=True)
taco_model.eval()
print(f"[TTS] Tacotron2 loaded from {_TACOTRON_PATH} on {DEVICE}")

tokenizer = Tokenizer()
a2m = AudioMelConversions(
    num_mels=taco_config.num_mels,
    sampling_rate=taco_config.sample_rate,
    n_fft=taco_config.n_fft,
    window_size=taco_config.win_length,
    hop_size=taco_config.hop_length,
    fmin=taco_config.fmin,
    fmax=taco_config.fmax,
    min_db=taco_config.min_db,
    max_scaled_abs=taco_config.max_scaled_abs,
)

# ── load HiFi-GAN (preferred) ─────────────────────────────────────────────────
hifigan_model = None
_asc_path = os.path.join(_REPO_ROOT, "hifigan-asc.pth")
if os.path.isfile(_asc_path):
    try:
        hifigan_model = load_hifigan(_asc_path, device=DEVICE)
        print(f"[TTS] HiFi-GAN loaded from {_asc_path}")
    except Exception as e:
        print(f"[TTS] HiFi-GAN failed ({e}) — trying WaveRNN.")

# ── load WaveRNN (fallback) ───────────────────────────────────────────────────
wavernn_model = None
wavernn_config = WaveRNNConfig()
if hifigan_model is None:
    _WR_DIR = os.path.join(_REPO_ROOT, "wavernn_checkpoints", "checkpoints")
    _wr_candidates = [os.path.join(_WR_DIR, "wavernn_last.pt")]
    if os.path.isdir(_WR_DIR):
        _wr_candidates += list(reversed(sorted(
            glob.glob(os.path.join(_WR_DIR, "wavernn_epoch*.pt"))
        )))
    _wr_path = next((p for p in _wr_candidates if os.path.isfile(p)), None)
    if _wr_path:
        try:
            wck = torch.load(_wr_path, map_location="cpu", weights_only=False)
        except TypeError:
            wck = torch.load(_wr_path, map_location="cpu")
        wavernn_model = _build_wavernn(wavernn_config).to(DEVICE)
        wavernn_model.load_state_dict(wck["model"], strict=True)
        wavernn_model.eval()
        print(f"[TTS] WaveRNN loaded from {_wr_path}")
    else:
        print("[TTS] No WaveRNN found — Griffin-Lim fallback.")

_VOCODER = (
    "HiFi-GAN (Arabic)" if hifigan_model
    else "WaveRNN" if wavernn_model
    else "Griffin-Lim"
)
print(f"[TTS] Active vocoder: {_VOCODER}")


# ── post-processing ───────────────────────────────────────────────────────────
def _post_process(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    idx = np.where(np.abs(audio) > 5e-3)[0]
    if idx.size:
        audio = audio[idx[0]: idx[-1] + 1]

    trim = min(int(30 * sample_rate / 1000), len(audio) // 4)
    if trim > 0:
        audio = audio[trim: len(audio) - trim]
    if len(audio) == 0:
        return audio.astype(np.float32)

    audio = np.tanh(audio)

    fade = min(int(0.010 * sample_rate), len(audio) // 4)
    if fade > 0:
        audio[:fade]  *= np.linspace(0.0, 1.0, fade)
        audio[-fade:] *= np.linspace(1.0, 0.0, fade)

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = 0.95 * (audio / peak)

    return audio.astype(np.float32)


# ── main synthesis function ───────────────────────────────────────────────────
def synthesize(text: str):
    text = (text or "").strip()
    if not text:
        return None, "", "⚠ No signal — enter text first"

    try:
        # diacritize
        diacritized = add_diacritics(text)

        tokens = tokenizer.encode(diacritized).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            mel_post, _ = taco_model.inference(tokens, max_decode_steps=2000)

        if hifigan_model is not None:
            mel_db = denormalize(
                mel_post[0].T.float().cpu(),
                min_db=taco_config.min_db,
                max_abs_val=taco_config.max_scaled_abs,
            )
            mel_ln = mel_db * (math.log(10) / 20)
            with torch.inference_mode():
                wav = hifigan_model.infer(mel_ln.unsqueeze(0).to(DEVICE))
            audio_f32 = wav.squeeze().cpu().numpy().astype(np.float32)

        elif wavernn_model is not None:
            mel_tac = mel_post[0].T.float().cpu()
            mel_db = denormalize(mel_tac, min_db=taco_config.min_db, max_abs_val=taco_config.max_scaled_abs)
            mel_wr = normalize(mel_db, min_db=wavernn_config.min_db, max_abs_val=wavernn_config.max_scaled_abs).to(DEVICE)
            with torch.inference_mode():
                wav, _ = wavernn_model.infer(mel_wr.unsqueeze(0))
            audio_f32 = wav[0, 0].float().cpu().numpy()

        else:
            audio_i16 = a2m.mel2audio(mel_post[0].T.cpu(), do_denorm=True, griffin_lim_iters=60)
            audio_f32 = audio_i16.astype(np.float32) / 32768.0

        audio_f32 = _post_process(audio_f32, taco_config.sample_rate)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio_f32, taco_config.sample_rate)

        dur = len(audio_f32) / taco_config.sample_rate
        status = f"◈ Done · {_VOCODER} · {dur:.1f}s"
        return tmp.name, diacritized, status

    except Exception as exc:
        return None, "", f"⚠ Synthesis failed: {exc}"


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

:root {
    --cyan:   #00f5ff;
    --violet: #c200fb;
    --gold:   #ffd166;
    --dark:   #040811;
    --glass:  rgba(4,16,36,0.82);
    --border: rgba(0,245,255,0.18);
}

body, .gradio-container, footer { background: var(--dark) !important; }

.gradio-container {
    font-family: 'Space Mono', monospace !important;
    color: #c8f0ff !important;
    min-height: 100vh;
    position: relative;
}

.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(circle at 30% 30%, rgba(0,245,255,0.09), transparent 40%),
        radial-gradient(circle at 70% 60%, rgba(194,0,251,0.10), transparent 45%),
        radial-gradient(circle at 50% 80%, rgba(0,255,100,0.05), transparent 40%);
    filter: blur(22px);
    animation: nebula 14s ease-in-out infinite alternate;
    z-index: 0;
    pointer-events: none;
}
@keyframes nebula {
    0%   { transform: scale(1)    translate(0,0); }
    100% { transform: scale(1.18) translate(-18px,-12px); }
}

#tts-header { text-align:center; padding:32px 0 8px; position:relative; z-index:1; }
#tts-badge {
    display:inline-block; font-family:'Orbitron',monospace; font-size:10px;
    letter-spacing:4px; color:var(--cyan); border:1px solid var(--border);
    padding:4px 14px; border-radius:20px; margin-bottom:12px;
    background:rgba(0,245,255,0.05); text-transform:uppercase;
}
#tts-title {
    font-family:'Orbitron',monospace !important;
    font-size:clamp(22px,3.5vw,44px) !important; font-weight:900 !important;
    letter-spacing:5px !important; text-transform:uppercase !important;
    line-height:1 !important;
    background:linear-gradient(90deg,var(--cyan) 0%,#ffffff 40%,var(--violet) 80%,var(--gold) 100%);
    -webkit-background-clip:text !important; -webkit-text-fill-color:transparent !important;
    background-clip:text !important;
    filter:drop-shadow(0 0 24px rgba(0,245,255,0.4)) !important;
    margin:0 !important; padding:0 !important;
}
#tts-sub {
    font-size:11px; letter-spacing:3px; color:rgba(200,240,255,0.45);
    text-transform:uppercase; margin-top:8px;
}

#tts-panel {
    max-width:680px; margin:20px auto 40px;
    background:var(--glass) !important; border:1px solid var(--border) !important;
    border-radius:24px !important; padding:32px 34px 28px !important;
    backdrop-filter:blur(28px) saturate(1.4);
    box-shadow:0 0 0 1px rgba(0,245,255,0.06),0 20px 80px rgba(0,0,0,0.6),
               inset 0 1px 0 rgba(255,255,255,0.06) !important;
    position:relative; z-index:1;
}
#tts-panel::before {
    content:""; position:absolute; top:-1px; left:20%; right:20%; height:1px;
    background:linear-gradient(90deg,transparent,var(--cyan),transparent); opacity:.7;
}

label span, .label-wrap span {
    font-family:'Orbitron',monospace !important; font-size:9px !important;
    letter-spacing:2.5px !important; color:rgba(0,245,255,0.6) !important;
    text-transform:uppercase !important;
}

textarea {
    background:rgba(0,8,24,0.7) !important; border:1px solid var(--border) !important;
    border-radius:14px !important; color:#d4f0ff !important;
    font-family:'Space Mono',monospace !important; font-size:13px !important;
    line-height:1.7 !important; caret-color:var(--cyan) !important;
}
textarea:focus {
    border-color:rgba(0,245,255,0.45) !important;
    box-shadow:0 0 0 3px rgba(0,245,255,0.07),0 0 18px rgba(0,245,255,0.55) !important;
    outline:none !important;
}
textarea::placeholder { color:rgba(200,230,255,0.22) !important; }

#btn-synthesize {
    background:linear-gradient(135deg,rgba(0,245,255,0.15) 0%,rgba(194,0,251,0.15) 100%) !important;
    border:1px solid rgba(0,245,255,0.35) !important; color:#fff !important;
    font-family:'Orbitron',monospace !important; font-size:13px !important;
    font-weight:700 !important; letter-spacing:3px !important;
    text-transform:uppercase !important; border-radius:14px !important;
    height:52px !important; width:100% !important;
    transition:transform .15s,box-shadow .2s,border-color .2s !important;
}
#btn-synthesize:hover {
    border-color:var(--cyan) !important;
    box-shadow:0 0 30px rgba(0,245,255,0.35),0 0 60px rgba(194,0,251,0.2) !important;
    transform:translateY(-1px) !important;
}

.gradio-audio, audio {
    background:rgba(0,5,18,0.75) !important; border:1px solid rgba(0,245,255,0.1) !important;
    border-radius:12px !important;
}

#status-box textarea {
    background:rgba(0,5,20,0.5) !important; border:1px solid rgba(0,245,255,0.07) !important;
    border-radius:8px !important; color:rgba(0,255,157,0.85) !important;
    font-family:'Orbitron',monospace !important; font-size:10px !important;
    letter-spacing:1.5px !important; text-transform:uppercase !important;
}

#diac-box textarea {
    background:rgba(0,8,24,0.5) !important; border:1px solid rgba(0,245,255,0.08) !important;
    border-radius:10px !important; color:rgba(200,240,255,0.6) !important;
    font-family:'Space Mono',monospace !important; font-size:12px !important;
    direction:rtl; text-align:right;
}

#tts-hint {
    text-align:center; color:rgba(0,245,255,0.45); font-family:'Space Mono',monospace;
    font-size:10px; letter-spacing:2px; text-transform:uppercase;
    padding-bottom:20px; position:relative; z-index:1;
}

footer { display:none !important; }
"""

JS_CORNERS = """
() => {
    const corners = [
        { cls:'c-tl', style:'top:16px;left:16px;', stroke:'#00f5ff' },
        { cls:'c-tr', style:'top:16px;right:16px;transform:scaleX(-1);', stroke:'#c200fb' },
        { cls:'c-bl', style:'bottom:36px;left:16px;transform:scaleY(-1);', stroke:'#00f5ff' },
        { cls:'c-br', style:'bottom:36px;right:16px;transform:scale(-1,-1);', stroke:'#c200fb' },
    ];
    corners.forEach(c => {
        if (document.querySelector('.'+c.cls)) return;
        const d = document.createElement('div');
        d.className = 'corner ' + c.cls;
        d.style.cssText = 'position:fixed;width:60px;height:60px;z-index:5;opacity:.5;pointer-events:none;' + c.style;
        d.innerHTML = `<svg viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%">
            <path d="M2 58 L2 2 L58 2" stroke="${c.stroke}" stroke-width="1.5" stroke-linecap="round"/>
            <circle cx="2" cy="2" r="2.5" fill="${c.stroke}"/>
        </svg>`;
        document.body.appendChild(d);
    });
}
"""

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="TTS Engine – Omani Dialect") as demo:

    gr.HTML(f"""
    <div id="tts-header">
        <div id="tts-badge">◈ String-Field Engine v0.1</div>
        <div id="tts-title">TTS Engine – Omani Dialect</div>
        <div id="tts-sub">Text-to-Speech · Neural Voice Synthesis · {_VOCODER}</div>
    </div>
    """)

    with gr.Column(elem_id="tts-panel"):

        text_in = gr.Textbox(
            lines=4,
            max_lines=8,
            placeholder="...أدخل النص ودع النموذج ينطقه ليتحول إلى واقع",
            label="▸ Input Text",
            rtl=True,
        )

        synth_btn = gr.Button("▶  TRANSMIT VOICE", elem_id="btn-synthesize", variant="primary")

        audio_out = gr.Audio(label="▸ Synthesized Audio", interactive=False)

        diac_out = gr.Textbox(
            label="▸ Diacritized Text",
            interactive=False,
            rtl=True,
            elem_id="diac-box",
            max_lines=3,
        )

        status_out = gr.Textbox(
            value="System ready — awaiting input",
            label="▸ Status",
            interactive=False,
            elem_id="status-box",
            max_lines=1,
        )

    gr.HTML('<div id="tts-hint">Enter Arabic text · click Transmit · Mishkal adds diacritics automatically</div>')

    synth_btn.click(
        fn=synthesize,
        inputs=[text_in],
        outputs=[audio_out, diac_out, status_out],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS, js=JS_CORNERS)
