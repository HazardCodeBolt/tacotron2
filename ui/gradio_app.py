"""
Gradio TTS UI – Omani Dialect
Mirrors the dark sci-fi aesthetic of ui/templates/index.html.
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


# ── available Tacotron 2 checkpoints ─────────────────────────────────────────
MODELS = {
    "Omani Speaker": os.path.normpath(os.path.join(_REPO_ROOT, "speaker_omani_epoch_0360.pth")),
    "MSA (Fusha)":   os.path.normpath(os.path.join(_REPO_ROOT, "tacotron2_epoch_0096.pth")),
}
for _name, _path in MODELS.items():
    if not os.path.isfile(_path):
        print(f"[TTS] WARNING: checkpoint for '{_name}' not found at {_path}")

# cache: { model_name -> (taco_model, taco_config, a2m) }
_model_cache: dict = {}

tokenizer = Tokenizer()


def _get_taco_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]
    path = MODELS[model_name]
    saved_cfg, state_dict = _load_checkpoint(path)
    cfg = saved_cfg if saved_cfg is not None else Tacotron2Config()
    model = Tacotron2(cfg).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    conv = AudioMelConversions(
        num_mels=cfg.num_mels,
        sampling_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        window_size=cfg.win_length,
        hop_size=cfg.hop_length,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        min_db=cfg.min_db,
        max_scaled_abs=cfg.max_scaled_abs,
    )
    print(f"[TTS] Tacotron2 '{model_name}' loaded from {path} on {DEVICE}")
    _model_cache[model_name] = (model, cfg, conv)
    return model, cfg, conv


# pre-load default model
_get_taco_model("Omani Speaker")

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


# ── diacritics handling ───────────────────────────────────────────────────────
_HARAKAT = set('ًٌٍَُِّْٰ')

def _strip_diacritics(text: str) -> str:
    return ''.join(c for c in text if c not in _HARAKAT)


try:
    import mishkal.tashkeel as _mishkal_mod
    _mishkal_mod.TashkeelClass()  # smoke-test import
    def _diacritize(text: str) -> str:
        return _mishkal_mod.TashkeelClass().tashkeel(text).strip()
    DIACRITIZE_AVAILABLE = True
    print("[TTS] mishkal diacritizer ready")
except Exception as _e:
    DIACRITIZE_AVAILABLE = False
    def _diacritize(text: str) -> str:
        return text
    print(f"[TTS] mishkal not available ({_e}) — diacritization disabled")


_PLACEHOLDERS = {
    "Omani Speaker": "...أدخل النص ودع النموذج ينطقه ليتحول إلى واقع",
    "MSA (Fusha)":   "...أَدْخِلِ النَّصَّ وَدَعِ النَّمُوذَجَ يَنْطِقُهُ",
}


def _update_placeholder(model_name: str):
    return gr.update(placeholder=_PLACEHOLDERS.get(model_name, _PLACEHOLDERS["Omani Speaker"]))


# ── MSA audio adjustments: lower pitch (-3 semitones) + 2x speed ─────────────
def _msa_postprocess(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    wav = torch.from_numpy(audio).unsqueeze(0)  # (1, T)

    # lower pitch by 3 semitones without changing speed
    wav = torchaudio.functional.pitch_shift(wav, sample_rate, n_steps=-3)

    # 2x speed: resample as if sample_rate were doubled, output stays at sample_rate
    wav = torchaudio.functional.resample(wav, orig_freq=sample_rate, new_freq=sample_rate // 2)

    return wav.squeeze(0).numpy().astype(np.float32)


# ── post-processing ───────────────────────────────────────────────────────────
def _post_process(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    idx = np.where(np.abs(audio) > 5e-3)[0]
    if idx.size:
        audio = audio[idx[0]: idx[-1] + 1]

    trim = min(int(200 * sample_rate / 1000), len(audio) // 4)
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
def synthesize(text: str, model_name: str, do_diacritize: bool):
    text = (text or "").strip()
    if not text:
        return None, "", "⚠ No signal — enter text first"

    try:
        taco_model, taco_config, a2m = _get_taco_model(model_name)

        if do_diacritize:
            text = _diacritize(text)
        elif model_name == "Omani Speaker":
            text = _strip_diacritics(text)

        tokens = tokenizer.encode(text).unsqueeze(0).to(DEVICE)
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

        if model_name == "MSA (Fusha)":
            audio_f32 = _msa_postprocess(audio_f32, taco_config.sample_rate)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio_f32, taco_config.sample_rate)

        dur = len(audio_f32) / taco_config.sample_rate
        status = f"◈ Done · {model_name} · {_VOCODER} · {dur:.1f}s"
        return tmp.name, text, status

    except Exception as exc:
        return None, "", f"⚠ Synthesis failed: {exc}"


# ── theme + CSS ───────────────────────────────────────────────────────────────
_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#f3f4f6",
    body_text_color="#111827",
    background_fill_primary="#ffffff",
    background_fill_secondary="#f9fafb",
    border_color_primary="#e5e7eb",
    block_border_width="1px",
    block_radius="8px",
    block_shadow="0 1px 3px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.04)",
    block_label_text_size="13px",
    block_label_text_weight="600",
    block_label_text_color="#111827",
    input_background_fill="#1f2937",
    input_border_color="#e5e7eb",
    input_border_width="1px",
    input_radius="8px",
    input_text_size="15px",
    button_primary_background_fill="#1a56db",
    button_primary_background_fill_hover="#1648c0",
    button_primary_text_color="#ffffff",
    button_primary_border_color="transparent",
    button_large_radius="8px",
    button_large_text_size="14px",
    button_large_text_weight="600",
    checkbox_background_color="#ffffff",
    checkbox_border_color="#e5e7eb",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Arabic:wght@400;500;600&display=swap');

#tts-header {
    text-align: center;
    padding: 48px 16px 32px;
    background: #111827;
    border-bottom: 1px solid #1f2937;
    margin-bottom: 32px;
}
/* strip white box Gradio puts around gr.HTML */
.gradio-container .prose,
.gradio-container .prose > div,
.gradio-container div:has(> #tts-header) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
#tts-title {
    font-family: 'Inter', sans-serif;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #ffffff;
    margin: 0 0 6px 0;
    line-height: 1.2;
}
#tts-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #60a5fa;
    margin: 0 0 14px 0;
}
#tts-desc {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: #9ca3af;
    margin: 0;
}

#tts-panel {
    max-width: 720px !important;
    margin: 0 auto 48px !important;
}

textarea, input[type="text"] {
    font-family: 'Noto Sans Arabic', 'Inter', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.75 !important;
    color: #ffffff !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #1a56db !important;
    box-shadow: 0 0 0 3px rgba(26,86,219,0.12) !important;
}
textarea::placeholder { color: #9ca3af !important; }

#btn-synthesize {
    height: 44px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08) !important;
}

#status-box textarea {
    background: #f9fafb !important;
    border-color: #e5e7eb !important;
    color: #6b7280 !important;
    font-size: 12px !important;
    font-family: 'Inter', sans-serif !important;
}

#diac-box textarea {
    color: #374151 !important;
    direction: rtl !important;
    text-align: right !important;
}

#tts-footer {
    text-align: center;
    color: #9ca3af;
    font-size: 12px;
    padding-bottom: 32px;
    font-family: 'Inter', sans-serif;
}

footer { display: none !important; }
"""

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Arabic TTS — Omani Dialect", theme=_THEME, css=CSS) as demo:

    gr.HTML(f"""
    <div id="tts-header">
        <p id="tts-subtitle">Neural Text-to-Speech</p>
        <h1 id="tts-title">Arabic TTS Engine</h1>
        <p id="tts-desc">Omani Dialect &amp; Modern Standard Arabic &nbsp;·&nbsp; {_VOCODER}</p>
    </div>
    """)

    with gr.Column(elem_id="tts-panel"):

        model_selector = gr.Radio(
            choices=list(MODELS.keys()),
            value="Omani Speaker",
            label="Voice Model",
            interactive=True,
        )

        diacritize_chk = gr.Checkbox(
            value=False,
            label="Auto-Diacritize (تشكيل تلقائي)",
            interactive=DIACRITIZE_AVAILABLE,
            info="Automatically add harakat before synthesis" if DIACRITIZE_AVAILABLE else "mishkal library not available",
        )

        text_in = gr.Textbox(
            lines=4,
            max_lines=8,
            placeholder="...أدخل النص ودع النموذج ينطقه ليتحول إلى واقع",
            label="Input Text",
            rtl=True,
        )

        synth_btn = gr.Button("Synthesize Speech", elem_id="btn-synthesize", variant="primary")

        audio_out = gr.Audio(label="Synthesized Audio", interactive=False)

        diac_out = gr.Textbox(
            label="Text as Sent to Model",
            interactive=False,
            rtl=True,
            elem_id="diac-box",
            max_lines=3,
        )

        status_out = gr.Textbox(
            value="Ready",
            label="Status",
            interactive=False,
            elem_id="status-box",
            max_lines=1,
        )

    gr.HTML('<div id="tts-footer">Tacotron 2 + HiFi-GAN &nbsp;·&nbsp; Omani Arabic Speech Synthesis</div>')

    model_selector.change(
        fn=_update_placeholder,
        inputs=[model_selector],
        outputs=[text_in],
    )

    synth_btn.click(
        fn=synthesize,
        inputs=[text_in, model_selector, diacritize_chk],
        outputs=[audio_out, diac_out, status_out],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
