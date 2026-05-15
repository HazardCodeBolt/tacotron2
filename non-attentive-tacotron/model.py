import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NATConfig:
    # Vocabulary
    vocab_size: int = 113
    pad_token_id: int = 0
    # Audio
    n_mels: int = 80
    # Encoder
    character_embed_dim: int = 512
    encoder_n_convolutions: int = 3
    encoder_kernel_size: int = 5
    encoder_embed_dim: int = 512
    encoder_dropout_p: float = 0.5
    # Duration predictor (2-layer BiLSTM)
    dur_lstm_units: int = 512
    dur_lstm_layers: int = 2
    dur_dropout_p: float = 0.1
    # Range predictor (2-layer BiLSTM)
    range_lstm_units: int = 512
    range_lstm_layers: int = 2
    range_dropout_p: float = 0.1
    range_init_scale: float = 1.0   # initial σ scale
    # Decoder
    decoder_prenet_dim: int = 256
    decoder_prenet_depth: int = 2
    decoder_prenet_dropout_p: float = 0.5
    decoder_embed_dim: int = 1024
    decoder_lstm_layers: int = 2
    decoder_dropout_p: float = 0.1
    zoneout_prob: float = 0.1
    # PostNet
    postnet_num_convs: int = 5
    postnet_n_filters: int = 512
    postnet_kernel_size: int = 5
    postnet_dropout_p: float = 0.5
    # Training
    dur_loss_weight: float = 2.0


# ---------------------------------------------------------------------------
# Shared helpers (mirrors tacotron2/model.py, kept self-contained)
# ---------------------------------------------------------------------------

class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w_init_gain="linear"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight,
                                gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear(x)


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain="linear"):
        super().__init__()
        if padding is None:
            padding = "same"
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              bias=bias)
        nn.init.xavier_uniform_(self.conv.weight,
                                gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder  (identical architecture to tacotron2/model.py Encoder)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, config: NATConfig):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.character_embed_dim,
                                       padding_idx=config.pad_token_id)
        self.convolutions = nn.ModuleList()
        for i in range(config.encoder_n_convolutions):
            in_ch = config.character_embed_dim if i == 0 else config.encoder_embed_dim
            self.convolutions.append(nn.Sequential(
                ConvNorm(in_ch, config.encoder_embed_dim,
                         kernel_size=config.encoder_kernel_size,
                         padding="same", w_init_gain="relu"),
                nn.BatchNorm1d(config.encoder_embed_dim),
                nn.ReLU(),
                nn.Dropout(config.encoder_dropout_p),
            ))
        self.lstm = nn.LSTM(
            input_size=config.encoder_embed_dim,
            hidden_size=config.encoder_embed_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor,
                input_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embeddings(x).transpose(1, 2)          # (B, E, T)
        B, C, T = x.shape
        if input_lengths is None:
            input_lengths = torch.full((B,), T, device=x.device, dtype=torch.long)
        for block in self.convolutions:
            x = block(x)
        x = x.transpose(1, 2)                           # (B, T, E)
        x = pack_padded_sequence(x, input_lengths.cpu(), batch_first=True,
                                 enforce_sorted=True)
        outputs, _ = self.lstm(x)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs                                   # (B, T, encoder_embed_dim)


# ---------------------------------------------------------------------------
# Duration Predictor  (2-layer BiLSTM → Linear → output per token)
# ---------------------------------------------------------------------------

class DurationPredictor(nn.Module):
    def __init__(self, config: NATConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.encoder_embed_dim,
            hidden_size=config.dur_lstm_units,
            num_layers=config.dur_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dur_dropout_p if config.dur_lstm_layers > 1 else 0.0,
        )
        self.proj = LinearNorm(config.dur_lstm_units * 2, 1)

    def forward(self, encoder_output: torch.Tensor,
                input_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = encoder_output.shape
        if input_lengths is not None:
            packed = pack_padded_sequence(encoder_output, input_lengths.cpu(),
                                         batch_first=True, enforce_sorted=True)
            out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(encoder_output)
        dur = self.proj(out).squeeze(-1)                # (B, T)
        return dur                                      # raw predicted durations (frames)


# ---------------------------------------------------------------------------
# Range Predictor  (2-layer BiLSTM → Softplus → σ per token)
# ---------------------------------------------------------------------------

class RangePredictor(nn.Module):
    def __init__(self, config: NATConfig):
        super().__init__()
        # Input: encoder output concatenated with duration predictions
        self.lstm = nn.LSTM(
            input_size=config.encoder_embed_dim + 1,
            hidden_size=config.range_lstm_units,
            num_layers=config.range_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.range_dropout_p if config.range_lstm_layers > 1 else 0.0,
        )
        self.proj = LinearNorm(config.range_lstm_units * 2, 1)
        self._init_scale = config.range_init_scale

    def forward(self, encoder_output: torch.Tensor,
                dur_pred: torch.Tensor,
                input_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.cat([encoder_output, dur_pred.unsqueeze(-1)], dim=-1)
        if input_lengths is not None:
            packed = pack_padded_sequence(x, input_lengths.cpu(), batch_first=True,
                                         enforce_sorted=True)
            out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)
        sigma = F.softplus(self.proj(out)).squeeze(-1) + 1e-5   # (B, T), always > 0
        return sigma


# ---------------------------------------------------------------------------
# Gaussian Upsampling  (paper Sec. 2.2, Eq. 1–3)
# No learnable parameters — fully differentiable.
# ---------------------------------------------------------------------------

class GaussianUpsampling(nn.Module):
    """
    Center position: c_i = d_i/2 + sum(d_j, j<i)
    Weight:          w_ti = N(t; c_i, σ_i²) / Σ_j N(t; c_j, σ_j²)
    Output:          u_t  = Σ_i w_ti * h_i
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                encoder_output: torch.Tensor,   # (B, T_text, E)
                durations: torch.Tensor,         # (B, T_text) float frame counts
                sigma: torch.Tensor,             # (B, T_text) range parameters
                output_lengths: Optional[torch.Tensor] = None,  # (B,) target T_mel
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T_text, E = encoder_output.shape

        # Determine output length
        if output_lengths is None:
            T_mel = int(durations.sum(dim=1).max().item())
        else:
            T_mel = int(output_lengths.max().item())

        # Center positions: c_i = cumsum(d) - d/2   shape (B, T_text)
        cumsum = torch.cumsum(durations, dim=1)             # (B, T_text)
        centers = cumsum - durations / 2.0                  # (B, T_text)

        # Frame indices  t = 0, 1, ..., T_mel-1
        t = torch.arange(T_mel, device=encoder_output.device,
                         dtype=encoder_output.dtype)        # (T_mel,)

        # (B, T_mel, T_text) — broadcast Gaussian
        t_exp = t.view(1, T_mel, 1)
        c_exp = centers.unsqueeze(1)                        # (B, 1, T_text)
        s_exp = sigma.unsqueeze(1)                          # (B, 1, T_text)
        log_weights = -0.5 * ((t_exp - c_exp) / s_exp) ** 2
        weights = torch.softmax(log_weights, dim=-1)        # (B, T_mel, T_text)

        # Weighted sum of encoder states
        upsampled = torch.bmm(weights, encoder_output)      # (B, T_mel, E)
        return upsampled, weights


# ---------------------------------------------------------------------------
# Prenet  (mirrors tacotron2/model.py Prenet — always-on dropout)
# ---------------------------------------------------------------------------

class Prenet(nn.Module):
    def __init__(self, input_dim: int, prenet_dim: int, prenet_depth: int,
                 dropout_p: float = 0.5):
        super().__init__()
        self.dropout_p = dropout_p
        dims = [input_dim] + [prenet_dim] * prenet_depth
        self.layers = nn.ModuleList([
            nn.Sequential(
                LinearNorm(i, o, bias=False, w_init_gain="relu"),
                nn.ReLU(),
            ) for i, o in zip(dims[:-1], dims[1:])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.dropout(layer(x), p=self.dropout_p, training=True)
        return x


# ---------------------------------------------------------------------------
# ZoneOut wrapper for LSTMCell
# ---------------------------------------------------------------------------

class ZoneoutLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, zoneout_prob: float = 0.1):
        super().__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.zoneout_prob = zoneout_prob

    def forward(self, x, state):
        h_prev, c_prev = state
        h_new, c_new = self.cell(x, (h_prev, c_prev))
        if self.training and self.zoneout_prob > 0:
            # ZoneOut: with prob p keep the old state
            h_mask = torch.bernoulli(torch.full_like(h_new, self.zoneout_prob))
            c_mask = torch.bernoulli(torch.full_like(c_new, self.zoneout_prob))
            h_new = h_mask * h_prev + (1 - h_mask) * h_new
            c_new = c_mask * c_prev + (1 - c_mask) * c_new
        return h_new, c_new


# ---------------------------------------------------------------------------
# Decoder  (autoregressive, no attention — driven by upsampled context)
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, config: NATConfig):
        super().__init__()
        self.config = config
        enc_dim = config.encoder_embed_dim
        dec_dim = config.decoder_embed_dim
        pre_dim = config.decoder_prenet_dim

        self.prenet = Prenet(config.n_mels, pre_dim, config.decoder_prenet_depth,
                             config.decoder_prenet_dropout_p)

        # LSTMCell 1: prenet_out + context → hidden
        self.rnn1 = ZoneoutLSTMCell(pre_dim + enc_dim, dec_dim, config.zoneout_prob)
        # LSTMCell 2: hidden + context → hidden
        self.rnn2 = ZoneoutLSTMCell(dec_dim + enc_dim, dec_dim, config.zoneout_prob)

        self.mel_proj  = LinearNorm(dec_dim + enc_dim, config.n_mels)
        self.stop_proj = LinearNorm(dec_dim + enc_dim, 1, w_init_gain="sigmoid")

    def _init_states(self, B: int, device: torch.device):
        dec_dim = self.config.decoder_embed_dim
        self.h = [torch.zeros(B, dec_dim, device=device),
                  torch.zeros(B, dec_dim, device=device)]
        self.c = [torch.zeros(B, dec_dim, device=device),
                  torch.zeros(B, dec_dim, device=device)]

    def _decode_step(self, mel_step: torch.Tensor,
                     ctx_step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pre = self.prenet(mel_step)                     # (B, pre_dim)
        rnn1_in = torch.cat([pre, ctx_step], dim=-1)
        self.h[0], self.c[0] = self.rnn1(rnn1_in, (self.h[0], self.c[0]))
        h1_drop = F.dropout(self.h[0], self.config.decoder_dropout_p, self.training)

        rnn2_in = torch.cat([h1_drop, ctx_step], dim=-1)
        self.h[1], self.c[1] = self.rnn2(rnn2_in, (self.h[1], self.c[1]))
        h2_drop = F.dropout(self.h[1], self.config.decoder_dropout_p, self.training)

        proj_in = torch.cat([h2_drop, ctx_step], dim=-1)
        mel_out  = self.mel_proj(proj_in)               # (B, n_mels)
        stop_out = self.stop_proj(proj_in).squeeze(-1)  # (B,)
        return mel_out, stop_out

    def forward(self,
                context: torch.Tensor,        # (B, T_mel, enc_dim)
                mels: torch.Tensor,           # (B, T_mel, n_mels)  teacher-forced
                dec_mask: torch.Tensor,       # (B, T_mel)  True=padding
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T_mel, _ = context.shape
        device = context.device
        self._init_states(B, device)

        # Shifted mel input: prepend BOS zero-frame, drop last
        bos = torch.zeros(B, 1, self.config.n_mels, device=device)
        mel_in = torch.cat([bos, mels[:, :-1, :]], dim=1)  # (B, T_mel, n_mels)

        mel_outs, stop_outs = [], []
        for t in range(T_mel):
            mel_out, stop_out = self._decode_step(mel_in[:, t, :], context[:, t, :])
            mel_outs.append(mel_out)
            stop_outs.append(stop_out)

        mel_outs  = torch.stack(mel_outs,  dim=1)   # (B, T_mel, n_mels)
        stop_outs = torch.stack(stop_outs, dim=1)   # (B, T_mel)

        # Mask padding positions
        mask3 = dec_mask.unsqueeze(-1)
        mel_outs  = mel_outs.masked_fill(mask3, 0.0)
        stop_outs = stop_outs.masked_fill(dec_mask, 1e3)
        return mel_outs, stop_outs

    @torch.inference_mode()
    def inference(self, context: torch.Tensor) -> torch.Tensor:
        B, T_mel, _ = context.shape
        assert B == 1
        device = context.device
        self._init_states(B, device)

        mel_prev = torch.zeros(B, self.config.n_mels, device=device)
        mel_outs = []
        for t in range(T_mel):
            mel_out, _ = self._decode_step(mel_prev, context[:, t, :])
            mel_outs.append(mel_out)
            mel_prev = mel_out

        return torch.stack(mel_outs, dim=1)             # (1, T_mel, n_mels)


# ---------------------------------------------------------------------------
# PostNet  (mirrors tacotron2/model.py PostNet)
# ---------------------------------------------------------------------------

class PostNet(nn.Module):
    def __init__(self, config: NATConfig):
        super().__init__()
        n = config.n_mels
        f = config.postnet_n_filters
        k = config.postnet_kernel_size
        p = config.postnet_dropout_p

        convs = [nn.Sequential(
            ConvNorm(n, f, kernel_size=k, padding="same", w_init_gain="tanh"),
            nn.BatchNorm1d(f), nn.Tanh(), nn.Dropout(p),
        )]
        for _ in range(config.postnet_num_convs - 2):
            convs.append(nn.Sequential(
                ConvNorm(f, f, kernel_size=k, padding="same", w_init_gain="tanh"),
                nn.BatchNorm1d(f), nn.Tanh(), nn.Dropout(p),
            ))
        convs.append(nn.Sequential(
            ConvNorm(f, n, kernel_size=k, padding="same"),
            nn.BatchNorm1d(n), nn.Dropout(p),
        ))
        self.convs = nn.ModuleList(convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_mels) → transpose → conv → transpose back
        x = x.transpose(1, 2)
        for block in self.convs:
            x = block(x)
        return x.transpose(1, 2)                        # (B, T, n_mels)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class NonAttentiveTacotron(nn.Module):
    def __init__(self, config: NATConfig):
        super().__init__()
        self.config = config
        self.encoder           = Encoder(config)
        self.duration_predictor = DurationPredictor(config)
        self.range_predictor   = RangePredictor(config)
        self.upsampler         = GaussianUpsampling()
        self.decoder           = Decoder(config)
        self.postnet           = PostNet(config)

    # ------------------------------------------------------------------
    # Training forward pass (teacher-forced)
    # ------------------------------------------------------------------
    def forward(
        self,
        text: torch.Tensor,             # (B, T_text)
        input_lengths: torch.Tensor,    # (B,)  sorted descending
        mels: torch.Tensor,             # (B, T_mel, n_mels)
        gt_durations: torch.Tensor,     # (B, T_text)  LongTensor GT frame counts
        enc_mask: torch.Tensor,         # (B, T_text)  True=padding
        dec_mask: torch.Tensor,         # (B, T_mel)   True=padding
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        enc_out     = self.encoder(text, input_lengths)             # (B, T_text, E)
        dur_pred    = self.duration_predictor(enc_out, input_lengths)  # (B, T_text)
        sigma       = self.range_predictor(enc_out, dur_pred, input_lengths)  # (B, T_text)

        # Use GT durations (float) for upsampling during training
        output_lengths = (~dec_mask).sum(dim=1)                     # (B,)
        gt_dur_float   = gt_durations.float()
        # Zero out durations on padding text positions
        gt_dur_float   = gt_dur_float.masked_fill(enc_mask, 0.0)

        context, _ = self.upsampler(enc_out, gt_dur_float, sigma, output_lengths)
        # Trim/pad context to exactly T_mel (ensures shape matches mels)
        T_mel = mels.shape[1]
        if context.shape[1] > T_mel:
            context = context[:, :T_mel, :]
        elif context.shape[1] < T_mel:
            pad = torch.zeros(context.shape[0], T_mel - context.shape[1],
                              context.shape[2], device=context.device)
            context = torch.cat([context, pad], dim=1)

        mel_out, stop_out = self.decoder(context, mels, dec_mask)
        mel_residual      = self.postnet(mel_out)
        mel_residual      = mel_residual.masked_fill(dec_mask.unsqueeze(-1), 0.0)
        mel_postnet       = mel_out + mel_residual

        return mel_out, mel_postnet, stop_out, dur_pred

    # ------------------------------------------------------------------
    # Inference (autoregressive, predicted durations)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,             # (1, T_text) or (T_text,)
        duration_scale: float = 1.0,
    ) -> torch.Tensor:
        if text.ndim == 1:
            text = text.unsqueeze(0)
        assert text.shape[0] == 1

        enc_out  = self.encoder(text)                               # (1, T_text, E)
        dur_pred = self.duration_predictor(enc_out)                 # (1, T_text)
        sigma    = self.range_predictor(enc_out, dur_pred)          # (1, T_text)

        # Convert predicted durations to integer frame counts
        durations = (dur_pred * duration_scale).clamp(min=1.0)      # (1, T_text)
        # Round to nearest integer but keep as float for upsampler
        durations_float = torch.round(durations)

        context, _ = self.upsampler(enc_out, durations_float, sigma)  # (1, T_mel, E)
        mel_out     = self.decoder.inference(context)                 # (1, T_mel, n_mels)
        mel_postnet = mel_out + self.postnet(mel_out)
        return mel_postnet                                            # (1, T_mel, n_mels)
