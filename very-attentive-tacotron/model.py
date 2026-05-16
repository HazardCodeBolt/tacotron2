#!/usr/bin/env python3
"""
Very Attentive Tacotron (VAT) - PyTorch Implementation
Based on arXiv:2410.22179 "Robust and Unbounded Length Generalization in
Autoregressive Transformer-Based TTS"

Architecture notes vs. paper:
- Encoder: paper uses 2 conv-downsampling stages + 3 SA blocks (We=512, 8 heads).
  We replicate that structure faithfully below.
- Decoder: paper uses 6 blocks, 16 heads, width 1024 (reference config).
- Loss: paper trains on NLL of VQ-VAE spectrogram codes; this file exposes the
  model logits so the caller supplies the appropriate loss.  A mel-MSE fallback
  head is provided for prototyping without a VQ-VAE.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Interpolated Relative Position Biases (IRPBs)
# ─────────────────────────────────────────────────────────────────────────────

class InterpolatedRPB(nn.Module):
    """
    IRPBs — paper Eq. 3–4.

    Bucketing (paper Eq. 3):
        f(d) = d                                            d ∈ [0, B/2)
             = B/2 + log(d/(B/2)) / log(D/(B/2)) * (B/2-1) d ∈ [B/2, D)
             = B - 1                                        d ≥ D
             = -f(-d)                                       d < 0

    Interpolation uses ⌊x⌋₀ = sgn(x)*⌊|x|⌋  (round toward zero, paper Eq. 4):
        β(d) = b_{⌊η⌋₀} + (|η| - ⌊|η|⌋) * (b_{⌈η⌉₀} - b_{⌊η⌋₀})
    """
    def __init__(
        self,
        num_buckets:  int   = 16,
        max_distance: int   = 64,
        num_heads:    int   = 8,
        gaussian_init: bool = False,   # True for cross-attn (paper uses σ=15 Gaussian)
        gaussian_sigma: float = 15.0,
    ):
        super().__init__()
        self.num_buckets  = num_buckets
        self.max_distance = max_distance
        self.num_heads    = num_heads

        bias = self._make_init(num_buckets, num_heads, gaussian_init, gaussian_sigma)
        self.bias = nn.Parameter(bias)

    @staticmethod
    def _make_init(num_buckets, num_heads, gaussian_init, sigma):
        if gaussian_init:
            # Paper: Gaussian window centred at 0, max normalised to 1, then
            # log-transformed so it adds to log-softmax scores.
            # Bucket 0 (distance 0) sits at index num_buckets in the flat array.
            idx = torch.arange(num_buckets * 2).float() - num_buckets  # −B…B-1
            g   = torch.exp(-0.5 * (idx / sigma) ** 2)                  # peak at 0
            g   = torch.log(g / g.max() + 1e-8)                         # log-normalised
            return g.unsqueeze(1).expand(-1, num_heads).clone()
        else:
            # Paper: truncated-normal init for self-attention biases
            t = torch.zeros(num_buckets * 2, num_heads)
            nn.init.trunc_normal_(t, std=0.02)
            return t

    # ------------------------------------------------------------------
    def _f(self, d: torch.Tensor) -> torch.Tensor:
        """Unsigned bucket index f(d) for d ≥ 0 (paper Eq. 3)."""
        nb2 = self.num_buckets / 2.0
        log_scale  = (nb2 - 1.0) / math.log(self.max_distance / nb2 + 1e-8)
        log_bucket = nb2 + log_scale * torch.log(d / nb2 + 1e-8)
        bucket = torch.where(d < nb2, d, log_bucket)
        return torch.clamp(bucket, 0.0, self.num_buckets - 1.0)

    def _bucket_index(self, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Fractional bucket index η = f(rel_pos) with sign:
            η > 0 for positive rel_pos, η < 0 for negative rel_pos (f(d) = -f(-d)).
        """
        eta_abs = self._f(rel_pos.abs().float())
        return torch.where(rel_pos >= 0, eta_abs, -eta_abs)

    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rel_pos: (B, T_dec, T_enc)
        Returns:
            irpbs:   (B, num_heads, T_dec, T_enc)
        """
        eta = self._bucket_index(rel_pos)            # (B, T_dec, T_enc)

        # ⌊η⌋₀ = sgn(η)*⌊|η|⌋  (round toward zero, paper Eq. 4)
        sign     = torch.sign(eta)
        abs_eta  = eta.abs()
        floor_abs = torch.floor(abs_eta)
        alpha    = abs_eta - floor_abs               # interpolation weight ∈ [0,1)
        eta_lo   = sign * floor_abs                  # ⌊η⌋₀
        eta_hi   = sign * (floor_abs + 1.0)          # ⌈η⌉₀

        nb = self.num_buckets
        idx_lo = torch.clamp(eta_lo.long() + nb, 0, 2 * nb - 1)
        idx_hi = torch.clamp(eta_hi.long() + nb, 0, 2 * nb - 1)

        b_lo  = self.bias[idx_lo]                              # (B, T_dec, T_enc, H)
        b_hi  = self.bias[idx_hi]
        irpbs = b_lo + alpha.unsqueeze(-1) * (b_hi - b_lo)    # (B, T_dec, T_enc, H)
        return irpbs.permute(0, 3, 1, 2)                       # (B, H, T_dec, T_enc)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """1-D conv + GeLU, paper encoder stage building block."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → conv expects (B, C, T)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return F.gelu(self.norm(x))


class VATEncoder(nn.Module):
    """
    Paper encoder: 2 conv stages (each 3 ConvBlocks) + 3 non-causal SA blocks.
    Stage 1: We/2 channels, stride=1 (no downsampling).
    Stage 2: We   channels, stride=2 (2× downsampling in time).
    """
    def __init__(self, vocab_size: int, embed_dim: int = 512, num_heads: int = 8,
                 ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        We  = embed_dim
        We2 = We // 2

        self.embedding = nn.Embedding(vocab_size, We2, padding_idx=0)

        # Conv stage 1: 3 blocks, width We/2, stride 1
        self.conv_stage1 = nn.Sequential(
            ConvBlock(We2, We2, stride=1),
            ConvBlock(We2, We2, stride=1),
            ConvBlock(We2, We2, stride=1),
        )
        # Conv stage 2: 3 blocks, width We, stride 2 on first block
        self.conv_stage2 = nn.Sequential(
            ConvBlock(We2, We, stride=2),
            ConvBlock(We,  We, stride=1),
            ConvBlock(We,  We, stride=1),
        )

        # 3 non-causal Transformer SA blocks (pre-norm, paper style)
        sa_layer = nn.TransformerEncoderLayer(
            We, num_heads, ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True, activation='gelu'
        )
        self.sa_blocks = nn.TransformerEncoder(sa_layer, num_layers=3,
                                                norm=nn.LayerNorm(We))

    def forward(self, tokens: torch.LongTensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(tokens)              # (B, T, We/2)
        x = self.conv_stage1(x)                 # (B, T, We/2)
        x = self.conv_stage2(x)                 # (B, T//2, We)
        # Adjust padding mask for 2× downsampling
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask[:, ::2]
        return self.sa_blocks(x, src_key_padding_mask=src_key_padding_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Location-Based Attention & Alignment Layer
# ─────────────────────────────────────────────────────────────────────────────

class LocationBasedAttention(nn.Module):
    """
    Pure location-based cross-attention (paper Eq. 5):
        s_{i,j}^(k) = β^(k)(p_i - j)
    No Q/K content — position bias only.
    """
    def __init__(self, num_heads: int = 4, num_buckets: int = 16,
                 max_distance: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.irpb = InterpolatedRPB(num_buckets, max_distance, num_heads,
                                    gaussian_init=False)

    def forward(self, enc_out: torch.Tensor, align_pos: torch.Tensor) -> torch.Tensor:
        """
        enc_out:   (B, T_enc, D)
        align_pos: (B, T_dec)
        returns:   (B, T_dec, D)
        """
        T_enc   = enc_out.shape[1]
        j       = torch.arange(T_enc, device=enc_out.device).float()
        rel_pos = align_pos.unsqueeze(-1) - j.view(1, 1, T_enc)   # (B, T_dec, T_enc)
        # Mean over heads (location-only, no values projection)
        attn = F.softmax(self.irpb(rel_pos).mean(dim=1), dim=-1)   # (B, T_dec, T_enc)
        return torch.bmm(attn, enc_out)                             # (B, T_dec, D)


class AlignmentLayer(nn.Module):
    """
    Serial alignment layer (paper Sec. 3.2).

    At each step i:
        context_i = LocationAttention(enc_out, p_{i-1})
        h_i, c_i  = LSTM([x_i; context_i], h_{i-1}, c_{i-1})
        Δp_i      = softplus(linear(h_i) + bias_init)
        p_i       = p_{i-1} + Δp_i

    Paper: LSTM width 256; softplus bias initialised to −1.25 so that
    initial average Δp ≈ 0.25 (softplus(−1.25) ≈ 0.25).
    """
    SOFTPLUS_BIAS_INIT = -1.25  # paper Sec. 3.2

    def __init__(self, embed_dim: int = 512, rnn_hidden: int = 256):
        super().__init__()
        self.embed_dim  = embed_dim
        self.rnn_hidden = rnn_hidden

        self.loc_attn = LocationBasedAttention(num_heads=4)
        self.lstm     = nn.LSTMCell(embed_dim + embed_dim, rnn_hidden)

        # Project hidden → scalar delta; bias init per paper
        self.delta_proj = nn.Linear(rnn_hidden, 1)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.constant_(self.delta_proj.bias, self.SOFTPLUS_BIAS_INIT)

    def reset_state(self, batch_size: int, device: torch.device):
        self.h = torch.zeros(batch_size, self.rnn_hidden, device=device)
        self.c = torch.zeros(batch_size, self.rnn_hidden, device=device)

    def forward(
        self,
        x:             torch.Tensor,  # (B, D)  — decoder layer input at step i
        enc_out:       torch.Tensor,  # (B, T_enc, D)
        prev_align:    torch.Tensor   # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context = self.loc_attn(enc_out, prev_align.unsqueeze(1)).squeeze(1)  # (B, D)
        self.h, self.c = self.lstm(torch.cat([x, context], dim=-1), (self.h, self.c))
        delta = F.softplus(self.delta_proj(self.h)).squeeze(-1)               # (B,)
        return prev_align + delta, delta


# ─────────────────────────────────────────────────────────────────────────────
# VAT Cross-Attention
# ─────────────────────────────────────────────────────────────────────────────

class VATCrossAttention(nn.Module):
    """
    Content-based relative cross-attention with IRPBs + MDP (paper Eq. 6–7).

    Score (Eq. 6):
        s_{i,j}^(k) = (q_i^(k) · k_j^(k)) / √L + β^(k)(p_i − j)

    Maximum Distance Penalty (Eq. 7):
        β_MD^(k)(d) = β^(k)(d) − P_MD * (|d| − D)   if |d| ≥ D
                    = β^(k)(d)                         otherwise

    Equivalent to:  scores += irpbs − P_MD * clamp(|rel_pos| − D, min=0)
    """
    def __init__(
        self,
        embed_dim:    int   = 1024,
        num_heads:    int   = 16,
        d_kv:         int   = 64,
        num_buckets:  int   = 16,
        max_distance: int   = 64,
        mdp_scale:    float = 1.0,
    ):
        super().__init__()
        self.num_heads    = num_heads
        self.d_kv         = d_kv
        self.mdp_scale    = mdp_scale
        self.max_distance = max_distance

        self.q_proj   = nn.Linear(embed_dim, num_heads * d_kv, bias=False)
        self.k_proj   = nn.Linear(embed_dim, num_heads * d_kv, bias=False)
        self.v_proj   = nn.Linear(embed_dim, num_heads * d_kv, bias=False)
        self.out_proj = nn.Linear(num_heads * d_kv, embed_dim)

        # Paper: cross-attention IRPBs use Gaussian init (σ=15)
        self.irpb = InterpolatedRPB(num_buckets, max_distance, num_heads,
                                    gaussian_init=True, gaussian_sigma=15.0)

    def forward(
        self,
        q_in:      torch.Tensor,   # (B, T_dec, D)
        enc_out:   torch.Tensor,   # (B, T_enc, D)
        align_pos: torch.Tensor,   # (B, T_dec)
    ) -> torch.Tensor:
        B, T_dec, _ = q_in.shape
        T_enc = enc_out.shape[1]

        q = self.q_proj(q_in  ).view(B, T_dec, self.num_heads, self.d_kv).transpose(1, 2)
        k = self.k_proj(enc_out).view(B, T_enc, self.num_heads, self.d_kv).transpose(1, 2)
        v = self.v_proj(enc_out).view(B, T_enc, self.num_heads, self.d_kv).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_kv)

        j       = torch.arange(T_enc, device=q.device).float()
        rel_pos = align_pos.unsqueeze(-1) - j.view(1, 1, T_enc)   # (B, T_dec, T_enc)

        irpbs = self.irpb(rel_pos)                                  # (B, H, T_dec, T_enc)

        # MDP (Eq. 7): subtract P_MD*(|d|-D) for |d| ≥ D
        mdp = self.mdp_scale * torch.clamp(
            rel_pos.abs() - self.max_distance, min=0.0
        ).unsqueeze(1)                                              # (B, 1, T_dec, T_enc)

        scores = scores + irpbs - mdp

        attn = F.softmax(scores, dim=-1)
        out  = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T_dec, -1)
        return self.out_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# VAT Decoder Layer
# ─────────────────────────────────────────────────────────────────────────────

class VATDecoderLayer(nn.Module):
    """
    One decoder block (paper Sec. 3.3):
        1. Causal self-attention  (with truncated-normal RPBs, 32 buckets)
        2. Alignment block        (serial, handled outside; receives align_pos)
        3. Relative cross-attention (IRPBs + MDP, Gaussian init)
        4. Feed-forward (GELU)
    All sub-layers use pre-norm (paper style).
    """
    def __init__(self, embed_dim: int = 1024, num_heads: int = 16, d_kv: int = 64,
                 ffn_dim: int = 4096, dropout: float = 0.1):
        super().__init__()
        # Self-attention: 32 buckets for causal decoder (paper Sec. 3.3)
        self.self_attn_irpb = InterpolatedRPB(32, 64, num_heads, gaussian_init=False)
        self.self_attn_qproj = nn.Linear(embed_dim, num_heads * d_kv, bias=False)
        self.self_attn_kproj = nn.Linear(embed_dim, num_heads * d_kv, bias=False)
        self.self_attn_vproj = nn.Linear(embed_dim, num_heads * d_kv, bias=False)
        self.self_attn_out   = nn.Linear(num_heads * d_kv, embed_dim)
        self.num_heads = num_heads
        self.d_kv      = d_kv

        self.cross_attn = VATCrossAttention(embed_dim, num_heads, d_kv)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)

    def _self_attn(self, x: torch.Tensor,
                   causal_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Causal self-attention with decoder IRPBs (32 buckets, truncated-normal)."""
        B, T, _ = x.shape
        q = self.self_attn_qproj(x).view(B, T, self.num_heads, self.d_kv).transpose(1, 2)
        k = self.self_attn_kproj(x).view(B, T, self.num_heads, self.d_kv).transpose(1, 2)
        v = self.self_attn_vproj(x).view(B, T, self.num_heads, self.d_kv).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_kv)

        # Self-attention RPBs: rel_pos = i - j
        i_idx   = torch.arange(T, device=x.device).float()
        rel_pos = i_idx.unsqueeze(1) - i_idx.unsqueeze(0)          # (T, T)
        rel_pos = rel_pos.unsqueeze(0)                              # (1, T, T)
        irpbs   = self.self_attn_irpb(rel_pos)                     # (1, H, T, T)
        scores  = scores + irpbs

        if causal_mask is not None:
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(scores, dim=-1)
        out  = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.self_attn_out(out)

    def forward(
        self,
        x:           torch.Tensor,                      # (B, T_dec, D)
        enc_out:     torch.Tensor,                      # (B, T_enc, D)
        align_pos:   torch.Tensor,                      # (B, T_dec)
        causal_mask: Optional[torch.Tensor] = None,     # (T_dec, T_dec)
    ) -> torch.Tensor:
        x = self.norm1(x + self.drop(self._self_attn(x, causal_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, enc_out, align_pos)))
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full VAT Model
# ─────────────────────────────────────────────────────────────────────────────

class VeryAttentiveTacotron(nn.Module):
    """
    Very Attentive Tacotron (VAT).

    Reference config (paper):
        Encoder:  We=512, 8 heads, 2 conv stages + 3 SA blocks
        Decoder:  Wd=1024, 16 heads, 6 blocks, d_kv=64
        Alignment LSTM: width 256
        Loss: NLL of VQ-VAE spectrogram codes (caller provides code targets)

    This implementation exposes both:
        - `logits` output for VQ-code NLL training (primary)
        - `mel_out` linear projection for raw-mel MSE prototyping (secondary)

    The model outputs (logits, mel_pred, stop_pred, align_pos).
    """
    def __init__(
        self,
        vocab_size:    int   = 256,     # phoneme/character vocab
        enc_embed_dim: int   = 512,     # We (encoder width)
        enc_heads:     int   = 8,
        dec_embed_dim: int   = 1024,    # Wd (decoder width, reference config)
        dec_heads:     int   = 16,
        dec_layers:    int   = 6,
        d_kv:          int   = 64,
        ffn_mult:      int   = 4,       # ffn_dim = ffn_mult * dec_embed_dim
        n_mels:        int   = 80,      # for mel-MSE head (prototyping only)
        vq_codebook_size: int = 1024,   # VQ code vocab size per codebook
        vq_num_codebooks: int = 8,      # number of VQ codebooks
        rnn_hidden:    int   = 256,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.n_mels        = n_mels
        self.vq_num_cb     = vq_num_codebooks

        # ── Encoder ─────────────────────────────────────────────────────────
        self.encoder = VATEncoder(vocab_size, enc_embed_dim, enc_heads,
                                   ffn_dim=enc_embed_dim * 4, dropout=dropout)
        # Bridge encoder → decoder width
        self.enc_proj = nn.Linear(enc_embed_dim, dec_embed_dim) \
            if enc_embed_dim != dec_embed_dim else nn.Identity()

        # ── Decoder prenet ───────────────────────────────────────────────────
        # Takes VQ code embeddings (or mel for prototyping); heavy dropout per Tacotron 2
        self.prenet = nn.Sequential(
            nn.Linear(dec_embed_dim, dec_embed_dim // 2),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(dec_embed_dim // 2, dec_embed_dim),
            nn.ReLU(), nn.Dropout(0.5),
        )

        # Input token embeddings for VQ codes (sum of per-codebook embeddings)
        self.code_embeds = nn.ModuleList([
            nn.Embedding(vq_codebook_size + 1, dec_embed_dim, padding_idx=0)
            for _ in range(vq_num_codebooks)
        ])

        # Mel embedding for prototyping without VQ-VAE
        self.mel_embed = nn.Linear(n_mels, dec_embed_dim)

        # ── Alignment layer (serial) ─────────────────────────────────────────
        self.align_layer = AlignmentLayer(dec_embed_dim, rnn_hidden)

        # ── Decoder layers ───────────────────────────────────────────────────
        ffn_dim = dec_embed_dim * ffn_mult
        self.decoder_layers = nn.ModuleList([
            VATDecoderLayer(dec_embed_dim, dec_heads, d_kv, ffn_dim, dropout)
            for _ in range(dec_layers)
        ])

        # ── Output heads ─────────────────────────────────────────────────────
        # VQ-code logits head (primary, paper loss = NLL)
        self.code_heads = nn.ModuleList([
            nn.Linear(dec_embed_dim, vq_codebook_size)
            for _ in range(vq_num_codebooks)
        ])
        # Raw mel head (for prototyping / MSE loss)
        self.mel_out  = nn.Linear(dec_embed_dim, n_mels)
        self.stop_out = nn.Linear(dec_embed_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for emb in self.code_embeds:
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.mel_out.weight)
        nn.init.xavier_uniform_(self.stop_out.weight)

    def _pos_encoding(self, length: int, device: torch.device) -> torch.Tensor:
        D   = self.dec_embed_dim
        pos = torch.arange(length, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, D, 2, device=device).float()
                        * -(math.log(10000.0) / D))
        pe  = torch.zeros(1, length, D, device=device)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((size, size), float('-inf'), device=device), diagonal=1)

    def encode(self, text: torch.LongTensor,
               src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        enc = self.encoder(text, src_key_padding_mask)   # (B, T_enc//2, We)
        return self.enc_proj(enc)                         # (B, T_enc//2, Wd)

    # ── Shared decoder body ──────────────────────────────────────────────────
    def _decode(
        self,
        dec_input: torch.Tensor,              # (B, T_dec, Wd) — prenet output
        enc_out:   torch.Tensor,              # (B, T_enc, Wd)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs serial alignment then parallel decoder layers.
        Returns (hidden, align_pos).
        """
        B, T_dec, _ = dec_input.shape
        device = dec_input.device

        # Serial alignment
        self.align_layer.reset_state(B, device)
        cur_pos    = torch.zeros(B, device=device)
        align_list = []
        for t in range(T_dec):
            cur_pos, _ = self.align_layer(dec_input[:, t], enc_out, cur_pos)
            align_list.append(cur_pos)
        align_pos = torch.stack(align_list, dim=1)   # (B, T_dec)

        # Parallel decoder
        x  = dec_input
        cm = self._causal_mask(T_dec, device)
        for layer in self.decoder_layers:
            x = layer(x, enc_out, align_pos, causal_mask=cm)

        return x, align_pos

    # ── Teacher-forcing forward (VQ codes) ──────────────────────────────────
    def forward(
        self,
        text:       torch.LongTensor,                       # (B, T_text)
        codes:      torch.LongTensor,                       # (B, T_dec, n_codebooks)
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Primary training forward (VQ-code targets).
        Returns:
            code_logits: list of (B, T_dec, codebook_size) — one per codebook
            stop_pred:   (B, T_dec, 1)  logits
            align_pos:   (B, T_dec)
        """
        B, T_dec, _ = codes.shape
        device = text.device

        enc_out = self.encode(text, src_key_padding_mask)

        # Shifted code embeddings (go-frame = zeros)
        shifted = torch.zeros(B, 1, codes.shape[-1], dtype=torch.long, device=device)
        shifted = torch.cat([shifted, codes[:, :-1]], dim=1)   # (B, T_dec, n_cb)
        token_emb = sum(self.code_embeds[cb](shifted[:, :, cb])
                        for cb in range(self.vq_num_cb))        # (B, T_dec, Wd)

        dec_h = self.prenet(token_emb) + self._pos_encoding(T_dec, device)
        x, align_pos = self._decode(dec_h, enc_out)

        code_logits = [head(x) for head in self.code_heads]
        return code_logits, self.stop_out(x), align_pos

    # ── Mel-MSE forward (prototyping, no VQ-VAE needed) ─────────────────────
    def forward_mel(
        self,
        text: torch.LongTensor,                             # (B, T_text)
        mels: torch.Tensor,                                 # (B, T_dec, n_mels)
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prototype forward using raw mel frames (MSE loss).
        Returns: mel_pred (B,T,n_mels), stop_pred (B,T,1), align_pos (B,T)
        """
        B, T_dec, _ = mels.shape
        device = text.device

        enc_out = self.encode(text, src_key_padding_mask)

        go_frame  = torch.zeros(B, 1, self.n_mels, device=device)
        mel_input = torch.cat([go_frame, mels[:, :-1]], dim=1)
        dec_h     = self.prenet(self.mel_embed(mel_input)) + self._pos_encoding(T_dec, device)
        x, align_pos = self._decode(dec_h, enc_out)

        return self.mel_out(x), self.stop_out(x), align_pos

    # ── Autoregressive inference ─────────────────────────────────────────────
    @torch.no_grad()
    def inference(
        self,
        text:            torch.LongTensor,   # (1, T_text)
        max_steps:       int   = 1000,
        stop_threshold:  float = 0.5,
        sample_temp:     float = 0.7,        # paper Sec. 4 uses temperature 0.7
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive inference (VQ-code sampling).
        Returns: codes (T_dec, n_codebooks) LongTensor, align_trace (T_dec,)
        """
        device  = text.device
        enc_out = self.encode(text, src_key_padding_mask)

        self.align_layer.reset_state(1, device)
        cur_pos    = torch.zeros(1, device=device)
        prev_codes = torch.zeros(1, 1, self.vq_num_cb, dtype=torch.long, device=device)
        code_out, align_trace = [], []

        for _ in range(max_steps):
            token_emb = sum(self.code_embeds[cb](prev_codes[:, :, cb])
                            for cb in range(self.vq_num_cb))          # (1, 1, Wd)
            dec_h = self.prenet(token_emb) + self._pos_encoding(1, device)

            cur_pos, _ = self.align_layer(dec_h.squeeze(1), enc_out, cur_pos)
            align_trace.append(cur_pos.item())

            x  = dec_h
            ap = cur_pos.unsqueeze(1)
            for layer in self.decoder_layers:
                x = layer(x, enc_out, ap)

            # Sample one code per codebook
            step_codes = []
            for head in self.code_heads:
                logits = head(x).squeeze(1) / sample_temp   # (1, codebook_size)
                code   = torch.multinomial(logits.softmax(-1), 1)  # (1, 1)
                step_codes.append(code)
            prev_codes = torch.stack(step_codes, dim=-1)            # (1, 1, n_cb)
            code_out.append(prev_codes.squeeze(0))                  # (1, n_cb)

            stop = torch.sigmoid(self.stop_out(x)).item()
            if stop > stop_threshold:
                break

        codes_tensor = torch.cat(code_out, dim=0)                   # (T, n_cb)
        return codes_tensor, torch.tensor(align_trace)


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Small config for quick smoke test
    model = VeryAttentiveTacotron(
        vocab_size=100, enc_embed_dim=128, enc_heads=4,
        dec_embed_dim=256, dec_heads=4, dec_layers=2,
        d_kv=32, n_mels=80, vq_codebook_size=64, vq_num_codebooks=4,
        rnn_hidden=64, dropout=0.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    B, T_text, T_mel = 2, 20, 50
    text  = torch.randint(1, 100, (B, T_text))
    mels  = torch.randn(B, T_mel, 80)

    mel_pred, stop_pred, align_pos = model.forward_mel(text, mels)
    assert mel_pred.shape  == (B, T_mel, 80),   mel_pred.shape
    assert stop_pred.shape == (B, T_mel, 1),    stop_pred.shape
    assert align_pos.shape == (B, T_mel),       align_pos.shape
    assert (align_pos[:, 1:] - align_pos[:, :-1] >= 0).all(), "Alignment not monotonic"

    # VQ-code forward
    codes = torch.randint(1, 64, (B, T_mel, 4))
    code_logits, stop2, ap2 = model(text, codes)
    assert len(code_logits) == 4
    assert code_logits[0].shape == (B, T_mel, 64)

    print(f"mel_pred:    {mel_pred.shape}")
    print(f"align_pos:   {align_pos.shape}  range [{align_pos.min():.2f}, {align_pos.max():.2f}]")
    print(f"code_logits: {len(code_logits)} x {code_logits[0].shape}")
    print("✓ VAT forward pass OK")
