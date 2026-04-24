"""Standalone HiFi-GAN V1 generator — no SpeechBrain dependency.

Architecture inferred from hifigan-asc.pth checkpoint:
  in_channels=80, upsample_initial_channel=512,
  upsample_factors=[8,8,2,2], upsample_kernel_sizes=[16,16,4,4],
  resblock_kernel_sizes=[3,7,11], resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm


def _get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=d,
                                  padding=_get_padding(kernel_size, d)))
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=1,
                                  padding=_get_padding(kernel_size, 1)))
            for _ in dilations
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class HifiGANGenerator(nn.Module):
    def __init__(
        self,
        in_channels=80,
        upsample_initial_channel=512,
        upsample_factors=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        inference_padding=5,
    ):
        super().__init__()
        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)

        self.conv_pre = weight_norm(
            nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)
        )

        self.ups = nn.ModuleList()
        ch = upsample_initial_channel
        for u, k in zip(upsample_factors, upsample_kernel_sizes):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(ch, ch // 2, k, stride=u,
                                   padding=(k - u) // 2)
            ))
            ch //= 2

        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for _ in upsample_factors:
            ch //= 2
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

    def forward(self, x):
        x = F.pad(x, (self.inference_padding, self.inference_padding))
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j]
                xs = rb(x) if xs is None else xs + rb(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for up in self.ups:
            remove_weight_norm(up)
        for rb in self.resblocks:
            rb.remove_weight_norm()
        remove_weight_norm(self.conv_post)

    @torch.inference_mode()
    def infer(self, mel):
        """mel: [batch, n_mels, time] → waveform: [batch, 1, time*hop]"""
        return self(mel)


def load_hifigan(path: str, device="cpu") -> HifiGANGenerator:
    """Load a HiFi-GAN generator from a checkpoint saved as {'generator': state_dict}."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["generator"] if "generator" in ck else ck
    model = HifiGANGenerator()
    model.load_state_dict(state, strict=True)
    model.remove_weight_norm()
    model.eval()
    return model.to(device)