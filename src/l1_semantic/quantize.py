"""INT8 post-training static quantisation of the wearable encoder (brief WP1).

Conv + BatchNorm + ReLU blocks are fused, weights are quantised per channel and activations
per tensor, with observers calibrated on training beats only. Only the mean head is deployed
(the variance head of the information bottleneck is a training device). The quantised
encoder runs on the CPU backend; its accuracy is an emulation of INT8 inference, not a
measurement on a microcontroller. Uses torch.ao eager-mode quantisation, deprecated upstream
but present in the pinned torch 2.14.1.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.ao import quantization as tq

from .model import Encoder


class _DeployEncoder(nn.Module):
    def __init__(self, enc: Encoder):
        super().__init__()
        self.quant, self.dequant = tq.QuantStub(), tq.DeQuantStub()
        self.body = copy.deepcopy(enc.body)
        self.mu = copy.deepcopy(enc.mu)

    def forward(self, x):
        return self.dequant(self.mu(self.body(self.quant(x))))


def quantize_encoder_int8(enc: Encoder, calib_x: torch.Tensor, batch_size: int = 1024,
                          backend: str = "x86") -> nn.Module:
    torch.backends.quantized.engine = backend
    m = _DeployEncoder(enc).eval()
    for i in range(len(m.body)):
        tq.fuse_modules(m.body[i], ["0", "1", "2"], inplace=True)
    m.qconfig = tq.get_default_qconfig(backend)
    tq.prepare(m, inplace=True)
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)  # observers use non-deterministic-flagged ops
    try:
        with torch.no_grad():
            for i in range(0, len(calib_x), batch_size):
                m(calib_x[i:i + batch_size])
        tq.convert(m, inplace=True)
    finally:
        torch.use_deterministic_algorithms(prev)
    return m


class QuantizedEncoderAdapter(nn.Module):
    """Drop-in replacement for ``Encoder`` inside SemanticCodec at evaluation time."""

    def __init__(self, qenc: nn.Module, latent_channels: int):
        super().__init__()
        self.q = qenc
        self.latent_channels = latent_channels

    def forward(self, x):
        prev = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        try:
            return self.q(x), None
        finally:
            torch.use_deterministic_algorithms(prev)
