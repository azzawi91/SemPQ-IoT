"""L1 semantic codec: wearable-side encoder, latent channel, gateway-side classifier.

Encoder: 1-D CNN, latent of D channels x T time steps. Rate levels are nested: level k keeps
the first k latent channels (channels ordered by importance through nested dropout during
training), so one trained encoder serves every rate level. Each kept element is uniformly
quantised to ``bits`` bits on [-clip, clip]; the payload per beat is
ceil(k * T * bits / 8) bytes plus the RR-interval bytes.

Training objective (brief WP1): task cross-entropy
  + beta   * KL( N(mu, sigma^2) || N(0, 1) )        (variational information bottleneck)
  + lambda * rate (bits under a learned factorised Gaussian entropy model,
               differentiable through additive uniform quantisation noise).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .channel import PayloadChannel


@dataclass(frozen=True)
class RateLevel:
    channels: int
    bits: int = 8


def conv_bn(cin, cout, k, s):
    return nn.Sequential(nn.Conv1d(cin, cout, k, stride=s, padding=k // 2, bias=False),
                         nn.BatchNorm1d(cout), nn.ReLU(inplace=True))


class Encoder(nn.Module):
    """Wearable-side encoder. ``widths`` and ``strides`` set the receptive field and T."""

    def __init__(self, latent_channels: int = 16, widths=(16, 24, 32, 32, 32),
                 kernels=(7, 5, 5, 3, 3), strides=(2, 2, 2, 2, 2), variational: bool = True,
                 in_channels: int = 1):
        super().__init__()
        layers, cin = [], in_channels
        for w, k, s in zip(widths, kernels, strides):
            layers.append(conv_bn(cin, w, k, s))
            cin = w
        self.body = nn.Sequential(*layers)
        self.mu = nn.Conv1d(cin, latent_channels, 1)
        self.logvar = nn.Conv1d(cin, latent_channels, 1) if variational else None
        self.latent_channels = latent_channels

    def forward(self, x):
        h = self.body(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-10, 4) if self.logvar is not None else None
        return mu, logvar


class FactorizedGaussianEntropy(nn.Module):
    """Per-channel zero-mean Gaussian prior on quantised latents; returns bits per element."""

    def __init__(self, channels: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(channels))

    def bits(self, zq: torch.Tensor, step: float) -> torch.Tensor:
        s = self.log_scale.exp()[None, :, None]
        normal = torch.distributions.Normal(0.0, 1.0)
        upper = normal.cdf((zq + step / 2) / s)
        lower = normal.cdf((zq - step / 2) / s)
        p = (upper - lower).clamp_min(1e-9)
        return -torch.log2(p)


class GatewayClassifier(nn.Module):
    def __init__(self, latent_channels: int, n_classes: int, rr_dim: int = 5, width: int = 48):
        # rr_dim RR features plus one flag telling whether the RR bytes were delivered
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2 * latent_channels, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(width, width, 3, padding=1), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Linear(2 * width + rr_dim + 1, 64), nn.ReLU(inplace=True),
                                  nn.Dropout(0.1), nn.Linear(64, n_classes))
        self.rr_dim = rr_dim

    def forward(self, z, mask, rr, rr_keep=None):
        h = self.conv(torch.cat([z * mask, mask], dim=1))
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        if rr_keep is None:
            rr_keep = torch.ones(len(rr), 1, device=rr.device)
        return self.head(torch.cat([h, rr * rr_keep, rr_keep], dim=1))


class SemanticCodec(nn.Module):
    def __init__(self, n_classes: int, rate_levels: list[RateLevel], latent_channels: int = 16,
                 clip: float = 4.0, rr_dim: int = 5, variational: bool = True, encoder_kwargs=None,
                 recon_weight: float = 0.0, window: int = 252, in_channels: int = 1):
        super().__init__()
        if max(r.channels for r in rate_levels) > latent_channels:
            raise ValueError("a rate level asks for more channels than the latent has")
        self.encoder = Encoder(latent_channels, variational=variational, in_channels=in_channels,
                               **(encoder_kwargs or {}))
        self.in_channels = in_channels
        self.entropy = FactorizedGaussianEntropy(latent_channels)
        self.classifier = GatewayClassifier(latent_channels, n_classes, rr_dim)
        self.rate_levels = list(rate_levels)
        self.clip = clip
        self.latent_channels = latent_channels
        # optional gateway-side decoder: task + reconstruction hybrid (decision 2026-10-11)
        self.recon_weight = recon_weight
        if recon_weight > 0:
            from .baselines import Decoder

            self.decoder = Decoder(latent_channels, out_len=window)

    # ---------------------------------------------------------------- latent geometry
    def latent_steps(self, window: int) -> int:
        with torch.no_grad():
            return self.encoder(torch.zeros(1, self.in_channels, window))[0].shape[-1]

    def payload_bytes(self, level: RateLevel, window: int, rr_bytes: int) -> int:
        return math.ceil(level.channels * self.latent_steps(window) * level.bits / 8) + rr_bytes

    # ---------------------------------------------------------------- quantisation
    def quantise(self, z, bits: int, train: bool):
        step = 2 * self.clip / (2 ** bits - 1)
        z = z.clamp(-self.clip, self.clip)
        if train:  # additive uniform noise: differentiable proxy for rounding
            return z + (torch.rand_like(z) - 0.5) * step, step
        return torch.round((z + self.clip) / step) * step - self.clip, step

    # ---------------------------------------------------------------- forward
    def forward(self, x, rr, level: RateLevel, channel: PayloadChannel | None = None,
                snr_db=None, loss_p=None):
        mu, logvar = self.encoder(x)
        if self.training and logvar is not None:
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            z = mu
        zq, step = self.quantise(z, level.bits, self.training)
        keep = torch.zeros(1, self.latent_channels, 1, device=x.device)
        keep[:, :level.channels] = 1.0
        mask = keep.expand_as(zq).clone()
        rx, rr_keep = zq * mask, None
        if channel is not None:
            rx, mask, rr_keep = channel(rx, mask, level.bits, snr_db=snr_db, loss_p=loss_p)
        logits = self.classifier(rx, mask, rr, rr_keep)
        kl = torch.zeros((), device=x.device)
        if logvar is not None:
            kl_el = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
            kl = (kl_el * keep).sum(dim=(1, 2)).mean()
        rate_bits = (self.entropy.bits(zq, step) * keep).sum(dim=(1, 2)).mean()
        aux = {"kl": kl, "rate_bits": rate_bits}
        if self.recon_weight > 0 and self.training:
            aux["extra_loss"] = self.recon_weight * F.mse_loss(self.decoder(rx, mask), x[:, :1])
        return logits, aux


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def encoder_macs(enc: Encoder, window: int) -> int:
    """Multiply-accumulates of one encoder forward pass (convolutions; BN folded at deployment)."""
    macs, length = 0, window
    for block in enc.body:
        conv = block[0]
        length = (length + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
        macs += conv.in_channels * conv.out_channels * conv.kernel_size[0] * length
    macs += enc.mu.in_channels * enc.mu.out_channels * length  # only mu is computed on device
    return int(macs)


def encoder_memory_estimate(enc: Encoder, window: int, weight_bytes: int = 1, act_bytes: int = 1) -> dict:
    """Flash = deployed weights (mu head only, BN folded into conv bias); RAM = two largest
    consecutive activation buffers (ping-pong) plus the input window. INT8 by default."""
    n_w, n_b = 0, 0
    acts, length = [window], window
    for block in enc.body:
        conv = block[0]
        n_w += conv.weight.numel()
        n_b += conv.out_channels
        length = (length + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
        acts.append(conv.out_channels * length)
    n_w += enc.mu.weight.numel()
    n_b += enc.mu.out_channels
    acts.append(enc.mu.out_channels * length)
    ram = max(a + b for a, b in zip(acts[:-1], acts[1:])) * act_bytes
    return {"flash_bytes": n_w * weight_bytes + n_b * 4, "ram_bytes": int(ram),
            "deployed_params": n_w + n_b}
