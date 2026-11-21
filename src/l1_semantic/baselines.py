"""WP1 baselines, all implemented (brief WP1). Every baseline exposes the same interface as
``SemanticCodec`` - ``rate_levels``, ``payload_bytes(level, window, rr_bytes)`` and
``forward(x, rr, level, channel, snr_db, loss_p) -> (logits, aux)`` - so the same trainer,
the same validation-only model selection and the same test grid apply to all methods.
All methods receive the same RR features (3 bytes), sent in the same payload stream.

* B-raw   : the beat waveform (11-bit samples, as digitised in MIT-BIH) sent uncompressed;
            CNN classifier at the gateway.
* B-DWT   : orthonormal db4 wavelet transform (4 levels, periodisation, beat padded to 256);
            the M largest-magnitude coefficients are sent as (8-bit index, 8-bit value);
            the gateway inverts the transform and applies the same CNN classifier.
* B-JSCC  : deep JSCC autoencoder trained for reconstruction (MSE) through the channel;
            the gateway classifier sees the reconstruction only (stop-gradient), so the
            encoder is not trained for the task.
* B-label : full classifier on the wearable; one byte (the label) is sent. If its fragment
            is lost the gateway falls back to the training-set class prior.

The security overhead of a TLS-like record layer is added for all methods in WP2 (L2 cost
model); WP1 compares payload bytes.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Encoder, RateLevel, conv_bn


def gateway_template_channel(rec: torch.Tensor, x: torch.Tensor) -> torch.Tensor | None:
    """rec - template for gateway-side baselines when the template feature is enabled.

    The template is taken from the clean input (x[:, 0] - x[:, 1]), i.e. as if the gateway had
    received every earlier beat without error. This is optimistic for the baselines, which
    makes the comparison with the proposed device-side template conservative.
    """
    if x.shape[1] < 2:
        return None
    return rec - (x[:, :1] - x[:, 1:2])


class WaveformClassifier(nn.Module):
    """Gateway CNN on a (possibly reconstructed or partially erased) waveform plus RR features."""

    def __init__(self, n_classes: int, rr_dim: int = 5, width: int = 32, extra_channels: int = 0):
        super().__init__()
        self.body = nn.Sequential(conv_bn(2 + extra_channels, 16, 7, 2), conv_bn(16, 24, 5, 2),
                                  conv_bn(24, width, 5, 2), conv_bn(width, width, 3, 2))
        self.head = nn.Sequential(nn.Linear(2 * width + rr_dim + 1, 64), nn.ReLU(inplace=True),
                                  nn.Dropout(0.1), nn.Linear(64, n_classes))

    def forward(self, x, mask, rr, rr_keep=None, extra=None):
        parts = [x * mask, mask] + ([extra * mask] if extra is not None else [])
        h = self.body(torch.cat(parts, dim=1))
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        if rr_keep is None:
            rr_keep = torch.ones(len(rr), 1, device=rr.device)
        return self.head(torch.cat([h, rr * rr_keep, rr_keep], dim=1))


def _zero_aux(x):
    z = torch.zeros((), device=x.device)
    return {"kl": z, "rate_bits": z}


def _quantise(v, bits: int, clip: float):
    step = 2 * clip / (2 ** bits - 1)
    v = v.clamp(-clip, clip)
    return torch.round((v + clip) / step) * step - clip


# ---------------------------------------------------------------------------- B-raw
class RawBaseline(nn.Module):
    def __init__(self, n_classes: int, sample_bits: int = 11, clip: float = 8.0, rr_dim: int = 5,
                 gateway_template: bool = False):
        super().__init__()
        self.rate_levels = [RateLevel(channels=1, bits=sample_bits)]
        self.clip = clip
        self.classifier = WaveformClassifier(n_classes, rr_dim, extra_channels=int(gateway_template))

    def payload_bytes(self, level, window, rr_bytes):
        return math.ceil(window * level.bits / 8) + rr_bytes

    def device_macs(self, window):
        return 0

    def forward(self, x, rr, level, channel=None, snr_db=None, loss_p=None):
        xfull, x = x, x[:, :1]
        xq = _quantise(x, level.bits, self.clip)
        mask, rr_keep = torch.ones_like(xq), None
        if channel is not None:
            xq, mask, rr_keep = channel(xq, mask, level.bits, snr_db=snr_db, loss_p=loss_p)
        extra = gateway_template_channel(xq, xfull)
        return self.classifier(xq, mask, rr, rr_keep, extra), _zero_aux(x)


# ---------------------------------------------------------------------------- B-DWT
def dwt_matrix(n: int = 256, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """Rows of the analysis matrix of an orthonormal periodised DWT (coeffs = x @ A.T)."""
    import pywt

    eye = np.eye(n)
    rows = [np.concatenate(pywt.wavedec(eye[i], wavelet, mode="periodization", level=level))
            for i in range(n)]
    a = np.asarray(rows).T  # (n_coeffs, n)
    if not np.allclose(a @ a.T, np.eye(a.shape[0]), atol=1e-8):
        raise ValueError("DWT matrix is not orthonormal")
    return a.astype(np.float32)


class DWTBaseline(nn.Module):
    """Top-M coefficient selection is done on the device per beat; levels are nested in M."""

    def __init__(self, n_classes: int, coeff_levels=(8, 16, 32, 64), value_bits: int = 8,
                 index_bits: int = 8, clip: float = 8.0, pad_to: int = 256, rr_dim: int = 5,
                 gateway_template: bool = False):
        super().__init__()
        self.register_buffer("A", torch.from_numpy(dwt_matrix(pad_to)))
        self.rate_levels = [RateLevel(channels=m, bits=value_bits) for m in coeff_levels]
        self.index_bits, self.clip, self.pad_to = index_bits, clip, pad_to
        self.classifier = WaveformClassifier(n_classes, rr_dim, extra_channels=int(gateway_template))

    def payload_bytes(self, level, window, rr_bytes):
        return math.ceil(level.channels * (level.bits + self.index_bits) / 8) + rr_bytes

    def device_macs(self, window):
        return int(self.A.shape[0] * self.pad_to)  # dense-matrix upper bound; a lifting DWT is cheaper

    def forward(self, x, rr, level, channel=None, snr_db=None, loss_p=None):
        xfull, x = x, x[:, :1]
        n, _, length = x.shape
        xp = F.pad(x[:, 0], (0, self.pad_to - length))
        c = xp @ self.A.T                                   # (n, 256) coefficients
        top = c.abs().topk(level.channels, dim=1).indices    # transmitted indices
        vals = _quantise(torch.gather(c, 1, top), level.bits, self.clip)
        vals = vals[:, None, :]                              # (n, 1, M) in index order
        mask, rr_keep = torch.ones_like(vals), None
        if channel is not None:  # each (index, value) pair is one element of index+value bits
            vals, mask, rr_keep = channel(vals, mask, level.bits + self.index_bits,
                                          snr_db=snr_db, loss_p=loss_p)
        c_rx = torch.zeros_like(c).scatter(1, top, vals[:, 0] * mask[:, 0])
        rec = (c_rx @ self.A)[:, None, :length]
        known = torch.ones_like(rec)  # the gateway sees a complete (approximate) waveform
        extra = gateway_template_channel(rec, xfull)
        return self.classifier(rec, known, rr, rr_keep, extra), _zero_aux(x)


# ---------------------------------------------------------------------------- B-JSCC
class Decoder(nn.Module):
    def __init__(self, latent_channels=16, widths=(32, 32, 24, 16), out_len=252):
        super().__init__()
        layers, cin = [], 2 * latent_channels
        for w in widths:
            layers += [nn.ConvTranspose1d(cin, w, 4, stride=2, padding=1), nn.BatchNorm1d(w),
                       nn.ReLU(inplace=True)]
            cin = w
        layers += [nn.ConvTranspose1d(cin, 1, 4, stride=2, padding=1)]
        self.net = nn.Sequential(*layers)
        self.out_len = out_len

    def forward(self, z, mask):
        y = self.net(torch.cat([z * mask, mask], dim=1))
        return F.interpolate(y, size=self.out_len, mode="linear", align_corners=False)


class JSCCBaseline(nn.Module):
    def __init__(self, n_classes: int, rate_levels, latent_channels: int = 16, clip: float = 4.0,
                 window: int = 252, recon_weight: float = 1.0, rr_dim: int = 5,
                 gateway_template: bool = False):
        super().__init__()
        self.encoder = Encoder(latent_channels, variational=False)
        self.decoder = Decoder(latent_channels, out_len=window)
        self.classifier = WaveformClassifier(n_classes, rr_dim, extra_channels=int(gateway_template))
        self.rate_levels = list(rate_levels)
        self.latent_channels, self.clip, self.recon_weight = latent_channels, clip, recon_weight

    def payload_bytes(self, level, window, rr_bytes):
        t = self.encoder(torch.zeros(1, 1, window))[0].shape[-1]
        return math.ceil(level.channels * t * level.bits / 8) + rr_bytes

    def forward(self, x, rr, level, channel=None, snr_db=None, loss_p=None):
        xfull, x = x, x[:, :1]
        z, _ = self.encoder(x)
        step = 2 * self.clip / (2 ** level.bits - 1)
        z = z.clamp(-self.clip, self.clip)
        zq = z + (torch.rand_like(z) - 0.5) * step if self.training else _quantise(z, level.bits, self.clip)
        keep = torch.zeros(1, self.latent_channels, 1, device=x.device)
        keep[:, :level.channels] = 1.0
        mask = keep.expand_as(zq).clone()
        rx, rr_keep = zq * mask, None
        if channel is not None:
            rx, mask, rr_keep = channel(rx, mask, level.bits, snr_db=snr_db, loss_p=loss_p)
        rec = self.decoder(rx, mask)
        extra = gateway_template_channel(rec.detach(), xfull)
        logits = self.classifier(rec.detach(), torch.ones_like(rec), rr, rr_keep, extra)
        aux = _zero_aux(x)
        aux["extra_loss"] = self.recon_weight * F.mse_loss(rec, x)
        return logits, aux


# ---------------------------------------------------------------------------- B-label
class LabelBaseline(nn.Module):
    """On-device classifier (same encoder backbone + the gateway head moved to the device)."""

    def __init__(self, n_classes: int, class_prior, latent_channels: int = 16, rr_dim: int = 5,
                 in_channels: int = 1):
        super().__init__()
        from .model import GatewayClassifier

        self.encoder = Encoder(latent_channels, variational=False, in_channels=in_channels)
        self.head = GatewayClassifier(latent_channels, n_classes, rr_dim)
        self.register_buffer("log_prior", torch.log(torch.as_tensor(class_prior, dtype=torch.float32)))
        self.rate_levels = [RateLevel(channels=1, bits=8)]

    def payload_bytes(self, level, window, rr_bytes):
        return 1  # the label; RR features are used on the device and not sent

    def forward(self, x, rr, level, channel=None, snr_db=None, loss_p=None):
        z, _ = self.encoder(x)
        logits = self.head(z, torch.ones_like(z), rr)
        if channel is not None and loss_p is not None:
            p = torch.as_tensor(loss_p, dtype=torch.float32, device=x.device).expand(len(x))
            lost = torch.rand(len(x), device=x.device) < p   # one-byte payload = one fragment
            if self.training:
                pass  # the device decision is unaffected by loss; training sees no erasure
            else:
                logits = torch.where(lost[:, None], self.log_prior.expand_as(logits), logits)
        return logits, _zero_aux(x)
