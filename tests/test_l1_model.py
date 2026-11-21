import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.l1_semantic.channel import PayloadChannel, fragment_keep  # noqa: E402
from src.l1_semantic.model import (RateLevel, SemanticCodec, count_params, encoder_macs,  # noqa: E402
                                   encoder_memory_estimate)

LEVELS = [RateLevel(1), RateLevel(2), RateLevel(4), RateLevel(8), RateLevel(16)]


def _codec(**kw):
    torch.manual_seed(0)
    return SemanticCodec(n_classes=4, rate_levels=LEVELS, **kw)


def test_encoder_budget_and_geometry():
    m = _codec()
    assert count_params(m.encoder) <= 50_000
    assert m.latent_steps(252) == 8
    assert m.payload_bytes(RateLevel(16, 8), 252, rr_bytes=3) == 16 * 8 + 3
    assert m.payload_bytes(RateLevel(1, 4), 252, rr_bytes=0) == math.ceil(1 * 8 * 4 / 8)
    macs = encoder_macs(m.encoder, 252)
    mem = encoder_memory_estimate(m.encoder, 252)
    assert 0 < macs < 5_000_000 and mem["flash_bytes"] < 50_000 and mem["ram_bytes"] > 252


def test_forward_shapes_and_losses():
    m = _codec().train()
    x, rr = torch.randn(32, 1, 252), torch.randn(32, 5)
    logits, aux = m(x, rr, LEVELS[2], PayloadChannel(), snr_db=10.0, loss_p=0.2)
    assert logits.shape == (32, 4)
    assert aux["kl"].item() >= 0 and aux["rate_bits"].item() > 0
    (logits.sum() + aux["kl"] + aux["rate_bits"]).backward()


def test_rate_level_masks_unused_channels():
    m = _codec().eval()
    x, rr = torch.randn(4, 1, 252), torch.zeros(4, 5)
    with torch.no_grad():
        mu, _ = m.encoder(x)
        out1, aux1 = m(x, rr, RateLevel(2))
        # the decision at level 2 must equal the classifier applied to channels 0-1 only
        z, _ = m.quantise(mu, 8, train=False)
        mask = torch.zeros_like(z)
        mask[:, :2] = 1
        out2 = m.classifier(z * mask, mask, rr)
    torch.testing.assert_close(out1, out2)
    assert aux1["rate_bits"] > 0


def test_eval_quantisation_is_on_grid():
    m = _codec()
    z = torch.linspace(-6, 6, 101)[None, None, :]
    zq, step = m.quantise(z, 4, train=False)
    levels = torch.round((zq + m.clip) / step)
    torch.testing.assert_close(levels, levels.round())
    assert zq.min() >= -m.clip - 1e-6 and zq.max() <= m.clip + 1e-6
    assert len(torch.unique(zq)) <= 16


def test_channel_awgn_snr():
    torch.manual_seed(1)
    z = torch.randn(4000, 16, 8)
    rx, m, rr_keep = PayloadChannel()(z, torch.ones_like(z), 8, snr_db=10.0)
    noise_p = (rx - z).pow(2).mean() / z.pow(2).mean()
    assert abs(10 * np.log10(1 / noise_p.item()) - 10.0) < 0.2
    assert torch.all(rr_keep == 1)


def test_fragment_keep_geometry():
    torch.manual_seed(2)
    # 3 RR bytes then 13 one-byte elements: fragment 0 = bytes 0-7, fragment 1 = bytes 8-15
    bits = torch.full((16,), 8.0)
    k = fragment_keep(20000, bits, 8, torch.full((20000,), 0.3))
    assert torch.all(k[:, :8] == k[:, :1]) and torch.all(k[:, 8:] == k[:, 8:9])
    assert abs(1 - k[:, 0].mean().item() - 0.3) < 0.015
    # a 12-bit element straddling a fragment boundary needs both fragments
    k = fragment_keep(20000, torch.tensor([60.0, 12.0]), 8, torch.full((20000,), 0.3))
    assert abs(k[:, 1].mean().item() - 0.49) < 0.015


def test_channel_erasure_order_and_rr():
    torch.manual_seed(3)
    z = torch.randn(30000, 16, 8)
    mask = torch.zeros_like(z)
    mask[:, :4] = 1  # k = 4 channels, 8 bits: 32 latent bytes + 3 RR bytes, 16-byte fragments
    rx, m, rr_keep = PayloadChannel(16, 3)(z * mask, mask, 8, loss_p=0.2)
    assert torch.all(m[:, 4:] == 0) and torch.all(rx[m == 0] == 0)
    # time-major order: latent bytes 0-12 (t=0..3, partly) share fragment 0 with the RR bytes
    assert torch.all(m[:, :, 0][:, :4] == rr_keep)
    assert abs(rr_keep.mean().item() - 0.8) < 0.01
    erased = 1 - m[:, :4].mean().item()
    assert abs(erased - 0.2) < 0.01
