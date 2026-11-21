"""Encoder-size option (decision 2026-10-13): smaller encoders keep the latent length and payload sizes."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from src.l1_semantic.model import RateLevel, SemanticCodec, encoder_macs  # noqa: E402
from wp1_encsize import CANDIDATES, macs  # noqa: E402


def test_candidates_keep_latent_length_and_match_mac_count():
    levels = [RateLevel(1, 4), RateLevel(1, 8), RateLevel(16, 8)]
    for name, (w, k, s) in CANDIDATES.items():
        m = SemanticCodec(4, levels, 16, 4.0, rr_dim=5, window=252, in_channels=2,
                          encoder_kwargs={"widths": tuple(w), "kernels": tuple(k), "strides": tuple(s)})
        mu, _ = m.encoder(torch.zeros(2, 2, 252))
        assert mu.shape[-1] == 8, name
        assert encoder_macs(m.encoder, 252) == macs(w, k, s)[0], name
