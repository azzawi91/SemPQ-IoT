import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pywt")

from src.l1_semantic import baselines as bl  # noqa: E402
from src.l1_semantic.channel import PayloadChannel  # noqa: E402
from src.l1_semantic.model import RateLevel  # noqa: E402


def test_dwt_matrix_orthonormal_and_reconstructs():
    a = torch.from_numpy(bl.dwt_matrix(256))
    x = torch.randn(5, 256)
    torch.testing.assert_close((x @ a.T) @ a, x, atol=1e-4, rtol=1e-4)


def test_payload_bytes():
    assert bl.RawBaseline(4).payload_bytes(RateLevel(1, 11), 252, 3) == 347 + 3
    d = bl.DWTBaseline(4)
    assert d.payload_bytes(RateLevel(16, 8), 252, 3) == 32 + 3
    assert bl.LabelBaseline(4, [0.7, 0.1, 0.15, 0.05]).payload_bytes(RateLevel(1), 252, 3) == 1
    j = bl.JSCCBaseline(4, [RateLevel(4)])
    assert j.payload_bytes(RateLevel(4, 8), 252, 3) == 4 * 8 + 3


@pytest.mark.parametrize("make", [lambda: bl.RawBaseline(4), lambda: bl.DWTBaseline(4),
                                  lambda: bl.JSCCBaseline(4, [RateLevel(2), RateLevel(16)])])
def test_forward_backward_through_channel(make):
    torch.manual_seed(0)
    m = make().train()
    x, rr = torch.randn(16, 1, 252), torch.rand(16, 5)
    logits, aux = m(x, rr, m.rate_levels[0], PayloadChannel(), snr_db=5.0, loss_p=0.3)
    assert logits.shape == (16, 4) and torch.isfinite(logits).all()
    (logits.sum() + aux.get("extra_loss", 0)).backward()


def test_dwt_full_level_reconstruction_is_close():
    torch.manual_seed(1)
    d = bl.DWTBaseline(4, coeff_levels=(256,), value_bits=8, clip=8.0).eval()
    x = torch.randn(4, 1, 252)
    c = torch.nn.functional.pad(x[:, 0], (0, 4)) @ d.A.T
    rec = (bl._quantise(c, 8, 8.0) @ d.A)[:, :252]
    assert (rec - x[:, 0]).abs().max() < 0.1


def test_label_fallback_to_prior_when_lost():
    torch.manual_seed(2)
    prior = [0.7, 0.1, 0.15, 0.05]
    m = bl.LabelBaseline(4, prior).eval()
    x, rr = torch.randn(1000, 1, 252), torch.rand(1000, 5)
    with torch.no_grad():
        logits, _ = m(x, rr, m.rate_levels[0], PayloadChannel(), loss_p=1.0)
    torch.testing.assert_close(torch.softmax(logits, 1)[0], torch.tensor(prior))


@pytest.mark.parametrize("make", [lambda: bl.RawBaseline(4, gateway_template=True),
                                  lambda: bl.DWTBaseline(4, gateway_template=True),
                                  lambda: bl.JSCCBaseline(4, [RateLevel(2)], gateway_template=True)])
def test_gateway_template_variants_accept_two_channels(make):
    torch.manual_seed(0)
    m = make().train()
    x, rr = torch.randn(8, 2, 252), torch.rand(8, 5)
    logits, aux = m(x, rr, m.rate_levels[0], PayloadChannel(), snr_db=5.0, loss_p=0.2)
    assert logits.shape == (8, 4)
    (logits.sum() + aux.get("extra_loss", 0)).backward()
