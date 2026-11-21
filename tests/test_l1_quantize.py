import pytest

torch = pytest.importorskip("torch")

from src.l1_semantic.model import RateLevel, SemanticCodec  # noqa: E402
from src.l1_semantic.quantize import QuantizedEncoderAdapter, quantize_encoder_int8  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def test_int8_encoder_close_to_float_and_int8_weights():
    set_seed(0)
    m = SemanticCodec(4, [RateLevel(16)]).eval()
    calib = torch.randn(512, 1, 252)
    q = quantize_encoder_int8(m.encoder, calib)
    x = torch.randn(64, 1, 252)
    with torch.no_grad():
        ref, _ = m.encoder(x)
        out, _ = QuantizedEncoderAdapter(q, 16)(x)
    rel = (out - ref).norm() / ref.norm()
    assert rel < 0.05
    w = q.body[0][0].weight()
    assert w.dtype == torch.qint8


def test_codec_runs_with_quantized_encoder():
    set_seed(1)
    m = SemanticCodec(4, [RateLevel(4)]).eval()
    m.encoder = QuantizedEncoderAdapter(quantize_encoder_int8(m.encoder, torch.randn(256, 1, 252)), 16)
    with torch.no_grad():
        logits, _ = m(torch.randn(8, 1, 252), torch.rand(8, 5), RateLevel(4))
    assert logits.shape == (8, 4)
