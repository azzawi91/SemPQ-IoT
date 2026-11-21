import numpy as np
import pytest

torch = pytest.importorskip("torch")
from omegaconf import OmegaConf  # noqa: E402

from src.l1_semantic.data import SplitArrays, class_weights, expand_rr, quantise_rr  # noqa: E402
from src.l1_semantic.model import RateLevel, SemanticCodec  # noqa: E402
from src.l1_semantic.train import evaluate_grid, train  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

CLASSES = ("N", "S", "V", "F")


def _toy(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    t = np.linspace(0, 1, 252)
    x = np.sin(2 * np.pi * (y[:, None] + 1) * t[None, :]) + 0.3 * rng.normal(size=(n, 252))
    rr = expand_rr(quantise_rr(rng.uniform(0.5, 1.2, (n, 3)), 0.01))
    return SplitArrays(x=x[:, None, :].astype(np.float32), rr=rr, y=y.astype(np.int64),
                       record=np.zeros(n, np.int32))


def _cfg(epochs):
    return OmegaConf.create({
        "epochs": epochs, "batch_size": 64, "lr": 3e-3, "weight_decay": 0.0, "grad_clip": 5.0,
        "patience": 100, "min_delta": 0.0, "beta_kl": 1e-3, "lambda_rate": 1e-3, "seed": 0,
        "channel": {"fragment_bytes": 16, "rr_bytes": 3, "train_snr_db": [0.0, 20.0], "clean_fraction": 0.3,
                    "train_loss_max": 0.4}})


def _model():
    return SemanticCodec(4, [RateLevel(2), RateLevel(16)])


def _run(tmp, epochs):
    set_seed(0)
    m = _model()
    tr, va = _toy(512, 1), _toy(256, 2)
    s = train(m, tr, va, _cfg(epochs), CLASSES, tmp, class_weights(tr.y, 4), log=lambda *_: None)
    return m, s


def test_training_learns_and_is_deterministic(tmp_path):
    m1, s1 = _run(tmp_path / "a", 8)
    m2, s2 = _run(tmp_path / "b", 8)
    assert s1 == s2
    for p1, p2 in zip(m1.state_dict().values(), m2.state_dict().values()):
        torch.testing.assert_close(p1, p2)
    assert s1["best_val_score"] > 0.9  # toy task is separable


def test_resume_matches_uninterrupted(tmp_path):
    m_full, _ = _run(tmp_path / "full", 4)
    _run(tmp_path / "split", 2)
    set_seed(123)  # different global state before resuming: the checkpoint must restore it
    m_res = _model()
    tr, va = _toy(512, 1), _toy(256, 2)
    train(m_res, tr, va, _cfg(4), CLASSES, tmp_path / "split", class_weights(tr.y, 4), log=lambda *_: None)
    for p1, p2 in zip(m_full.state_dict().values(), m_res.state_dict().values()):
        torch.testing.assert_close(p1, p2)


def test_evaluate_grid_rows_and_consistency(tmp_path):
    m, _ = _run(tmp_path, 2)
    te = _toy(300, 3)
    rows = evaluate_grid(m, te, CLASSES, [0.0, 10.0], [0.2], eval_seed=0)
    assert len(rows) == 2 * (1 + 2 + 1)
    again = evaluate_grid(m, te, CLASSES, [0.0, 10.0], [0.2], eval_seed=0)
    assert [r["metrics"]["macro_f1"] for r in rows] == [r["metrics"]["macro_f1"] for r in again]


def test_class_weights_use_train_counts_only():
    w = class_weights(np.array([0, 0, 0, 1]), 2)
    assert w[1] > w[0]
    with pytest.raises(ValueError):
        class_weights(np.array([0, 0]), 2)


def test_augment_disabled_is_identity_and_enabled_changes_input():
    from src.l1_semantic.train import augment

    x = torch.randn(8, 1, 252)
    g = torch.Generator().manual_seed(0)
    assert torch.equal(augment(x, None, g), x)
    cfg = OmegaConf.create({"enabled": True, "scale": 0.2, "wander": 0.3, "noise": 0.05, "shift": 5})
    y1 = augment(x, cfg, torch.Generator().manual_seed(1))
    y2 = augment(x, cfg, torch.Generator().manual_seed(1))
    assert torch.equal(y1, y2) and not torch.equal(y1, x)


def test_hybrid_reconstruction_term():
    m = SemanticCodec(4, [RateLevel(4)], recon_weight=1.0).train()
    _, aux = m(torch.randn(8, 1, 252), torch.rand(8, 5), RateLevel(4))
    assert "extra_loss" in aux and aux["extra_loss"].item() > 0
    m.eval()
    _, aux = m(torch.randn(8, 1, 252), torch.rand(8, 5), RateLevel(4))
    assert "extra_loss" not in aux
