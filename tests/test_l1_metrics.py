import numpy as np
import pytest

from src.l1_semantic.metrics import (InconsistentMetrics, assert_consistent, auroc_binary,
                                     consistency_report, multiclass_metrics, multilabel_metrics)

NAMES = ("N", "S", "V", "F")


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y + np.round(rng.normal(size=300))  # ties included
    pos, neg = s[y == 1], s[y == 0]
    pairwise = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    assert auroc_binary(y, s) == pytest.approx(pairwise)
    assert np.isnan(auroc_binary(np.zeros(5), np.arange(5)))


def test_hand_computed_multiclass():
    y = np.array([0, 0, 0, 1, 1, 2])
    scores = np.eye(4)[[0, 0, 1, 1, 0, 2]]
    m = multiclass_metrics(y, scores, NAMES)
    assert m["confusion_matrix"][0] == [2, 1, 0, 0]
    assert m["accuracy"] == pytest.approx(4 / 6)
    # F1: N p=2/3 r=2/3; S p=1/2 r=1/2; V 1; F absent (excluded)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 1 / 2 + 1) / 3)
    assert m["per_class"]["F"]["support"] == 0
    assert consistency_report(m, y, scores)["ok"]


def test_consistency_catches_tampering():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 500)
    scores = rng.random((500, 4)) + 2 * np.eye(4)[y]
    m = multiclass_metrics(y, scores, NAMES)
    assert_consistent(m, y, scores)
    bad = dict(m, macro_f1=m["macro_f1"] + 0.01)
    with pytest.raises(InconsistentMetrics, match="macro-F1"):
        assert_consistent(bad, y, scores)
    bad = dict(m, per_class={**m["per_class"], "S": dict(m["per_class"]["S"], auroc=0.99)})
    with pytest.raises(InconsistentMetrics, match="AUROC"):
        assert_consistent(bad, y, scores)
    with pytest.raises(InconsistentMetrics, match="argmax"):
        assert_consistent(m, y, scores[:, ::-1])


def test_warning_for_high_auroc_with_poor_decisions():
    # Class 1 is perfectly ranked by its score but never wins the argmax:
    # AUROC = 1 while its decisions are at chance - the v4.4 pattern.
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    scores = np.zeros((400, 2))
    scores[:, 0] = 10.0
    scores[:, 1] = y + rng.random(400) * 0.1
    m = multiclass_metrics(y, scores, ("a", "b"))
    rep = consistency_report(m, y, scores)
    assert rep["ok"] and any("b: AUROC" in w for w in rep["warnings"])
    with pytest.raises(InconsistentMetrics):
        assert_consistent(m, y, scores, fail_on_warnings=True)


def test_multilabel():
    y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.1, 0.2]])
    m = multilabel_metrics(y, p, ("A", "B"))
    assert m["per_class"]["A"]["f1"] == pytest.approx(1.0)
    assert m["per_class"]["B"]["f1"] == pytest.approx(2 / 3)
    assert m["subset_accuracy"] == pytest.approx(3 / 4)
    assert m["per_class"]["A"]["auroc"] == pytest.approx(1.0)
