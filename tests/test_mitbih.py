import numpy as np
import pytest

from src.data import mitbih

ALL_RECORDS = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116,
               117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210,
               212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234}


def test_split_sizes_and_disjointness():
    assert len(mitbih.DS1) == len(set(mitbih.DS1)) == 22
    assert len(mitbih.DS2) == len(set(mitbih.DS2)) == 22
    assert not set(mitbih.DS1) & set(mitbih.DS2)
    assert not (set(mitbih.DS1) | set(mitbih.DS2)) & set(mitbih.PACED)
    assert set(mitbih.DS1) | set(mitbih.DS2) | set(mitbih.PACED) == ALL_RECORDS


def test_same_subject_pair_is_flagged():
    a, b = mitbih.SAME_SUBJECT_PAIRS[0]
    assert a in mitbih.DS1 and b in mitbih.DS2


def test_aami_map_covers_classes_only():
    assert set(mitbih.AAMI_MAP.values()) == set(mitbih.AAMI_CLASSES)
    for non_beat in ("+", "~", "|", '"', "x", "!", "[", "]"):
        assert non_beat not in mitbih.AAMI_MAP


def test_rr_features():
    s = np.array([0, 360, 720, 1260])
    rr = mitbih.rr_features(s, fs=360, local=2)
    np.testing.assert_allclose(rr[:, 0], [1.0, 1.0, 1.0, 1.5])
    np.testing.assert_allclose(rr[:, 1], [1.0, 1.0, 1.5, 1.5])
    np.testing.assert_allclose(rr[:, 2], [1.0, 1.0, 1.0, 1.25])


def test_validation_choice_is_deterministic_and_in_ds1():
    rng = np.random.default_rng(1)
    counts = {r: rng.integers(1, 500, 5) for r in mitbih.DS1}
    v1 = mitbih.choose_validation_records(counts, sizes=(3, 4))
    v2 = mitbih.choose_validation_records(dict(reversed(list(counts.items()))), sizes=(3, 4))
    assert v1 == v2 and set(v1) <= set(mitbih.DS1) and 3 <= len(v1) <= 4


def test_zscore_per_beat():
    x = np.random.default_rng(0).normal(3, 2, (10, 50)).astype(np.float32)
    z = mitbih.zscore_per_beat(x)
    np.testing.assert_allclose(z.mean(1), 0, atol=1e-5)
    np.testing.assert_allclose(z.std(1), 1, atol=1e-3)


def test_extract_record_on_synthetic_wfdb(tmp_path):
    wfdb = pytest.importorskip("wfdb")
    n = 3600
    v5 = np.zeros(n)
    mlii = np.sin(np.arange(n) / 50.0)
    # MLII deliberately on channel 1 to check selection by name, as in record 114
    wfdb.wrsamp("900", fs=360, units=["mV", "mV"], sig_name=["V5", "MLII"],
                p_signal=np.stack([v5, mlii], 1), fmt=["212", "212"], write_dir=str(tmp_path))
    samples = np.array([50, 400, 800, 1200, 1700, 2100, 3550])
    symbols = ["N", "V", "+", "A", "F", "/", "N"]
    wfdb.wrann("900", "atr", samples, symbol=symbols, write_dir=str(tmp_path))
    b = mitbih.extract_record(tmp_path, 900, pre=90, post=162)
    # first beat (50 < 90) and last beat (3550 + 162 > 3600) are dropped; '+' is not a beat
    assert b.r_sample.tolist() == [400, 1200, 1700, 2100]
    assert [mitbih.AAMI_CLASSES[i] for i in b.y] == ["V", "S", "F", "Q"]
    assert b.x.shape == (4, 252)
    np.testing.assert_allclose(b.x[0], mlii[400 - 90:400 + 162], atol=0.01)
    bq = mitbih.extract_record(tmp_path, 900, pre=90, post=162, include_q=False)
    assert "Q" not in [mitbih.AAMI_CLASSES[i] for i in bq.y]
