"""INCART external-cohort helpers (decision 2026-10-13)."""
import numpy as np

from src.data import incart, mitbih


def test_position_mapping_and_resampling_length():
    assert incart.map_positions(np.array([0, 257, 514])).tolist() == [0, 360, 720]
    assert len(incart.resample_signal(np.zeros(257 * 4))) == 360 * 4


def test_ec57_grouping_of_external_symbols():
    smp, lab, counts = incart.beat_annotations(["N", "B", "n", "r", "+", "V", "Q"], [1, 2, 3, 4, 5, 6, 7])
    names = [mitbih.AAMI_CLASSES[i] for i in lab]
    assert names == ["N", "N", "S", "V", "V", "Q"]
    assert smp.tolist() == [1, 2, 3, 4, 6, 7] and counts["+"] == 1
    assert "B" not in mitbih.AAMI_MAP  # MIT-BIH mapping unchanged


def test_subject_groups_from_patient_line():
    g = incart.subject_groups({"I01": ["<age>: 65", "patient 1", "PVCs"], "I02": ["<age>: 65", "patient 1", "noise"],
                               "I03": ["patient 2", "x"], "I04": [], "I05": []})
    assert g["I01"] == g["I02"] != g["I03"]
    assert g["I04"] != g["I05"] and g["I04"] not in (g["I01"], g["I03"])


def test_edge_windows_dropped_rr_over_all_beats():
    sig = np.arange(1000, dtype=np.float32)
    b = incart.windows_from_signal(sig, np.array([10, 300, 600, 990]), np.zeros(4, int), 1, 90, 162)
    assert b.r_sample.tolist() == [300, 600]
    assert np.allclose(b.rr[0, 0], 290 / 360)
