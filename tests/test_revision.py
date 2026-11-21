"""Revision additions (decision 2026-10-13): combined L3 baselines and BLE connection-event energy."""
from types import SimpleNamespace

import numpy as np

from src.channel import ble_conn
from src.l3_scheduler import policies as pol


def test_combined_baselines_appended_and_fixed():
    cfg = SimpleNamespace(V=100.0, E_bar=1e-4, D_bar=0.15, level_names=["k1_b4", "k4_b8"], fixed_rate_level="k4_b8",
                          powers_dbm=[-4.0, 0.0, 4.0, 8.0], threshold_gain_db=0.0)
    ps = pol.make_policies(cfg, 2, 4)
    assert [p.name for p in ps[:9]][0] == "DPP (proposed)" and len(ps) == 11
    a, b = ps[9], ps[10]
    assert a.rate_fixed == 0 and a.tier_fixed == 2 and a.power_fixed is None
    x = b.decide(None, None, np.zeros((2, 4)), None, None, 0, 0)
    assert (x.r, x.p, x.s) == (0, 3, 2)


def test_connection_event_energy_monotone_and_variants():
    e1, e2 = ble_conn.event_energy_j(1)["e_event_j"], ble_conn.event_energy_j(2)["e_event_j"]
    assert 0 < e1 < e2
    assert ble_conn.event_energy_j(1, "fast_rampup")["e_event_j"] < e1 < ble_conn.event_energy_j(1, "pessimistic_hfxo")["e_event_j"]
    assert abs(ble_conn.sleep_energy_j(1.0) - 3.16e-6 * 3.0) < 1e-12


def test_external_evaluation_runs_without_result_branches(tmp_path, monkeypatch):
    """A copy of the code has no result branches: the frozen models are then read from results/wp1/ of the run."""
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("reproduce", root / "reproduce.py")
    rep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rep)
    monkeypatch.setenv("EXT_REF", "origin/r/some-release")
    assert rep.ext_ref() == "origin/r/some-release"                 # an explicit choice always wins
    monkeypatch.delenv("EXT_REF")
    monkeypatch.setattr(rep, "REPO", tmp_path)                       # a directory without the release branch
    assert rep.ext_ref() == "local"

    sys.path.insert(0, str(root / "experiments"))
    import wp1_external as ext
    monkeypatch.setattr(ext, "REPO", tmp_path)
    (tmp_path / "results" / "wp1").mkdir(parents=True)
    (tmp_path / "results" / "wp1" / "stats.json").write_text('{"levels": {}}')
    assert ext.git_bytes("local", "results/wp1/stats.json") == b'{"levels": {}}'
