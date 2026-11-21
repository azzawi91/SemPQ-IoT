import itertools

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.channel.ble_phy import BerModel
from src.l3_scheduler import policies as pol
from src.l3_scheduler.model import SystemModel
from src.l3_scheduler.sim import run, threat_trace, traces

CFG = OmegaConf.load("configs/wp3.yaml")


@pytest.fixture(scope="module")
def model():
    return SystemModel.build(CFG, BerModel("noncoherent_fsk"))


def test_tables_monotone(model):
    q, u, e, d = model.tables(1.0, False)
    assert np.all(np.diff(q, axis=1) >= -1e-12)   # more power -> higher delivery
    assert np.all(np.diff(e, axis=1) > 0)         # more power -> more energy
    assert np.all(np.diff(e, axis=0) > 0)         # larger payload -> more energy
    q0, u0, _, d0 = model.tables(1.0, True)
    assert np.all(q0 == 0) and np.all(d0 == model.slot_s)


def test_dpp_is_exact_argmin(model):
    rng = np.random.default_rng(0)
    for _ in range(50):
        g, theta, sigma = rng.exponential(), int(rng.integers(0, 3)), int(rng.integers(0, 3))
        q, u, e, d = model.tables(g, False)
        p = pol.DPP(V=rng.uniform(1, 100), E_bar=1e-5, D_bar=0.1, Q=rng.uniform(0, 50), Z=rng.uniform(0, 50))
        x = p.decide(model, q, u, e, d, theta, sigma)
        best = min(((-p.V * u[r, k] + p.Q * (e[r, k] + model.security_cost(s, sigma)[0]) / p.E_bar
                     + p.Z * (d[r, k] + model.security_cost(s, sigma)[1]) / p.D_bar, r, k, s)
                    for r, k, s in itertools.product(range(u.shape[0]), range(u.shape[1]), (0, 1, 2))
                    if s >= model.floor[theta]))
        val = -p.V * u[x.r, x.p] + p.Q * (e[x.r, x.p] + model.security_cost(x.s, sigma)[0]) / p.E_bar \
            + p.Z * (d[x.r, x.p] + model.security_cost(x.s, sigma)[1]) / p.D_bar
        assert val == pytest.approx(best[0], rel=1e-12, abs=1e-15)


def test_floor_always_met_and_ablation_violates(model):
    cfg = OmegaConf.merge(CFG, {"slots": 3000})
    tr = traces(cfg, np.random.default_rng(1))
    assert set(np.unique(tr["theta"])) >= {0, 1}
    r = run(pol.DPP(cfg.V, (model.e_enc + cfg.E_bar_extra), cfg.D_bar), model, tr)
    assert r["floor_compliance"] == 1.0
    r2 = run(pol.DPP(cfg.V, (model.e_enc + cfg.E_bar_extra), cfg.D_bar, use_floor=False), model, tr)
    assert r2["floor_compliance"] < 1.0


def test_threat_chain_stationary_distribution():
    P = np.array([[0.95, 0.04, 0.01], [0.1, 0.85, 0.05], [0.05, 0.15, 0.8]])  # fast-mixing test chain
    th = threat_trace(P, 300_000, np.random.default_rng(2))
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi /= pi.sum()
    emp = np.bincount(th, minlength=3) / len(th)
    assert np.allclose(emp, pi, atol=0.02)


def test_larger_v_more_utility_and_larger_queues(model):
    cfg = OmegaConf.merge(CFG, {"slots": 4000})
    tr = traces(cfg, np.random.default_rng(3))
    lo = run(pol.DPP(1.0, (model.e_enc + cfg.E_bar_extra), cfg.D_bar), model, tr)
    hi = run(pol.DPP(1000.0, (model.e_enc + cfg.E_bar_extra), cfg.D_bar), model, tr)
    assert hi["utility"] >= lo["utility"] - 1e-3
    assert hi["Q_mean"] >= lo["Q_mean"]


def test_vectorised_tables_match_scalar(model):
    rng = np.random.default_rng(5)
    g = rng.exponential(size=50)
    o = rng.random(50) < 0.2
    QA, UA, EA, DA = model.tables_all(g, o)
    for t in range(50):
        q, u, e, d = model.tables(float(g[t]), bool(o[t]))
        np.testing.assert_allclose(QA[t], q, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(UA[t], u, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(EA[t], e, rtol=1e-12)
        np.testing.assert_allclose(DA[t], d, rtol=1e-12, atol=1e-15)
