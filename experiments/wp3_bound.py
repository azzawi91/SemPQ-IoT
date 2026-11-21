"""Sample-path utility upper bound for the implemented scheduler (decision 2026-10-13, external review point 2).

For a trace omega(0..T-1), every policy of the implemented system (tier-dependent handshakes, history- or
future-dependent, clairvoyant included) that meets both budgets on the trace satisfies, for all mu, nu >= 0,

  (1/T) sum_t u(t) <= (1/T) sum_t max_{x in X(omega_t)} [ u(x, w_t) - mu (e0~(x, w_t) - 1) - nu (d0~(x, w_t) - 1) ],

where e0~ and d0~ are the handshake-free lower bounds of the normalized costs (re-key charged only when the floor
forces tier 2), because the true costs are at least these bounds (weak duality). The bound U_UB is the minimum of
the right-hand side over (mu, nu); it is computed per trace and compared with the time-average utility of the
implemented policies (results/wp3/main.json, gamma.json of the same commit).

    python experiments/wp3_bound.py inputs_path=results/wp3/inputs.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig, OmegaConf, open_dict  # noqa: E402


MULT_MAX = 1e4


def dual_bound(u: np.ndarray, et: np.ndarray, dt: np.ndarray, a: float = 0.0, b: float = 0.0) -> dict:
    """min over 0 <= mu, nu <= MULT_MAX of  mean_t max_x [u - mu et - nu dt] + mu a + nu b ; arrays (T, K).
    With a = b = 0 this bounds every policy that meets both budgets on the trace; with a = e - 1, b = d - 1 it
    bounds every policy whose average normalized costs on the trace are (e, d). The objective is convex and
    piecewise linear: coarse grid, then Nelder-Mead on (log mu, log nu); any evaluated point is a valid upper
    bound, so an inexact minimiser only loosens it."""
    from scipy.optimize import minimize

    def g(mu, nu):
        mu, nu = min(mu, MULT_MAX), min(nu, MULT_MAX)
        return float(np.mean(np.max(u - mu * et - nu * dt, axis=1))) + mu * a + nu * b

    grid = np.concatenate([[0.0], np.logspace(-4, 4, 33)])
    best = min((g(x, y), x, y) for x in grid for y in grid)
    x0 = np.log(np.maximum([best[1], best[2]], 1e-6))
    r = minimize(lambda z: g(*np.exp(np.minimum(z, np.log(MULT_MAX)))), x0, method="Nelder-Mead",
                 options={"xatol": 1e-4, "fatol": 1e-8})
    mu_r, nu_r = np.exp(np.minimum(r.x, np.log(MULT_MAX)))
    val, mu, nu = min(best, (g(mu_r, nu_r), mu_r, nu_r))
    return {"bound": val, "mu": float(mu), "nu": float(nu), "at_cap": bool(max(mu, nu) >= MULT_MAX * 0.999)}


def phi_max(et: np.ndarray, dt: np.ndarray) -> float:
    """Feasibility certificate of Proposition 2 on the trace (handshake-free lower bounds)."""
    return max(float(np.mean(np.min(l * et + (1 - l) * dt, axis=1))) for l in np.linspace(0, 1, 201))


@hydra.main(config_path="../configs", config_name="wp3", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from src.channel.ble_phy import BerModel, gfsk_ber_curve
    from src.l3_scheduler.model import SystemModel
    from src.l3_scheduler.sim import traces
    from src.utils.provenance import provenance
    from src.utils.seed import pin_threads, set_seed

    set_seed(cfg.seed)
    pin_threads(1)
    grid = np.asarray(cfg.ber.ebn0_grid_db, float)
    ber = BerModel("gfsk_mc", grid, gfsk_ber_curve(grid, np.random.default_rng(cfg.ber.seed), n_bits=cfg.ber.n_bits))
    m0 = SystemModel.build(cfg, ber)
    with open_dict(cfg):
        cfg.E_bar = float(m0.e_enc + cfg.E_bar_extra)
    res = REPO / "results" / "wp3"
    achieved = {}
    for fname, key in (("main.json", None), ("gamma.json", "gamma0_db")):
        p = res / fname
        if p.exists():
            for row in json.loads(p.read_text())["rows"]:
                g0 = float(row[key]) if key else float(cfg.gamma0_db)
                achieved.setdefault((g0, int(row["seed"])), {})[row["policy"]] = {
                    "utility": row["utility"], "e_ratio": row["energy_per_slot_j"] / cfg.E_bar,
                    "d_ratio": row["delay_per_slot_s"] / cfg.D_bar}
    rows = []
    for g0 in sorted(set(cfg.gamma0_sweep) | {cfg.gamma0_db}):
        m = SystemModel.build(OmegaConf.merge(cfg, {"gamma0_db": g0}), ber)
        for seed in cfg.seeds:
            tr = traces(cfg, np.random.default_rng(seed))
            _, UA, EA, DA = m.tables_all(tr["gain"], tr["outage"])
            th = np.asarray(tr["theta"], int)
            rk = np.where(np.array([m.floor[int(x)] for x in th]) == 2, m.rekey_energy_j / m.rekey_every, 0.0)
            T = len(th)
            u = UA.reshape(T, -1)
            et = ((EA + rk[:, None, None]) / cfg.E_bar - 1.0).reshape(T, -1)
            dt = (DA / cfg.D_bar - 1.0).reshape(T, -1)
            ph = phi_max(et, dt)
            feasible = ph <= 0
            ub = dual_bound(u, et, dt) if feasible else None
            ach = achieved.get((float(g0), int(seed)), {})
            # policy-specific bound: largest utility of any policy (clairvoyant included) whose average normalized
            # costs on this trace do not exceed those of the given policy
            for v in ach.values():
                bb = dual_bound(u, et, dt, v["e_ratio"] - 1, v["d_ratio"] - 1)
                v["bound_at_own_costs"], v["gap_to_own_bound"] = bb["bound"], bb["bound"] - v["utility"]
                v["bound_at_cap"] = bb["at_cap"]
            row = {"gamma0_db": float(g0), "seed": int(seed), "phi_max": ph, "feasible": feasible,
                   "U_UB": ub["bound"] if ub else None, "mu": ub["mu"] if ub else None, "nu": ub["nu"] if ub else None,
                   "U_unconstrained": float(np.mean(np.max(u, axis=1))), "achieved": ach}
            rows.append(row)
            print(g0, seed, round(ph, 4), row["U_UB"] and round(row["U_UB"], 4),
                  {k[:24]: (round(v["utility"], 4), round(v["gap_to_own_bound"], 4)) for k, v in ach.items()}, flush=True)
    out = {"provenance": provenance(OmegaConf.to_container(cfg, resolve=True), cfg.seed), "rows": rows}
    res.mkdir(parents=True, exist_ok=True)
    (res / "bound.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
