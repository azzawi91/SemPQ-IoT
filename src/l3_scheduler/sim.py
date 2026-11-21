"""Slot simulator for the L3 scheduler: exogenous traces (threat chain, fading, outages) are
generated once per seed and shared by every policy, so policies are compared on identical
sample paths (paired comparison)."""
from __future__ import annotations

import numpy as np

from src.channel.fading import block_fading_trace, lin_to_db
from src.channel.gilbert_elliott import GilbertElliott


def threat_trace(P: np.ndarray, n: int, rng: np.random.Generator, start: int = 0) -> np.ndarray:
    th = np.empty(n, dtype=int)
    s = start
    u = rng.random(n)
    cum = np.cumsum(P, axis=1)
    for t in range(n):
        th[t] = s
        s = int(np.searchsorted(cum[s], u[t], side="right"))
        s = min(s, P.shape[0] - 1)
    return th


def traces(cfg, rng: np.random.Generator) -> dict:
    n = cfg.slots
    P = np.asarray(cfg.threat.P, dtype=float)
    ge = GilbertElliott.from_outage(cfg.outage.mean_slots, cfg.outage.fraction) if cfg.outage.fraction > 0 else None
    return {"theta": threat_trace(P, n, rng),
            "gain": block_fading_trace(n, cfg.block_len, cfg.k_factor, rng),
            "outage": ge.states_fast(n, rng) if ge is not None else np.zeros(n, bool),
            "u": rng.random(n)}


def run(policy, model, tr: dict, tables=None, beat_preds=None, fallback: int = 0, record: bool = False) -> dict:
    """Simulate one policy on a trace. ``tables`` = model.tables_all(gain, outage) may be passed
    to share the per-slot tables across policies (identical numbers, computed once).
    ``beat_preds`` (end-to-end mode): list over rate levels of per-beat predicted classes, one beat
    per slot; a delivered beat gets the prediction of the chosen level, a lost beat ``fallback``.
    The realised predictions are returned under "yhat". ``record=True`` adds per-slot arrays under
    "trace" (decisions, utility, energy, delay, delivery, queues) without changing other outputs."""
    n = len(tr["theta"])
    if tables is None:
        tables = model.tables_all(tr["gain"], tr["outage"])
    QA, UA, EA, DA = tables
    sigma = model.floor[int(tr["theta"][0])]   # initial session at the initial floor
    util = energy = delay = 0.0
    delivered = floor_ok = 0
    f1_delivered = 0.0
    Q_hist, Z_hist = [], []
    yhat = np.full(n, fallback, dtype=np.int8) if beat_preds is not None else None
    rec = {k: np.zeros(n) for k in ("r", "p", "s", "u", "e", "d", "ok")} if record else None
    for t in range(n):
        th, g, o = int(tr["theta"][t]), float(tr["gain"][t]), bool(tr["outage"][t])
        q, u, e, d = QA[t], UA[t], EA[t], DA[t]
        if hasattr(policy, "gain_db_threshold"):
            x = policy.decide(model, q, u, e, d, th, sigma, gain_db=10 * np.log10(max(g, 1e-300)))
        else:
            x = policy.decide(model, q, u, e, d, th, sigma)
        es, ds = model.security_cost(x.s, sigma)
        e_t, d_t = e[x.r, x.p] + es, d[x.r, x.p] + ds
        ok = tr["u"][t] < q[x.r, x.p]
        util += u[x.r, x.p]
        energy += e_t
        delay += d_t
        delivered += ok
        if yhat is not None and ok:
            yhat[t] = beat_preds[x.r][t]
        f1_delivered += ok * model.levels[x.r]["macro_f1"]
        floor_ok += x.s >= model.floor[th]
        if rec is not None:
            for k, v in (("r", x.r), ("p", x.p), ("s", x.s), ("u", u[x.r, x.p]), ("e", e_t), ("d", d_t), ("ok", ok)):
                rec[k][t] = v
        sigma = x.s
        policy.update(e_t, d_t)
        if hasattr(policy, "Q"):
            Q_hist.append(policy.Q)
            Z_hist.append(policy.Z)
    out = {"policy": policy.name, "slots": n, "utility": util / n, "energy_per_slot_j": energy / n,
           "delay_per_slot_s": delay / n, "delivery_ratio": delivered / n,
           "f1_delivered_per_slot": f1_delivered / n, "floor_compliance": floor_ok / n}
    if yhat is not None:
        out["yhat"] = yhat
    if rec is not None:
        rec["theta"] = np.asarray(tr["theta"], float)
        rec["gain_db"] = 10 * np.log10(np.maximum(np.asarray(tr["gain"], float), 1e-300))
        if Q_hist:
            rec["Q"], rec["Z"] = np.asarray(Q_hist), np.asarray(Z_hist)
        out["trace"] = rec
    if Q_hist:
        out.update({"Q_mean": float(np.mean(Q_hist)), "Z_mean": float(np.mean(Z_hist)),
                    "Q_max": float(np.max(Q_hist)), "Z_max": float(np.max(Z_hist)),
                    "Q_final": float(Q_hist[-1]), "Z_final": float(Z_hist[-1])})
    return out
