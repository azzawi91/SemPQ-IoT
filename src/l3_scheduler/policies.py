"""Scheduling policies (brief WP3).

Proposed: drift-plus-penalty (DPP). Constraints are normalised by their budgets
(e~ = e / E_bar, d~ = d / D_bar, both with budget 1) so that V is dimensionless.
Each slot, given virtual queues Q (energy) and Z (delay), choose x = (r, p, s) with
s >= s_min(Theta) that minimises
        V * (-u(x)) + Q * e~(x) + Z * d~(x)
by enumeration over |R| x |P| x |S| candidates (per-slot complexity O(|R||P||S|)).
Queues: Q <- max(Q + e~ - 1, 0),  Z <- max(Z + d~ - 1, 0).

Baselines: fixed maximum security, fixed rate (DPP over power and tier), myopic greedy
(maximise utility under a per-slot energy budget E_bar), threshold heuristic on the channel
gain. Ablations: DPP without the floor, without rate adaptation, without power control, and with the
tier fixed at 2 (maximum security in every slot, i.e. no threat awareness).
Combined baselines (external review, decision 2026-10-13): smallest rate level (8 B) and tier 2 fixed, with
(a) DPP over transmit power only and (b) the maximum transmit power in every slot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TIERS = (0, 1, 2)


@dataclass
class Decision:
    r: int
    p: int
    s: int


@dataclass
class DPP:
    V: float
    E_bar: float
    D_bar: float
    use_floor: bool = True
    rate_fixed: int | None = None     # index of the rate level, or None to adapt
    power_fixed: int | None = None
    tier_fixed: int | None = None     # e.g. 2: maximum security in every slot (threat-unaware)
    Q: float = 0.0
    Z: float = 0.0
    name: str = "DPP (proposed)"

    def decide(self, m, q, u, e, d, theta, sigma) -> Decision:
        best, arg = np.inf, None
        rs = [self.rate_fixed] if self.rate_fixed is not None else range(u.shape[0])
        ps = [self.power_fixed] if self.power_fixed is not None else range(u.shape[1])
        smin = m.floor[theta] if self.use_floor else 0
        for s in TIERS:
            if s < smin or (self.tier_fixed is not None and s != self.tier_fixed):
                continue
            es, ds = m.security_cost(s, sigma)
            for r in rs:
                for p in ps:
                    val = (-self.V * u[r, p] + self.Q * (e[r, p] + es) / self.E_bar
                           + self.Z * (d[r, p] + ds) / self.D_bar)
                    if val < best - 1e-15:
                        best, arg = val, Decision(r, p, s)
        return arg

    def update(self, e_t: float, d_t: float):
        self.Q = max(self.Q + e_t / self.E_bar - 1.0, 0.0)
        self.Z = max(self.Z + d_t / self.D_bar - 1.0, 0.0)


@dataclass
class FixedMaxSecurity:
    name: str = "Fixed max. security"

    def decide(self, m, q, u, e, d, theta, sigma):
        return Decision(u.shape[0] - 1, u.shape[1] - 1, 2)

    def update(self, *a):
        pass


@dataclass
class FixedAction:
    r: int
    p: int
    s: int
    name: str = "Fixed action"

    def decide(self, m, q, u, e, d, theta, sigma):
        return Decision(self.r, self.p, self.s)

    def update(self, *a):
        pass


@dataclass
class MyopicGreedy:
    E_bar: float
    name: str = "Myopic greedy"

    def decide(self, m, q, u, e, d, theta, sigma):
        best, arg = -np.inf, None
        for s in TIERS:
            if s < m.floor[theta]:
                continue
            es, _ = m.security_cost(s, sigma)
            for r in range(u.shape[0]):
                for p in range(u.shape[1]):
                    if e[r, p] + es <= self.E_bar or arg is None:
                        if u[r, p] > best + 1e-15 or arg is None:
                            best, arg = u[r, p], Decision(r, p, s)
        return arg

    def update(self, *a):
        pass


@dataclass
class Threshold:
    gain_db_threshold: float
    name: str = "Threshold heuristic"

    def decide(self, m, q, u, e, d, theta, sigma, gain_db=None):
        s = m.floor[theta]
        if gain_db is not None and gain_db >= self.gain_db_threshold:
            return Decision(u.shape[0] - 1, 1 if u.shape[1] > 1 else 0, s)   # good channel: top rate, 2nd power
        return Decision(0, u.shape[1] - 1, s)                                # bad channel: lowest rate, max power

    def update(self, *a):
        pass


def make_policies(cfg, n_levels: int, n_powers: int) -> list:
    V = cfg.V
    mid_rate = [i for i, lv in enumerate(cfg.level_names) if lv == cfg.fixed_rate_level][0]
    p0 = cfg.powers_dbm.index(0.0)
    return [
        DPP(V, cfg.E_bar, cfg.D_bar),
        FixedMaxSecurity(),
        DPP(V, cfg.E_bar, cfg.D_bar, rate_fixed=mid_rate, name="Fixed rate (DPP over power, tier)"),
        MyopicGreedy(cfg.E_bar),
        Threshold(cfg.threshold_gain_db),
        DPP(V, cfg.E_bar, cfg.D_bar, use_floor=False, name="Ablation: no tier floor"),
        DPP(V, cfg.E_bar, cfg.D_bar, rate_fixed=n_levels - 1, name="Ablation: no rate adaptation"),
        DPP(V, cfg.E_bar, cfg.D_bar, power_fixed=p0, name="Ablation: no power control"),
        DPP(V, cfg.E_bar, cfg.D_bar, tier_fixed=2, name="Ablation: tier fixed at 2 (threat-unaware)"),
        # appended (decision 2026-10-13) so that earlier policies keep their order and results
        DPP(V, cfg.E_bar, cfg.D_bar, rate_fixed=0, tier_fixed=2, name="Combined: 8 B, tier 2, DPP over power"),
        FixedAction(0, n_powers - 1, 2, name="Combined: 8 B, tier 2, max. power"),
    ]
