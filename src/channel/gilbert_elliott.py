"""Gilbert-Elliott two-state burst outage process (slot-level).

States: G (good) and B (bad). Transition probabilities per slot: p = P(G -> B), r = P(B -> G).
Loss probability in each state: e_g (usually 0) and e_b (1 for a hard outage).
Stationary P(B) = p / (p + r); mean sojourn in B = 1 / r slots.

``from_outage`` parameterises the chain by the quantities the WP2 experiments sweep:
mean outage duration (slots) and long-run fraction of time in outage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GilbertElliott:
    p_gb: float
    p_bg: float
    e_g: float = 0.0
    e_b: float = 1.0

    def __post_init__(self):
        for name in ("p_gb", "p_bg", "e_g", "e_b"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name}={v} outside [0, 1]")
        if self.p_gb + self.p_bg == 0:
            raise ValueError("chain is not ergodic (p_gb = p_bg = 0)")

    @classmethod
    def from_outage(cls, mean_outage_slots: float, outage_fraction: float,
                    e_g: float = 0.0, e_b: float = 1.0) -> "GilbertElliott":
        if mean_outage_slots < 1:
            raise ValueError("mean outage must be >= 1 slot")
        if not 0.0 <= outage_fraction < 1.0:
            raise ValueError("outage fraction must be in [0, 1)")
        r = 1.0 / mean_outage_slots
        p = outage_fraction * r / (1.0 - outage_fraction)
        if p > 1.0:
            raise ValueError("infeasible (mean outage, fraction) pair: p_gb > 1")
        return cls(p_gb=p, p_bg=r, e_g=e_g, e_b=e_b)

    @property
    def stationary_bad(self) -> float:
        return self.p_gb / (self.p_gb + self.p_bg)

    @property
    def mean_loss(self) -> float:
        pb = self.stationary_bad
        return (1 - pb) * self.e_g + pb * self.e_b

    def states(self, n_slots: int, rng: np.random.Generator, start: str = "stationary") -> np.ndarray:
        """Boolean array, True where the slot is in state B."""
        u = rng.random(n_slots)
        if start == "stationary":
            bad = rng.random() < self.stationary_bad
        elif start in ("G", "B"):
            bad = start == "B"
        else:
            raise ValueError(start)
        out = np.empty(n_slots, dtype=bool)
        for t in range(n_slots):  # sequential by definition; n_slots is moderate per run
            out[t] = bad
            bad = (u[t] >= self.p_bg) if bad else (u[t] < self.p_gb)
        return out

    def states_fast(self, n_slots: int, rng: np.random.Generator) -> np.ndarray:
        """Same process as ``states`` (stationary start), generated from geometric sojourn times
        instead of slot-by-slot transitions; much faster for long traces. Uses a different random
        stream than ``states``, so the two are equal in distribution, not sample by sample."""
        bad = rng.random() < self.stationary_bad
        out, filled = [], 0
        while filled < n_slots:
            k = max(64, 4 * int(n_slots * min(self.p_gb, self.p_bg) + 1))
            g = rng.geometric(self.p_gb, k) if self.p_gb > 0 else np.full(k, n_slots)
            b = rng.geometric(self.p_bg, k) if self.p_bg > 0 else np.full(k, n_slots)
            seq = np.empty(2 * k, dtype=np.int64)
            seq[0::2], seq[1::2] = (b, g) if bad else (g, b)
            vals = np.empty(2 * k, dtype=bool)
            vals[0::2], vals[1::2] = bad, not bad
            out.append(np.repeat(vals, seq))
            filled += int(seq.sum())
        return np.concatenate(out)[:n_slots]

    def losses(self, n_slots: int, rng: np.random.Generator, start: str = "stationary"):
        """(state_bad, lost) arrays; a slot is lost with probability e_b in B and e_g in G."""
        bad = self.states(n_slots, rng, start)
        p = np.where(bad, self.e_b, self.e_g)
        return bad, rng.random(n_slots) < p


def run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True runs (used by tests and outage statistics)."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
