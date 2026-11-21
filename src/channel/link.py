"""Slot-level link simulator: block Rician fading x BLE 1M PHY PER x Gilbert-Elliott outages.

One packet is attempted per slot. A packet is delivered iff the slot is not lost to an
outage and the PHY draw succeeds at the instantaneous Eb/N0 (average Eb/N0 times |h|^2).
Every random draw comes from the Generator passed in, so a run is a function of its seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ble_phy import BerModel, BleFrame, packet_error_rate
from .fading import block_fading_trace, lin_to_db
from .gilbert_elliott import GilbertElliott


@dataclass
class LinkConfig:
    avg_ebn0_db: float
    k_factor: float = 3.0
    block_len: int = 1
    outage: GilbertElliott | None = None
    frame: BleFrame = field(default_factory=BleFrame)


class LinkSimulator:
    def __init__(self, cfg: LinkConfig, ber_model: BerModel, rng: np.random.Generator):
        self.cfg, self.ber, self.rng = cfg, ber_model, rng

    def run(self, payload_bytes) -> dict:
        pb = np.asarray(payload_bytes)
        n = pb.size
        gain = block_fading_trace(n, self.cfg.block_len, self.cfg.k_factor, self.rng)
        inst_db = self.cfg.avg_ebn0_db + lin_to_db(np.maximum(gain, 1e-300))
        per = packet_error_rate(inst_db, pb, self.ber, self.cfg.frame)
        phy_ok = self.rng.random(n) >= per
        if self.cfg.outage is not None:
            bad, lost = self.cfg.outage.losses(n, self.rng)
        else:
            bad = lost = np.zeros(n, dtype=bool)
        return {
            "delivered": phy_ok & ~lost,
            "per": per,
            "inst_ebn0_db": inst_db,
            "outage_state": bad,
            "outage_lost": lost,
            "airtime_s": self.cfg.frame.airtime_s(pb),
        }


def expected_delivery(avg_ebn0_db: float, payload_bytes: int, k_factor: float, ber_model: BerModel,
                      frame: BleFrame = BleFrame(), n_mc: int = 200_000, rng=None) -> float:
    """Fading-averaged delivery probability (no outages), by Monte Carlo over |h|^2."""
    from .fading import rician_power_gains

    rng = rng or np.random.default_rng(0)
    g = rician_power_gains(n_mc, k_factor, rng)
    per = packet_error_rate(avg_ebn0_db + lin_to_db(np.maximum(g, 1e-300)), payload_bytes, ber_model, frame)
    return float(np.mean(1.0 - per))
