"""Handshake over an intermittent BLE link (brief WP2: completion probability and latency vs
outage duration and loss rate).

Model
-----
* Time advances in connection events of ``slot_s`` seconds. In each event one link-layer packet
  can be sent in the scheduled direction; the link layer retransmits a lost packet in the next
  event (BLE acknowledged data), so each fragment is attempted until delivered.
* A packet is delivered with probability 1 - PER(instantaneous Eb/N0, payload) outside an outage
  and with probability 1 - e_b inside one (src/channel: block Rician fading, BLE PER model,
  Gilbert-Elliott outages). The channel trace is shared by all messages of a handshake.
* Before sending, a party spends its compute time (node: cost model; gateway: ``gw_compute_s``);
  compute runs during the events that elapse meanwhile.
* The handshake fails if it is not complete within ``deadline_s``.
Simplifications (stated in the paper): one packet per connection event, no supervision-timeout
link loss, no reassembly loss (fragments are retransmitted until delivered).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.channel.ble_phy import BerModel, BleFrame, packet_error_rate
from src.channel.fading import block_fading_trace, lin_to_db
from src.channel.gilbert_elliott import GilbertElliott

from .fragment import ll_payload_sizes


@dataclass
class Step:
    sender: str          # "node" or "gw"
    size: int            # message bytes
    compute_s: float     # sender compute before sending


@dataclass
class LinkParams:
    avg_ebn0_db: float
    k_factor: float
    block_len: int
    outage: GilbertElliott | None
    slot_s: float = 0.0075
    deadline_s: float = 10.0


def simulate(steps: list[Step], lp: LinkParams, ber: BerModel, rng: np.random.Generator,
             n_trials: int, frame: BleFrame = BleFrame()) -> dict:
    n_slots = int(math.ceil(lp.deadline_s / lp.slot_s))
    plan = [(int(math.ceil(st.compute_s / lp.slot_s)), ll_payload_sizes(st.size)) for st in steps]
    sizes = sorted({p for _, pls in plan for p in pls})
    e_b = lp.outage.e_b if lp.outage is not None else 0.0
    lat, done, attempts = [], 0, []
    for _ in range(n_trials):
        gain = block_fading_trace(n_slots, lp.block_len, lp.k_factor, rng)
        snr = lp.avg_ebn0_db + lin_to_db(np.maximum(gain, 1e-300))
        bad = lp.outage.states_fast(n_slots, rng) if lp.outage is not None else np.zeros(n_slots, bool)
        u = rng.random(n_slots)
        # per-slot delivery outcome for every payload size used (vectorised once per trial)
        p_out = np.where(bad, 1.0 - e_b, 1.0)
        ok_by_size = {pl: (u < (1.0 - packet_error_rate(snr, pl, ber, frame)) * p_out) for pl in sizes}
        t, ok, tries = 0, True, 0
        for wait, pls in plan:
            t += wait
            for pl in pls:
                nxt = np.flatnonzero(ok_by_size[pl][t:]) if t < n_slots else np.empty(0, int)
                if len(nxt) == 0:
                    tries += max(0, n_slots - t)
                    ok = False
                    break
                tries += int(nxt[0]) + 1
                t += int(nxt[0]) + 1
            if not ok:
                break
        attempts.append(tries)
        if ok:
            done += 1
            lat.append(t * lp.slot_s)
    lat = np.asarray(lat)
    return {"trials": n_trials, "completion_prob": done / n_trials,
            "latency_mean_s": float(lat.mean()) if len(lat) else None,
            "latency_p95_s": float(np.percentile(lat, 95)) if len(lat) else None,
            "attempts_mean": float(np.mean(attempts))}
