"""L3 system model: per-slot state, decisions, utility, energy and delay (brief WP3).

Slot t carries one beat decision of one wearable node.
Random state  omega(t) = (Theta(t), g(t), o(t)): threat level (Markov chain on {0, 1, 2}),
              block Rician power gain g(t) (unit mean) and Gilbert-Elliott outage indicator o(t);
              observed at the start of the slot. sigma(t) = tier of the current session.
Decision      x(t) = (r, p, s): rate level r (WP1), TX power p in dBm, security tier s (WP2),
              subject to the hard floor s >= s_min(Theta(t)).
Delivery      q = (1 - PER(gamma0 * g * 10^(p/10), LL payload of the record)) * (1 - o)
Utility       u = q * F1(r)                     (expected delivered macro-F1; WP1 DS2 curve)
Energy        e = E_enc + E_rec(r) + E_tx(r, p) + 1[s != sigma] E_hs(s) + 1[s = 2] E_rekey / K
              (E_enc: INT8 encoder inference per beat, MACs x cycles/MAC; decision 2026-10-12)
Delay         d = (1 - q) * slot_s + 1[s != sigma] T_hs(s)
All energies come from configs/constants.yaml and configs/wp3_inputs.yaml (WP1/WP2 results);
they are model quantities, not measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.channel.ble_phy import BerModel, BleFrame, packet_error_rate
from src.l2_pqlite.fragment import ll_payload_sizes

REPO = Path(__file__).resolve().parents[2]


@dataclass
class SystemModel:
    levels: list[dict]           # rate levels: payload_bytes, macro_f1, record_bytes, record_cycles
    powers_dbm: list[float]
    i_tx_a: dict                 # power (dBm) -> system TX current (A)
    tiers: dict                  # tier -> handshake energy (J), latency (s)
    rekey_energy_j: float
    rekey_every: int
    f_hz: float
    i_run_a: float
    v: float
    gamma0_db: float             # average Eb/N0 at the receiver for 0 dBm
    slot_s: float
    floor: dict                  # theta -> s_min
    ber: BerModel
    e_enc: float = 0.0           # encoder inference energy per beat (J)
    frame: BleFrame = BleFrame()

    # --------------------------------------------------------------- construction
    @classmethod
    def build(cls, cfg, ber: BerModel):
        const = yaml.safe_load((REPO / "configs" / "constants.yaml").read_text())
        inp_path = Path(cfg.get("inputs_path") or "configs/wp3_inputs.yaml")
        inp = yaml.safe_load((inp_path if inp_path.is_absolute() else REPO / inp_path).read_text())
        f_hz = const["mcu"]["clock_mhz"]["value"] * 1e6
        i_run = const["mcu"]["run_current_ma"]["value"] * 1e-3
        v = const["mcu"]["supply_v"]["value"]
        over = const["radio_ble"]["tx_system_overhead_ma"]["value"]
        radio = const["radio_ble"]["tx_radio_only_ma"]
        i_tx = {float(p): (radio[f"{p:+g}" if p != 0 else "0"]["value"] + over) * 1e-3 for p in cfg.powers_dbm}
        i_tx0 = const["radio_ble"]["tx_current_ma"]["value"] * 1e-3   # WP2 handshake radio (0 dBm)
        i_rx = const["radio_ble"]["rx_current_ma"]["value"] * 1e-3
        frame = BleFrame()

        def air(nbytes):
            return sum(frame.preamble_bits + frame.protected_bits(p) for p in ll_payload_sizes(nbytes)) / 1e6

        levels = []
        for lv in inp["rate_levels"]:
            rec = inp["record"][str(lv["payload_bytes"])]
            if len(ll_payload_sizes(rec["record_bytes"])) != 1:
                raise ValueError("a record must fit one BLE packet in this model")
            levels.append({**lv, "record_bytes": rec["record_bytes"],
                           "ll_bytes": ll_payload_sizes(rec["record_bytes"])[0],
                           "e_rec": rec["cycles"] / f_hz * i_run * v,
                           "airtime": air(rec["record_bytes"])})
        tiers = {}
        for s, t in inp["tiers"].items():
            e = (t["handshake_cycles"] / f_hz * i_run * v + air(t["handshake_bytes_tx"]) * i_tx0 * v
                 + air(t["handshake_bytes_rx"]) * i_rx * v)
            tiers[int(s)] = {"e_hs": e, "t_hs": t["handshake_latency_s"]}
        rk = inp["rekey"]
        e_rekey = rk["cycles"] / f_hz * i_run * v + air(rk["bytes_tx"]) * i_tx0 * v + air(rk["bytes_rx"]) * i_rx * v
        tiers[2]["e_hs"] += e_rekey   # tier 2 opens with a tier-1 handshake plus an immediate re-key
        cpm_key = cfg.get("cycles_per_mac", "cycles_per_mac")
        e_enc = (inp.get("encoder_macs") or 0) * const["nn_inference"][cpm_key]["value"] / f_hz * i_run * v \
            if cfg.get("include_encoder_energy", True) else 0.0
        return cls(levels, [float(p) for p in cfg.powers_dbm], i_tx, tiers, e_rekey, cfg.rekey_every,
                   f_hz, i_run, v, cfg.gamma0_db, cfg.slot_s, {int(k): int(v_) for k, v_ in cfg.floor.items()},
                   ber, frame=frame, e_enc=e_enc)

    # --------------------------------------------------------------- per-slot quantities
    def tables(self, g: float, o: bool):
        """Delivery probability, utility, record+TX energy and loss delay for every (r, p)."""
        R, P = len(self.levels), len(self.powers_dbm)
        q = np.zeros((R, P))
        for i, lv in enumerate(self.levels):
            snr = self.gamma0_db + np.asarray(self.powers_dbm) + 10 * np.log10(max(g, 1e-300))
            q[i] = (1.0 - packet_error_rate(snr, lv["ll_bytes"], self.ber, self.frame)) * (0.0 if o else 1.0)
        f1 = np.array([lv["macro_f1"] for lv in self.levels])[:, None]
        e_rec = np.array([lv["e_rec"] for lv in self.levels])[:, None]
        air = np.array([lv["airtime"] for lv in self.levels])[:, None]
        itx = np.array([self.i_tx_a[p] for p in self.powers_dbm])[None, :]
        e = self.e_enc + e_rec + air * itx * self.v
        return q, q * f1, e, (1.0 - q) * self.slot_s

    def tables_all(self, gain: np.ndarray, outage: np.ndarray):
        """Vectorised ``tables`` for a whole trace: arrays of shape (T, R, P)."""
        g_db = 10 * np.log10(np.maximum(gain, 1e-300))[:, None]
        snr = self.gamma0_db + np.asarray(self.powers_dbm)[None, :] + g_db          # (T, P)
        avail = (~np.asarray(outage, bool)).astype(float)[:, None]
        q = np.stack([(1.0 - packet_error_rate(snr, lv["ll_bytes"], self.ber, self.frame)) * avail
                      for lv in self.levels], axis=1)                                 # (T, R, P)
        f1 = np.array([lv["macro_f1"] for lv in self.levels])[None, :, None]
        e_rec = np.array([lv["e_rec"] for lv in self.levels])[:, None]
        air = np.array([lv["airtime"] for lv in self.levels])[:, None]
        itx = np.array([self.i_tx_a[p] for p in self.powers_dbm])[None, :]
        e = np.broadcast_to(self.e_enc + e_rec + air * itx * self.v, q.shape)
        return q, q * f1, e, (1.0 - q) * self.slot_s

    def security_cost(self, s: int, sigma: int):
        e = (self.tiers[s]["e_hs"] if s != sigma else 0.0) + (self.rekey_energy_j / self.rekey_every if s == 2 else 0.0)
        d = self.tiers[s]["t_hs"] if s != sigma else 0.0
        return e, d
