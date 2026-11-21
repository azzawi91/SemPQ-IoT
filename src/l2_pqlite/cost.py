"""On-device cost model for the wearable node (brief WP2): a MODEL built from published
measurements (configs/constants.yaml), never a hardware measurement.

compute  : cycles = sum over ledger operations of count x cycles(op), at the MCU clock;
           energy = cycles / f_clk x I_run x V.
radio    : every message is fragmented into BLE packets (fragment.py); each packet's air time is
           (preamble + access address + header + payload + CRC) bits at 1 Mbit/s (LE 1M PHY);
           energy = air time x I_tx (sent) or I_rx (received) x V. Inter-frame spacing, empty
           acknowledgement packets and radio ramp-up are not modelled (stated as a limitation).
Cycle counts measured on other Cortex-M4 parts (pqm4: STM32L4R5 at 24 MHz without wait states;
X25519: MK20DX256 at 72 MHz; AES: STM32F4) are assumed to transfer to the nRF52840 at 64 MHz.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.channel.ble_phy import BleFrame

from .fragment import ll_payload_sizes
from .primitives import Ledger

REPO = Path(__file__).resolve().parents[2]

# ledger operation -> constants key (per-call ops) ; block ops are listed separately
CALL_OPS = {
    "x25519_keygen": "x25519", "x25519_shared": "x25519",
    "mlkem512_keygen": "ml_kem_512_keygen", "mlkem512_encaps": "ml_kem_512_encaps",
    "mlkem512_decaps": "ml_kem_512_decaps",
    "mldsa44_keygen": "ml_dsa_44_keygen", "mldsa44_sign": "ml_dsa_44_sign",
    "mldsa44_verify": "ml_dsa_44_verify",
}
BLOCK_OPS = {"sha256_block": "sha256_compression", "aes128_block": "aes128_block",
             "ghash_block": "ghash_per_block", "aes128_keyschedule": "aes128_keyschedule"}
IGNORED = {"sha256", "hkdf", "hkdf_expand", "hmac_sha256", "aes128gcm"}  # costed via blocks


class MissingConstant(ValueError):
    pass


@dataclass
class CostModel:
    cycles: dict
    f_hz: float
    i_run_a: float
    v: float
    i_tx_a: float
    i_rx_a: float
    frame: BleFrame = BleFrame()

    @classmethod
    def from_yaml(cls, path: Path = REPO / "configs" / "constants.yaml", overrides: dict | None = None):
        c = yaml.safe_load(Path(path).read_text())
        cyc = {k: v["value"] for k, v in c["pqm4"]["cycles"].items()}
        cyc.update(overrides or {})
        return cls(cycles=cyc, f_hz=c["mcu"]["clock_mhz"]["value"] * 1e6,
                   i_run_a=c["mcu"]["run_current_ma"]["value"] * 1e-3, v=c["mcu"]["supply_v"]["value"],
                   i_tx_a=c["radio_ble"]["tx_current_ma"]["value"] * 1e-3,
                   i_rx_a=c["radio_ble"]["rx_current_ma"]["value"] * 1e-3)

    def _c(self, key: str) -> float:
        v = self.cycles.get(key)
        if v is None:
            raise MissingConstant(f"constant '{key}' has no sourced value (configs/constants.yaml)")
        return float(v)

    def compute(self, led: Ledger) -> dict:
        per_op, total = {}, 0.0
        for op, n in led.ops.items():
            if op in CALL_OPS:
                c = n * self._c(CALL_OPS[op])
            elif op in BLOCK_OPS:
                c = n * self._c(BLOCK_OPS[op])
            elif op in IGNORED:
                continue
            else:
                raise KeyError(f"no cost mapping for ledger operation '{op}'")
            per_op[op] = c
            total += c
        t = total / self.f_hz
        return {"cycles": total, "time_s": t, "energy_j": t * self.i_run_a * self.v, "per_op_cycles": per_op}

    def radio(self, sent_msgs: list[int], recv_msgs: list[int]) -> dict:
        def air(msgs):
            pk = [p for m in msgs for p in ll_payload_sizes(m)]
            bits = sum(self.frame.preamble_bits + self.frame.protected_bits(p) for p in pk)
            return len(pk), bits / 1e6
        n_tx, t_tx = air(sent_msgs)
        n_rx, t_rx = air(recv_msgs)
        return {"packets_tx": n_tx, "packets_rx": n_rx, "airtime_tx_s": t_tx, "airtime_rx_s": t_rx,
                "bytes_tx": sum(sent_msgs), "bytes_rx": sum(recv_msgs),
                "energy_tx_j": t_tx * self.i_tx_a * self.v, "energy_rx_j": t_rx * self.i_rx_a * self.v}

    def total(self, led: Ledger, sent: list[int], recv: list[int]) -> dict:
        c, r = self.compute(led), self.radio(sent, recv)
        return {**c, **r, "energy_total_j": c["energy_j"] + r["energy_tx_j"] + r["energy_rx_j"]}
