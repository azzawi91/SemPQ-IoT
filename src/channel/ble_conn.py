"""BLE connection-event and sleep-floor energy of the node (decision 2026-10-13, external review point 6).

The node is the peripheral of a BLE connection and uses peripheral latency, so it wakes only for the connection
event that carries a record. One event with n data packets costs, in addition to the transmission already in the
record radio energy (air time x TX current):
  HFXO start-up    (t_power + t_debounce) x I_start_HFXO
  per data packet  RX ramp-up + reception of the central's packet (empty PDU, 80 us at 1 Mbps) + TX ramp-up
  acknowledgement  RX ramp-up + reception of the central's next packet (empty PDU) after the last data packet
The TX ramp-up lies inside the 150 us inter-frame space. Between events the node sleeps at I_sleep for the rest of
the slot. CPU activity of the BLE stack outside the record protection is not modelled. All values come from
configs/constants.yaml (nRF52840 PS v1.1); they are model quantities, not measurements.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .ble_phy import BleFrame

REPO = Path(__file__).resolve().parents[2]


def _c(const, k):
    return const["ble_connection"][k]["value"]


def event_energy_j(n_packets: int, variant: str = "nominal", const: dict | None = None) -> dict:
    """Connection-event overhead (J) for ``n_packets`` data packets. variant: nominal | fast_rampup | pessimistic_hfxo."""
    const = const or yaml.safe_load((REPO / "configs" / "constants.yaml").read_text())
    v = const["mcu"]["supply_v"]["value"]
    i_rx = const["radio_ble"]["rx_current_ma"]["value"] * 1e-3
    fr = BleFrame()
    t_empty = float(fr.airtime_s(0))  # central's empty PDU
    fast = variant == "fast_rampup"
    t_rxen = _c(const, "t_rxen_fast_us" if fast else "t_rxen_us") * 1e-6
    t_txen = _c(const, "t_txen_fast_us" if fast else "t_txen_us") * 1e-6
    pess = variant == "pessimistic_hfxo"
    t_hfxo = (_c(const, "t_hfxo_power_pess_us" if pess else "t_hfxo_power_us")
              + _c(const, "t_hfxo_debounce_pess_us" if pess else "t_hfxo_debounce_us")) * 1e-6
    i_hfxo = _c(const, "i_hfxo_start_pess_ua" if pess else "i_hfxo_start_ua") * 1e-6
    i_srx, i_stx = _c(const, "i_start_rx_ma") * 1e-3, _c(const, "i_start_tx_ma") * 1e-3
    e_hfxo = t_hfxo * i_hfxo * v
    e_rx_turn = (t_rxen * i_srx + t_empty * i_rx) * v
    e_tx_ramp = t_txen * i_stx * v
    e = e_hfxo + n_packets * (e_rx_turn + e_tx_ramp) + e_rx_turn
    # active radio time of the event besides the data packets (for the latency model)
    t_evt = t_hfxo + n_packets * (t_rxen + t_empty + _c(const, "t_ifs_us") * 1e-6) + t_rxen + t_empty
    return {"e_event_j": e, "e_hfxo_j": e_hfxo, "e_rx_turnaround_j": e_rx_turn, "e_tx_rampup_j": e_tx_ramp,
            "t_event_overhead_s": t_evt}


def sleep_energy_j(slot_s: float, active_s: float = 0.0, const: dict | None = None) -> float:
    const = const or yaml.safe_load((REPO / "configs" / "constants.yaml").read_text())
    return max(slot_s - active_s, 0.0) * _c(const, "i_sleep_ua") * 1e-6 * const["mcu"]["supply_v"]["value"]
