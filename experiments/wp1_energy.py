"""Per-beat node energy of every L1 method (decision 2026-10-12, review 1): device computation
(MACs x cycles/MAC, INT8 CNN figures from configs/constants.yaml) + PQ-Lite record protection
(AES-GCM and ratchet, from the WP2 cost model) + BLE transmission of the record (fragmented into
BLE packets). Added 2026-10-13 (external review): the BLE connection-event overhead of the event carrying the
record (HFXO start-up, radio ramp-ups, reception of the central's packets; src/channel/ble_conn.py) and the
sleep floor of the rest of the slot, reported as extra columns with sensitivity variants.
A MODEL from published measurements, not a hardware measurement.

    python experiments/wp1_energy.py --summary results/wp1/aggregate/summary.json

Writes results/wp1/energy.json and tables/wp1_energy.tex.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))


def main():
    import yaml

    from src.channel import ble_conn
    from src.l2_pqlite.cost import CostModel
    from wp2_pqlite import run_records

    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=REPO / "results" / "wp1" / "aggregate" / "summary.json")
    a = ap.parse_args()
    s = json.loads(a.summary.read_text())
    const = yaml.safe_load((REPO / "configs" / "constants.yaml").read_text())
    cpm = {k: const["nn_inference"][k]["value"] for k in ("cycles_per_mac", "cycles_per_mac_pessimistic")}
    cm = CostModel.from_yaml()
    slot_s = float(yaml.safe_load((REPO / "configs" / "wp3.yaml").read_text())["slot_s"])
    rows = [r for r in s["clean"] if r["method"] in ("semantic", "raw", "dwt", "jscc", "label", "rr_only")]
    recs = {int(r["payload_bytes"]): r for r in run_records(cm, sorted({int(r["bytes"]) for r in rows}))}
    out = []
    for r in rows:
        b = int(r["bytes"])
        rec = recs[b]
        e_tx = rec["energy_tx_j"]
        e_rec = rec["energy_j"]
        row = {"method": r["method"], "level": r["level"], "payload_bytes": b, "record_bytes": rec["record_bytes"],
               "packets": rec["packets_tx"], "airtime_us": rec["airtime_tx_s"] * 1e6, "macs": int(r["macs"]),
               "e_record_compute_j": e_rec, "e_radio_j": e_tx, "f1_mean": r["f1_mean"]}
        for k, c in cpm.items():
            e_dev = r["macs"] * c / cm.f_hz * cm.i_run_a * cm.v
            row[f"e_device_j_{k}"] = e_dev
            row[f"e_total_j_{k}"] = e_dev + e_rec + e_tx
        for var in ("nominal", "fast_rampup", "pessimistic_hfxo"):
            ev = ble_conn.event_energy_j(int(rec["packets_tx"]), var, const)
            row[f"e_conn_event_j_{var}"] = ev["e_event_j"]
        ev = ble_conn.event_energy_j(int(rec["packets_tx"]), "nominal", const)
        active = (r["macs"] * cpm["cycles_per_mac"] / cm.f_hz + rec["time_s"] + rec["airtime_tx_s"]
                  + ev["t_event_overhead_s"])
        row["slot_s"] = slot_s
        row["e_sleep_j"] = ble_conn.sleep_energy_j(slot_s, active, const)
        row["e_total_conn_j"] = row["e_total_j_cycles_per_mac"] + ev["e_event_j"]
        row["e_total_conn_sleep_j"] = row["e_total_conn_j"] + row["e_sleep_j"]
        out.append(row)
    res = REPO / "results" / "wp1"
    res.mkdir(parents=True, exist_ok=True)
    (res / "energy.json").write_text(json.dumps({"cycles_per_mac": cpm, "f_hz": cm.f_hz, "rows": out}, indent=1) + "\n")
    import pandas as pd

    df = pd.DataFrame(out)
    tab = pd.DataFrame({"Method": df.method, "Level": df.level, "Bytes": df.payload_bytes, "Packets": df.packets,
                        "Device (uJ)": (df.e_device_j_cycles_per_mac * 1e6).map("{:.1f}".format),
                        "Record (uJ)": (df.e_record_compute_j * 1e6).map("{:.1f}".format),
                        "Radio (uJ)": (df.e_radio_j * 1e6).map("{:.1f}".format),
                        "Total (uJ)": (df.e_total_j_cycles_per_mac * 1e6).map("{:.1f}".format),
                        "Total, pessimistic (uJ)": (df.e_total_j_cycles_per_mac_pessimistic * 1e6).map("{:.1f}".format),
                        "Conn. event (uJ)": (df.e_conn_event_j_nominal * 1e6).map("{:.1f}".format),
                        "Sleep (uJ)": (df.e_sleep_j * 1e6).map("{:.1f}".format),
                        "Total incl. BLE (uJ)": (df.e_total_conn_sleep_j * 1e6).map("{:.1f}".format)})
    (REPO / "tables").mkdir(exist_ok=True)
    (REPO / "tables" / "wp1_energy.tex").write_text(tab.to_latex(index=False, escape=True))
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
