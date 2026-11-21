"""Build configs/wp3_inputs.yaml from the signed-off WP1 and WP2 results (no new numbers).

    python experiments/wp3_build_inputs.py

* Rate levels and macro-F1 / accuracy per level: proposed encoder, configuration B, DS2,
  mean over seeds 0-4 (r/wp1-configB, results/wp1/aggregate/summary.json).
* Per-record node cost (compute cycles) for each payload size, handshake and re-key cost
  (compute cycles, node TX/RX bytes) per tier: r/wp2-0e5a9f6 results/wp2/{records,handshakes}.json.
* Handshake latency without outages at 18 dB: r/wp2-0e5a9f6 results/wp2/linksim.json.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
WP1_REF, WP2_REF = "origin/r/wp1-configB", "origin/r/wp2-0e5a9f6"


LOCAL = {}  # set by --wp1-summary / --wp2-dir: read files from disk instead of result branches


def git_json(ref: str, path: str):
    if ref in LOCAL:
        return json.loads(Path(LOCAL[ref](path)).read_text())
    out = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=REPO)
    return json.loads(out)


def git_commit(ref: str) -> str:
    if ref in LOCAL:
        return "local files"
    return subprocess.check_output(["git", "rev-parse", "--short", ref], cwd=REPO, text=True).strip()


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--wp1-summary", type=Path, help="local results/wp1/aggregate/summary.json")
    ap.add_argument("--wp2-dir", type=Path, help="local results/wp2 directory")
    ap.add_argument("--out", type=Path, default=REPO / "configs" / "wp3_inputs.yaml")
    ap.add_argument("--wp1-cv-dir", type=Path, help="DS1 cross-validation results (semantic_seed*.json); "
                    "if given, the scheduler's F1 per level comes from DS1 CV, not from the DS2 test set")
    a = ap.parse_args()
    if a.wp1_summary:
        LOCAL[WP1_REF] = lambda _p, f=a.wp1_summary: f
    if a.wp2_dir:
        LOCAL[WP2_REF] = lambda p_, d=a.wp2_dir: d / Path(p_).name
    s1 = git_json(WP1_REF, "results/wp1/aggregate/summary.json")
    levels, enc_macs = [], None
    for r in s1["clean"]:
        if r["method"] == "semantic":
            enc_macs = int(r["macs"])
            levels.append({"level": r["level"], "payload_bytes": int(r["bytes"]),
                           "macro_f1": round(float(r["f1_mean"]), 6), "accuracy": round(float(r["acc_mean"]), 6),
                           "f1_source": "DS2 test (mean over seeds)"})
    levels.sort(key=lambda d: d["payload_bytes"])
    if a.wp1_cv_dir:  # decision 2026-10-12: no test-set information in the scheduler design
        import statistics
        cv = [json.loads(f.read_text()) for f in sorted(a.wp1_cv_dir.glob("semantic_seed*.json"))]
        if not cv:
            raise SystemExit(f"no CV results in {a.wp1_cv_dir}")
        for lv in levels:
            k, b = (int(x) for x in lv["level"][1:].split("_b"))
            vals = [g["metrics"]["macro_f1"] for d in cv for g in d["val_grid"]
                    if g["channels"] == k and g["bits"] == b and g["snr_db"] is None and g["loss_p"] is None]
            lv["macro_f1_ds2_test"] = lv["macro_f1"]
            lv["macro_f1"] = round(statistics.mean(vals), 6)
            lv["f1_source"] = f"DS1 4-fold patient CV, out-of-fold, mean over {len(vals)} seeds"

    recs = git_json(WP2_REF, "results/wp2/records.json")["rows"]
    rec_by_payload = {int(r["payload_bytes"]): {"record_bytes": int(r["record_bytes"]), "cycles": float(r["cycles"])}
                      for r in recs}
    # sizes not run in WP2 are computed with the same code and constants; the sizes that WP2 did
    # run must be reproduced exactly, which checks that code and constants have not drifted
    import sys
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "experiments"))
    from src.l2_pqlite.cost import CostModel
    from wp2_pqlite import run_records

    sizes = sorted({lv["payload_bytes"] for lv in levels} | set(rec_by_payload))
    fresh = {int(r["payload_bytes"]): r for r in run_records(CostModel.from_yaml(), sizes)}
    for p_, v in rec_by_payload.items():
        if abs(fresh[p_]["cycles"] - v["cycles"]) > 1e-6 or fresh[p_]["record_bytes"] != v["record_bytes"]:
            raise SystemExit(f"record cost for {p_} B differs from WP2 results: code or constants drifted")
    rec_by_payload = {p_: {"record_bytes": int(r["record_bytes"]), "cycles": float(r["cycles"]),
                           "source": "WP2 results" if p_ in rec_by_payload else "recomputed (same code/constants)"}
                      for p_, r in fresh.items()}

    hs = {r["protocol"]: r for r in git_json(WP2_REF, "results/wp2/handshakes.json")["rows"]}
    t0 = hs["PQ-Lite tier 0 (classical, PSK)"]
    t1 = hs["PQ-Lite tier 1 (hybrid, cached cert.)"]
    t2 = hs["PQ-Lite tier 2 (hybrid + immediate re-key)"]
    rekey = {"cycles": t2["cycles"] - t1["cycles"], "bytes_tx": t2["bytes_tx"] - t1["bytes_tx"],
             "bytes_rx": t2["bytes_rx"] - t1["bytes_rx"]}
    handshake = {0: t0, 1: t1, 2: t1}  # tier 2 opens with a tier-1 handshake (+ immediate re-key)
    tiers = {}
    for s, h in handshake.items():
        tiers[s] = {"handshake_cycles": float(h["cycles"]), "handshake_bytes_tx": int(h["bytes_tx"]),
                    "handshake_bytes_rx": int(h["bytes_rx"]),
                    "immediate_rekey": s == 2}
    sim = git_json(WP2_REF, "results/wp2/linksim.json")["rows"]

    def lat(proto):
        v = [r["latency_mean_s"] for r in sim if r["protocol"] == proto and r["ebn0_db"] == 18.0
             and r["outage_fraction"] == 0.0]
        return float(sum(v) / len(v))

    tiers[0]["handshake_latency_s"] = lat("PQ-Lite tier 0 (classical, PSK)")
    tiers[1]["handshake_latency_s"] = lat("PQ-Lite tier 1 (hybrid, cached cert.)")
    tiers[2]["handshake_latency_s"] = lat("PQ-Lite tier 2 (hybrid + immediate re-key)")

    out = {
        "provenance": {"wp1": (str(a.wp1_summary) if a.wp1_summary else
                               f"{WP1_REF} @ {git_commit(WP1_REF)} results/wp1/aggregate/summary.json (semantic, DS2, seeds 0-4)"),
                       "wp2": (str(a.wp2_dir) if a.wp2_dir else
                               f"{WP2_REF} @ {git_commit(WP2_REF)} results/wp2/{{records,handshakes,linksim}}.json")},
        "rate_levels": levels,
        "record": {str(k): v for k, v in rec_by_payload.items()},
        "tiers": {str(k): v for k, v in tiers.items()},
        "rekey": rekey,
        "encoder_macs": enc_macs,
    }
    path = a.out
    path.write_text("# GENERATED by experiments/wp3_build_inputs.py - do not edit by hand\n"
                    + yaml.safe_dump(out, sort_keys=False))
    print(yaml.safe_dump(out, sort_keys=False))


if __name__ == "__main__":
    main()
