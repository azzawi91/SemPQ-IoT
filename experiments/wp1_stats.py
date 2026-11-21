"""WP1 statistics added after internal review 1 (decision 2026-10-12), from the per-beat DS2
predictions saved by wp1_train.py (results/wp1/<run>/seed*_pred.npz):

1. Record-level bootstrap: DS2 records are resampled with replacement (B resamples); for each
   resample the macro-F1 of every seed is computed and averaged over seeds. Reported: 95 %
   percentile intervals of each method's macro-F1 and of the difference proposed - baseline.
2. DS2 without record 202 (same subject as DS1 record 201): macro-F1 from the seed files.
3. Whole-record loss over the BLE link: each beat is one PQ-Lite record (payload + 24 B),
   fragmented into BLE packets; it is delivered only if every packet survives (block Rician
   fading, Gilbert-Elliott outages, GFSK PER at the highest transmit power of the scheduler). A lost beat is classified as the
   training-majority class N at the gateway. Realised macro-F1 over all DS2 beats per seed.

    python experiments/wp1_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

METHODS = ["semantic", "raw", "dwt", "jscc", "label", "rr_only"]
CLASSES = 4


def macro_f1(y, p):
    f = []
    for c in range(CLASSES):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def load_preds(res: Path):
    out = {}
    for m in METHODS:
        files = sorted((res / f"{m}_mitbih").glob("seed*_pred.npz"))
        out[m] = [dict(np.load(f)) for f in files]
    return out


def main():
    import yaml
    from omegaconf import OmegaConf

    from src.channel.ble_phy import BerModel, BleFrame, gfsk_ber_curve, packet_error_rate
    from src.l2_pqlite.fragment import ll_payload_sizes
    from src.l3_scheduler.sim import traces

    res = REPO / "results" / "wp1"
    summ = json.loads((res / "aggregate" / "summary.json").read_text())
    preds = load_preds(res)
    nseed = min(len(v) for v in preds.values())
    if nseed == 0:
        raise SystemExit("no per-beat predictions found (seed*_pred.npz)")
    # level used for each method: the proposed encoder at its smallest level; baselines at the
    # level with the highest mean DS2 macro-F1 (as in the paired tests)
    clean = [r for r in summ["clean"] if r["method"] in METHODS]
    best = {m: max((r for r in clean if r["method"] == m), key=lambda r: r["f1_mean"]) for m in METHODS}
    best["semantic"] = min((r for r in clean if r["method"] == "semantic"), key=lambda r: r["bytes"])
    lvl = {m: best[m]["level"] for m in METHODS}
    nbytes = {m: int(best[m]["bytes"]) for m in METHODS}

    # ---- 1. record-level bootstrap
    y0, rec0 = preds["semantic"][0]["y"], preds["semantic"][0]["record"]
    records = np.unique(rec0)
    idx_by_rec = {r: np.flatnonzero(rec0 == r) for r in records}
    for m in METHODS:  # all prediction files share the DS2 beat order
        for d in preds[m]:
            assert np.array_equal(d["y"], y0) and np.array_equal(d["record"], rec0)
    rng = np.random.default_rng(20261012)
    B = 2000
    boot = {m: np.empty(B) for m in METHODS}
    for b in range(B):
        idx = np.concatenate([idx_by_rec[r] for r in rng.choice(records, len(records), replace=True)])
        for m in METHODS:
            boot[m][b] = np.mean([macro_f1(y0[idx], d[lvl[m]][idx]) for d in preds[m][:nseed]])
    point = {m: float(np.mean([macro_f1(y0, d[lvl[m]]) for d in preds[m][:nseed]])) for m in METHODS}
    ci = {m: [float(np.percentile(boot[m], 2.5)), float(np.percentile(boot[m], 97.5))] for m in METHODS}
    diff = {}
    for m in METHODS[1:]:
        dd = boot["semantic"] - boot[m]
        diff[m] = {"point": point["semantic"] - point[m], "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                   "frac_positive": float(np.mean(dd > 0))}

    # ---- 2. DS2 without record 202
    no202 = {}
    for m in METHODS:
        vals = []
        for f in sorted((res / f"{m}_mitbih").glob("seed[0-9].json")):
            d = json.loads(f.read_text())
            k, bits = (int(x) for x in lvl[m][1:].split("_b"))
            g = [r for r in d["test_without_records"]["grid"] if r["channels"] == k and r["bits"] == bits]
            vals.append(g[0]["metrics"]["macro_f1"])
        no202[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "records_excluded": [202]}

    # ---- 3. whole-record loss over the link
    cfg = OmegaConf.load(REPO / "configs" / "wp3.yaml")
    cfg.slots = len(y0)
    grid = np.asarray(cfg.ber.ebn0_grid_db, float)
    ber = BerModel("gfsk_mc", grid, gfsk_ber_curve(grid, np.random.default_rng(cfg.ber.seed), n_bits=cfg.ber.n_bits))
    frame = BleFrame()
    overhead = 24  # PQ-Lite record header + tag (results/wp2/records.json)
    loss_rows = []
    for g0 in cfg.gamma0_sweep:
        for s in range(nseed):
            tr = traces(cfg, np.random.default_rng(s))
            p_tx = float(max(cfg.powers_dbm))
            snr = g0 + p_tx + 10 * np.log10(np.maximum(tr["gain"], 1e-300))
            for m in METHODS:
                pk = ll_payload_sizes(nbytes[m] + overhead)
                ok = np.ones(len(y0), bool) & ~tr["outage"]
                u = np.random.default_rng(1000 + s).random((len(pk), len(y0)))
                for j, p in enumerate(pk):
                    ok &= u[j] >= packet_error_rate(snr, p, ber, frame)
                yhat = np.where(ok, preds[m][s][lvl[m]], 0)
                loss_rows.append({"gamma0_db": float(g0), "tx_power_dbm": p_tx, "seed": s, "method": m, "level": lvl[m],
                                  "record_bytes": nbytes[m] + overhead, "packets": len(pk),
                                  "delivery": float(ok.mean()), "macro_f1": macro_f1(y0, yhat)})
    out = {"levels": lvl, "bytes": nbytes, "seeds": nseed, "bootstrap_resamples": B,
           "point": point, "ci95": ci, "diff_vs_semantic": diff, "ds2_without_202": no202,
           "record_loss": loss_rows}
    (res / "stats.json").write_text(json.dumps(out, indent=1) + "\n")
    for m in METHODS:
        print(f"{m:9s} {lvl[m]:7s} F1 {point[m]:.3f} CI [{ci[m][0]:.3f}, {ci[m][1]:.3f}]  no202 {no202[m]['mean']:.3f}")
    for m, v in diff.items():
        print(f"semantic - {m:8s} {v['point']:+.3f} CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    import pandas as pd

    print(pd.DataFrame(loss_rows).groupby(["gamma0_db", "method"])[["delivery", "macro_f1"]].mean().to_string())


if __name__ == "__main__":
    main()
