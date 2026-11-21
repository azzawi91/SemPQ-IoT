"""Summarise validation-only development runs (results/wp1_dev/<variant>/<method>_seed*.json).

    python experiments/wp1_dev_summary.py --results <dir containing results/wp1_dev>

Prints, per variant: mean +/- std over dev seeds of validation macro-F1 per rate level on a
clean channel, the level-average (the selection criterion) and, for reference, the mean
validation macro-F1 at 20 % fragment loss and at 10 dB latent SNR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    a = ap.parse_args()
    rows = []
    for f in sorted((a.results / "results" / "wp1_dev").glob("*/*_seed*.json")):
        d = json.loads(f.read_text())
        for r in d["val_grid"]:
            rows.append({"variant": d["variant"], "method": d["method"], "seed": d["provenance"]["seed"],
                         "level": f"k{r['channels']}_b{r['bits']}", "snr": r["snr_db"], "loss": r["loss_p"],
                         "f1": r["metrics"]["macro_f1"], "best_epoch": d["training"]["best_epoch"]})
    df = pd.DataFrame(rows)
    clean = df[df.snr.isna() & df.loss.isna()]
    per_seed = clean.groupby(["variant", "seed"]).f1.mean().reset_index()
    crit = per_seed.groupby("variant").f1.agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    lv = clean.groupby(["variant", "level"]).f1.mean().unstack()
    loss = df[df.loss == 0.2].groupby("variant").f1.mean().rename("f1@loss0.2")
    snr = df[df.snr == 10.0].groupby("variant").f1.mean().rename("f1@snr10")
    ep = clean.groupby("variant").best_epoch.mean().rename("best_epoch")
    out = crit.join([loss, snr, ep]).join(lv)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
