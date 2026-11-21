"""Aggregate WP1 runs: mean +/- std over seeds, Pareto check, paired tests, tables, figures.

    python experiments/wp1_aggregate.py --results <dir containing results/wp1> --tag <label>

Reads <dir>/results/wp1/<method>_mitbih/seed*.json. Writes into <dir>/results/wp1/aggregate/:
summary.json, rate_f1.{pdf,png}, snr.{pdf,png}, loss.{pdf,png} and tables/wp1_*.tex.
Paired tests use per-seed test macro-F1 (seeds matched by index): paired t-test and exact
Wilcoxon signed-rank, Holm-corrected across the baselines compared.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.utils import plotting  # noqa: E402

LABELS = {"semantic": "Proposed (semantic)", "raw": "B-raw", "dwt": "B-DWT", "jscc": "B-JSCC",
          "label": "B-label", "rr_only": "RR only (ablation)",
          "semantic_norec": "Proposed w/o reconstruction", "semantic_noaug": "Proposed w/o augmentation",
          "semantic_notpl": "Proposed w/o template", "semantic_configA": "Proposed, configuration A"}
BASELINES = {"raw", "dwt", "jscc", "label", "rr_only"}


def load(root: Path, extra=()) -> pd.DataFrame:
    rows = []
    files = [(f, None) for f in sorted((root / "results" / "wp1").glob("*_mitbih/seed*.json"))]
    for spec in extra:  # "<dir with results/wp1>:<method>:<alias>"
        d, meth, alias = spec.split(":")
        files += [(f, alias) for f in sorted((Path(d) / "results" / "wp1" / f"{meth}_mitbih").glob("seed*.json"))]
    for f, alias in files:
        d = json.loads(f.read_text())
        meth, seed = alias or d["method"], d["provenance"]["seed"]
        for r in d["test_grid"]:
            key = f"k{r['channels']}_b{r['bits']}"
            m = r["metrics"]
            rows.append({"method": meth, "seed": seed, "level": key, "bytes": d["payload_bytes"][key],
                         "snr_db": r["snr_db"], "loss_p": r["loss_p"],
                         "macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "macro_auroc": m["macro_auroc"],
                         "se_S": m["per_class"]["S"]["sensitivity"], "se_V": m["per_class"]["V"]["sensitivity"],
                         "ppv_S": m["per_class"]["S"]["ppv"], "ppv_V": m["per_class"]["V"]["ppv"],
                         "warnings": len(r["consistency_warnings"]), "cpu": d["provenance"].get("cpu_model"),
                         "commit": d["provenance"]["commit"][:7], "macs": d["device_cost_model"]["macs_per_beat"]})
    return pd.DataFrame(rows)


def holm(p: dict) -> dict:
    keys = sorted(p, key=p.get)
    m, out, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * p[k]))
        out[k] = running
    return out


def pareto_front(points):
    """Indices of points not dominated (fewer bytes and higher F1 is better)."""
    idx = []
    for i, (b, f) in enumerate(points):
        if not any((b2 <= b and f2 >= f) and (b2 < b or f2 > f) for b2, f2 in points):
            idx.append(i)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="<dir>:<method>:<alias> runs to include")
    args = ap.parse_args()
    df = load(args.results, args.extra)
    out = args.results / "results" / "wp1" / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    clean = df[df.snr_db.isna() & df.loss_p.isna()]
    agg = (clean.groupby(["method", "level", "bytes"])
           .agg(n=("seed", "nunique"), f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                auc_mean=("macro_auroc", "mean"), auc_std=("macro_auroc", "std"),
                seS_mean=("se_S", "mean"), seV_mean=("se_V", "mean"), macs=("macs", "first"))
           .reset_index().sort_values(["method", "bytes"]))

    # Pareto front over (bytes, mean macro-F1) across all methods and levels
    pts = list(zip(agg.bytes, agg.f1_mean))
    front = set(pareto_front(pts))
    agg["on_front"] = [i in front for i in range(len(agg))]

    # paired tests: proposed (each level) vs the best level of each baseline (by mean test F1)
    tests = {}
    sem = clean[clean.method == "semantic"]
    for lvl in sorted(sem.level.unique()):
        a = sem[sem.level == lvl].sort_values("seed")
        pvals_t, rows = {}, {}
        for b in sorted(set(clean.method) & BASELINES):
            bb = clean[clean.method == b]
            best_lvl = bb.groupby("level").macro_f1.mean().idxmax()
            bsel = bb[bb.level == best_lvl].sort_values("seed")
            common = sorted(set(a.seed) & set(bsel.seed))
            if len(common) < 2:
                continue
            x = a.set_index("seed").loc[common, "macro_f1"].to_numpy()
            y = bsel.set_index("seed").loc[common, "macro_f1"].to_numpy()
            t = stats.ttest_rel(x, y)
            w = stats.wilcoxon(x, y, method="exact") if np.any(x != y) else None
            pvals_t[b] = float(t.pvalue)
            rows[b] = {"baseline_level": best_lvl, "n": len(common), "mean_diff": float((x - y).mean()),
                       "t": float(t.statistic), "p_t": float(t.pvalue),
                       "p_wilcoxon": None if w is None else float(w.pvalue),
                       "cohens_dz": float((x - y).mean() / ((x - y).std(ddof=1) + 1e-12))}
        for b, ph in holm(pvals_t).items():
            rows[b]["p_t_holm"] = ph
        tests[lvl] = rows
    # ablations: proposed (same level) vs each ablation, paired by seed, Holm across ablations
    abl_tests = {}
    for lvl in sorted(sem.level.unique()):
        a = sem[sem.level == lvl].set_index("seed").macro_f1
        pv, rows = {}, {}
        for ab in sorted(m for m in clean.method.unique() if m.startswith("semantic_")):
            bsel = clean[(clean.method == ab) & (clean.level == lvl)].set_index("seed").macro_f1
            common = sorted(set(a.index) & set(bsel.index))
            if len(common) < 2:
                continue
            x, y = a.loc[common].to_numpy(), bsel.loc[common].to_numpy()
            t = stats.ttest_rel(x, y)
            pv[ab] = float(t.pvalue)
            rows[ab] = {"n": len(common), "mean_diff": float((x - y).mean()), "p_t": float(t.pvalue)}
        for ab, ph in holm(pv).items():
            rows[ab]["p_t_holm"] = ph
        abl_tests[lvl] = rows

    # robustness curves at the largest level of each method
    def sweep(col):
        sub = df[df[col].notna()]
        res = {}
        for meth, g in sub.groupby("method"):
            top = g[g.bytes == g.bytes.max()]
            res[meth] = top.groupby(col).macro_f1.agg(["mean", "std"]).reset_index().to_dict("list")
        return res

    summary = {"tag": args.tag, "commits": sorted(df.commit.unique()), "cpus": sorted(df.cpu.dropna().unique()),
               "consistency_warnings_total": int(df.warnings.sum()),
               "clean": agg.to_dict("records"), "paired_tests": tests, "ablation_tests": abl_tests,
               "snr_sweep": sweep("snr_db"), "loss_sweep": sweep("loss_p")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")

    # ---------------- figures: (a) proposed vs baselines, (b) proposed vs its ablations
    plotting.apply_style()
    style = {"semantic": (0, "o"), "raw": (1, "D"), "dwt": (2, "s"), "jscc": (3, "^"), "label": (4, "v"),
             "rr_only": (6, "P"), "semantic_norec": (1, "s"), "semantic_noaug": (2, "^"),
             "semantic_notpl": (3, "v"), "semantic_configA": (6, "X")}
    groups = {"rate_f1": ["semantic", "raw", "dwt", "jscc", "label", "rr_only"],
              "rate_f1_ablations": ["semantic", "semantic_norec", "semantic_noaug", "semantic_notpl",
                                    "semantic_configA"]}
    for name, methods in groups.items():
        fig, ax = plt.subplots(figsize=(plotting.IEEE_COL_IN, 2.6))
        for meth in methods:
            g = agg[agg.method == meth].sort_values("bytes")
            if g.empty:
                continue
            ci, mk = style.get(meth, (5, "o"))
            ax.errorbar(g.bytes, g.f1_mean, yerr=g.f1_std, marker=mk, capsize=2, lw=0.9,
                        color=plotting.PALETTE[ci], label=LABELS.get(meth, meth))
        ax.set_xscale("log")
        ax.set_xlabel("Payload per beat (bytes)")
        ax.set_ylabel("Macro-F1 (DS2, N/S/V/F)")
        ax.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
        plotting.save(fig, out / name)
    for col, xl, name in (("snr_db", "Latent SNR (dB)", "snr"), ("loss_p", "Fragment loss probability", "loss")):
        fig, ax = plt.subplots(figsize=(plotting.IEEE_COL_IN, 2.5))
        for meth in groups["rate_f1"]:
            sw = summary[f"{name}_sweep"].get(meth)
            if sw is None:
                continue
            ci, mk = style.get(meth, (5, "o"))
            ax.errorbar(sw[col], sw["mean"], yerr=sw["std"], marker=mk, capsize=2, lw=0.9,
                        color=plotting.PALETTE[ci], label=LABELS.get(meth, meth))
        ax.set_xlabel(xl)
        ax.set_ylabel("Macro-F1 (largest level)")
        ax.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24))
        plotting.save(fig, out / name)

    # ---------------- table
    tab = agg.copy()
    tab["Method"] = tab.method.map(LABELS)
    tab["Macro-F1"] = [f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(tab.f1_mean, tab.f1_std)]
    tab["Accuracy"] = [f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(tab.acc_mean, tab.acc_std)]
    tab["AUROC"] = [f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(tab.auc_mean, tab.auc_std)]
    tab["Se(S)"] = tab.seS_mean.map("{:.3f}".format)
    tab["Se(V)"] = tab.seV_mean.map("{:.3f}".format)
    tab = tab.rename(columns={"bytes": "Bytes", "level": "Level", "macs": "Device MACs"})
    cols = ["Method", "Level", "Bytes", "Macro-F1", "Accuracy", "AUROC", "Se(S)", "Se(V)", "Device MACs"]
    (out / "wp1_clean.tex").write_text(tab[cols].to_latex(index=False, escape=False))
    print(agg[["method", "level", "bytes", "n", "f1_mean", "f1_std", "acc_mean", "auc_mean",
               "seS_mean", "seV_mean", "on_front"]].to_string(index=False))
    print(json.dumps(tests, indent=1)[:4000])


if __name__ == "__main__":
    main()
