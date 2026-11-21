"""Presentation figures for L1 from the release branch (decision 2026-10-13); no training, no new data.

  figures/wp1_confusion.*   row-normalized DS2 confusion matrices (mean over seeds) at the tested levels
  figures/wp1_curves.*      DS1 4-fold CV training curves of the proposed encoder: training loss terms and
                            held-out macro-F1 per epoch (mean +/- std over seeds; folds averaged per seed)
  figures/wp1_summary.*     modeled node energy per beat vs DS2 macro-F1 with record-bootstrap 95% CIs
  results/wp1/figures_data.json  every plotted value (traceable to the release files)

    python experiments/paper_figures_wp1.py --ref origin/r/release-a2a4d9a
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
LEVEL = {"semantic": (1, 4), "rr_only": (0, 8), "raw": None, "jscc": None, "dwt": None, "label": None}
NAME = {"semantic": "Proposed", "rr_only": "RR-only", "raw": "B-raw", "jscc": "B-JSCC", "dwt": "B-DWT", "label": "B-label"}


def git_show(ref, path):
    return subprocess.check_output(["git", "-C", str(REPO), "show", f"{ref}:{path}"]).decode()


def main():
    import matplotlib.pyplot as plt

    from src.utils import plotting

    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/r/release-a2a4d9a")
    a = ap.parse_args()
    stats = json.loads(git_show(a.ref, "results/wp1/stats.json"))
    energy = json.loads(git_show(a.ref, "results/wp1/energy.json"))
    lv = stats["levels"]  # method -> level used in the tests
    data = {"ref": a.ref, "levels": lv}

    # ---- confusion matrices (row-normalized, mean over seeds)
    cms = {}
    for m in ["semantic", "rr_only", "raw", "jscc", "dwt", "label"]:
        k, b = (int(x) for x in lv[m][1:].split("_b"))
        mats = []
        for s in range(5):
            d = json.loads(git_show(a.ref, f"results/wp1/{m}_mitbih/seed{s}.json"))
            g = [r for r in d["test_grid"] if r["channels"] == k and r["bits"] == b
                 and r["snr_db"] is None and r["loss_p"] is None][0]["metrics"]
            cm = np.asarray(g["confusion_matrix"], float)
            mats.append(cm / cm.sum(axis=1, keepdims=True))
            names = g["class_names"]
        cms[m] = np.mean(mats, axis=0)
    data["confusion_row_normalized_mean"] = {m: v.tolist() for m, v in cms.items()}
    data["class_names"] = names

    # ---- CV curves of the proposed encoder (held-out folds)
    curves = {"ce": [], "kl": [], "f1_k1_b4": [], "f1_k16_b8": []}
    for s in range(5):
        recs = [json.loads(ln) for ln in git_show(a.ref, f"results/wp1/logs/train_semantic_cv_seed{s}.log").splitlines()
                if ln.startswith('{"epoch"')]
        ep = max(r["epoch"] for r in recs) + 1
        folds = [recs[i * ep:(i + 1) * ep] for i in range(len(recs) // ep)]
        for key, fn in (("ce", lambda r: r["ce"]), ("kl", lambda r: r["kl"]),
                        ("f1_k1_b4", lambda r: r["val_macro_f1"]["k1_b4"]),
                        ("f1_k16_b8", lambda r: r["val_macro_f1"]["k16_b8"])):
            curves[key].append(np.mean([[fn(r) for r in f] for f in folds], axis=0))
    data["cv_curves_mean"] = {k: np.mean(v, axis=0).tolist() for k, v in curves.items()}
    data["cv_curves_std"] = {k: np.std(v, axis=0, ddof=1).tolist() for k, v in curves.items()}
    data["cv_folds"] = len(folds)

    # ---- energy vs F1 summary
    erow = {(r["method"], r["level"]): r for r in energy["rows"]}
    summ = {}
    for m in LEVEL:
        e = erow[(m, lv[m])]
        summ[m] = {"bytes": stats["bytes"][m], "e_total_uj": e["e_total_j_cycles_per_mac"] * 1e6,
                   "e_total_pess_uj": e["e_total_j_cycles_per_mac_pessimistic"] * 1e6,
                   "f1": stats["point"][m], "ci95": stats["ci95"][m]}
    data["summary"] = summ
    (REPO / "results" / "wp1").mkdir(parents=True, exist_ok=True)
    (REPO / "results" / "wp1" / "figures_data.json").write_text(json.dumps(data, indent=1) + "\n")

    plotting.apply_style()
    # confusion heatmaps
    fig, axs = plt.subplots(2, 3, figsize=(plotting.IEEE_PAGE_IN * 0.8, 3.6))
    ax = axs.ravel()
    for i, m in enumerate(cms):
        im = ax[i].imshow(cms[m], cmap="Greys", vmin=0, vmax=1)
        for r in range(4):
            for c in range(4):
                v = cms[m][r, c]
                ax[i].text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6,
                           color="white" if v > 0.55 else "black")
        ax[i].set_xticks(range(4), names if i >= 3 else [""] * 4)
        ax[i].set_yticks(range(4), names if i % 3 == 0 else [""] * 4)
        ax[i].set_title(f"{NAME[m]} ({stats['bytes'][m]} B)", fontsize=7)
        ax[i].set_xlabel("Predicted" if i >= 3 else "")
        ax[i].grid(False)
    ax[0].set_ylabel("True")
    ax[3].set_ylabel("True")
    fig.subplots_adjust(hspace=0.25)
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="Row share")
    plotting.save(fig, REPO / "figures" / "wp1_confusion")

    # CV curves
    x = np.arange(1, len(data["cv_curves_mean"]["ce"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(plotting.IEEE_COL_IN * 2, 1.8))
    mu, sd = data["cv_curves_mean"], data["cv_curves_std"]
    ax[0].plot(x, mu["ce"], label="Weighted cross-entropy", color=plotting.PALETTE[0])
    ax[0].fill_between(x, np.subtract(mu["ce"], sd["ce"]), np.add(mu["ce"], sd["ce"]), color=plotting.PALETTE[0], alpha=0.2)
    ax0b = ax[0].twinx()
    ax0b.plot(x, mu["kl"], "--", label="KL term (nats)", color=plotting.PALETTE[2])
    ax0b.grid(False)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Training cross-entropy")
    ax0b.set_ylabel("KL term")
    h = ax[0].get_legend_handles_labels()
    h2 = ax0b.get_legend_handles_labels()
    ax[0].legend(h[0] + h2[0], h[1] + h2[1], fontsize=6, loc="center right")
    for j, (key, lab) in enumerate((("f1_k1_b4", f"Smallest level ({stats['bytes']['semantic']} B)"), ("f1_k16_b8", "Largest level"))):
        ax[1].plot(x, mu[key], ["-", "--"][j], label=lab, color=plotting.PALETTE[j])
        ax[1].fill_between(x, np.subtract(mu[key], sd[key]), np.add(mu[key], sd[key]), color=plotting.PALETTE[j], alpha=0.2)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Held-out macro-F1 (DS1 CV)")
    ax[1].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    plotting.save(fig, REPO / "figures" / "wp1_curves")

    # summary energy vs F1
    fig, ax = plt.subplots(figsize=(plotting.IEEE_COL_IN, 2.6))
    for i, m in enumerate(LEVEL):
        s = summ[m]
        col = ["#2F4B7C", "#4D7F5E", "#8C6D31", "#A05195", "#5A5A5A", "#D45087"][i]
        ax.errorbar(s["e_total_uj"], s["f1"], yerr=[[s["f1"] - s["ci95"][0]], [s["ci95"][1] - s["f1"]]],
                    fmt=plotting.MARKERS[i], color=col, capsize=2, ms=4,
                    label=f"{NAME[m]} ({s['bytes']} B)")
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    ax.xaxis.set_major_locator(FixedLocator([15, 20, 30, 50, 100, 150]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r"Modeled node energy per beat ($\mu$J)")
    ax.set_ylabel("DS2 macro-F1 (95% CI)")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    fig.tight_layout()
    plotting.save(fig, REPO / "figures" / "wp1_summary")
    print(json.dumps({m: (round(v["e_total_uj"], 1), round(v["f1"], 3)) for m, v in summ.items()}))


if __name__ == "__main__":
    main()
