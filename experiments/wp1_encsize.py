"""Encoder-size selection by DS1 4-fold patient CV (decision 2026-10-13). DS2 is never loaded.

Candidates keep the latent length (stride product 32 -> 8 time steps), so every rate level keeps
its payload size; only widths/kernels/strides of the on-node encoder change.

Pre-registered rule (docs/DECISIONS.md, 2026-10-13, written before the runs):
  among the candidates, choose the one with the fewest encoder MACs whose mean out-of-fold
  macro-F1 at the smallest rate level (k1_b4, 8 bytes) over 5 seeds is at least
  (reference mean - 0.010), where the reference is the current encoder trained in the same sweep.

    python experiments/wp1_encsize.py --run --jobs 14     # train all CV jobs (CPU)
    python experiments/wp1_encsize.py                     # summarise + apply the rule
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 2, 3, 4]
MARGIN = 0.010
CANDIDATES = {  # name -> (widths, kernels, strides)
    "ref": ([16, 24, 32, 32, 32], [7, 5, 5, 3, 3], [2, 2, 2, 2, 2]),
    "w12": ([12, 16, 24, 24, 24], [7, 5, 5, 3, 3], [2, 2, 2, 2, 2]),
    "w8": ([8, 12, 16, 16, 16], [7, 5, 5, 3, 3], [2, 2, 2, 2, 2]),
    "w6": ([6, 8, 12, 12, 12], [7, 5, 5, 3, 3], [2, 2, 2, 2, 2]),
    "w4": ([4, 8, 8, 8, 8], [7, 5, 5, 3, 3], [2, 2, 2, 2, 2]),
    "w8s4": ([8, 16, 16, 16], [7, 5, 3, 3], [4, 2, 2, 2]),
}


def macs(widths, kernels, strides, cin=2, latent=16, window=252):
    """Same count as src.l1_semantic.model.encoder_macs (conv layers + mu head, BN folded)."""
    m, length = 0, window
    for w, k, s in zip(widths, kernels, strides):
        length = (length + 2 * (k // 2) - k) // s + 1
        m += cin * w * k * length
        cin = w
    return m + cin * latent * length, length


def job(name_seed):
    name, seed = name_seed
    w, k, s = CANDIDATES[name]
    enc = "{widths:[%s],kernels:[%s],strides:[%s]}" % tuple(",".join(map(str, v)) for v in (w, k, s))
    args = [sys.executable, "experiments/wp1_train.py", "--config-name", "wp1_mitbih_configB", "method=semantic",
            f"seed={seed}", "dev.enabled=true", "dev.cv_folds=4", "dev.cv_epochs=15", f"dev.variant=enc_{name}",
            f"run_name=semantic_enc_{name}_mitbih", f"model.encoder={enc}"]
    log = REPO / "results" / "wp1_dev" / "logs" / f"enc_{name}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        r = subprocess.run(args, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    print(f"enc_{name} seed {seed}: rc={r.returncode}", flush=True)
    return r.returncode


def summarise():
    import yaml

    const = yaml.safe_load((REPO / "configs" / "constants.yaml").read_text())
    cpm = const["nn_inference"]["cycles_per_mac"]["value"]
    f = const["mcu"]["clock_mhz"]["value"] * 1e6
    i_a = const["mcu"]["run_current_ma"]["value"] * 1e-3
    v = const["mcu"]["supply_v"]["value"]
    rows = []
    for name, (w, k, s) in CANDIDATES.items():
        m, length = macs(w, k, s)
        files = sorted((REPO / "results" / "wp1_dev" / f"enc_{name}").glob("semantic_seed*.json"))
        per_level = {}
        for fp in files:
            d = json.loads(fp.read_text())
            for g in d["val_grid"]:
                if g["snr_db"] is None and g["loss_p"] is None:
                    per_level.setdefault(f"k{g['channels']}_b{g['bits']}", []).append(g["metrics"]["macro_f1"])
        lv = {k_: {"mean": st.mean(x), "std": st.stdev(x) if len(x) > 1 else 0.0, "n": len(x)}
              for k_, x in per_level.items()}
        rows.append({"name": name, "widths": w, "kernels": k, "strides": s, "latent_len": length, "macs": m,
                     "e_enc_uj": m * cpm / f * i_a * v * 1e6, "seeds": len(files), "cv_macro_f1": lv})
    ref = next(r for r in rows if r["name"] == "ref")
    thr = ref["cv_macro_f1"]["k1_b4"]["mean"] - MARGIN
    ok = [r for r in rows if r["seeds"] == len(SEEDS) and r["cv_macro_f1"]["k1_b4"]["mean"] >= thr]
    chosen = min(ok, key=lambda r: r["macs"]) if ok else None
    out = {"rule": f"fewest MACs with mean DS1-CV macro-F1 (k1_b4) >= ref - {MARGIN}", "threshold": thr,
           "chosen": chosen["name"] if chosen else None, "candidates": rows}
    dd = REPO / "results" / "wp1_dev" / "encsize"
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        c = r["cv_macro_f1"].get("k1_b4", {"mean": float("nan"), "std": float("nan")})
        print(f"{r['name']:6s} MACs {r['macs']:7d}  E_enc {r['e_enc_uj']:6.1f} uJ  seeds {r['seeds']}  "
              f"CV F1(8 B) {c['mean']:.4f} +/- {c['std']:.4f}")
    print(f"threshold {thr:.4f}; chosen: {out['chosen']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.run:
        js = [(n, s) for n in CANDIDATES for s in SEEDS]
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            rcs = list(ex.map(job, js))
        if any(rcs):
            print(f"{sum(1 for r in rcs if r)} job(s) failed")
    summarise()


if __name__ == "__main__":
    main()
