"""External evaluation of the frozen L1 models on INCART (pre-registered 2026-10-13, docs/DECISIONS.md).

No training, calibration or level change: the 5-seed weights of the release of record are loaded from its
result branch, every model sees INCART beats prepared with the configuration stored in its own checkpoint, and
the clean-channel metrics are computed at every rate level. Statistics follow the pre-registration:

  primary      seed-averaged macro-F1 (N/S/V/F) on all INCART beats
  NI           proposed (k1_b4, 8 B) vs each baseline at its DS2 test level, margin 0.02;
               record bootstrap (B = 2000, RNG 20261013), one-sided p = (1 + #{d* <= -0.02}) / (B + 1),
               Holm over the 5 baselines, NI where adjusted p < 0.025; secondary: seed-paired one-sided
               t-test of d + 0.02 > 0, Holm
  superiority  two-sided seed-paired t-test (Holm) and record-bootstrap 95% CI, as on DS2
  sensitivity  subject-cluster bootstrap when header comments identify subjects

Outputs: results/wp1_ext/incart/<run>/seed<k>.json, seed<k>_pred.npz; results/wp1_ext/incart_stats.json.

    python experiments/wp1_external.py --ref origin/r/release-a2a4d9a
    python experiments/wp1_external.py --ref local      # the models under results/wp1/ of this checkout
"""
from __future__ import annotations

import argparse
import io
import os
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BASELINES = ["rr_only", "raw", "dwt", "jscc", "label"]
RUNS = ["semantic", "rr_only", "raw", "dwt", "jscc", "label", "semantic_norec", "semantic_noaug",
        "semantic_notpl", "semantic_configA"]
MARGIN = 0.02
B = 2000
RNG_SEED = 20261013
ALPHA_NI = 0.025


def git_bytes(ref: str, path: str) -> bytes:
    """A file of the release: from its result branch, or, with ref "local", from this checkout, where the stages
    wp1 and wp1stats of the same run wrote it (a copy of the code without result branches has only those)."""
    if ref == "local":
        return (REPO / path).read_bytes()
    return subprocess.check_output(["git", "-C", str(REPO), "show", f"{ref}:{path}"])


def macro_f1(y, p, n=4):
    f = []
    for c in range(n):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def holm(p: dict) -> dict:
    keys = sorted(p, key=p.get)
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def fetch_incart(raw: Path, manifests: Path) -> Path:
    from src.data import physionet

    try:
        return physionet.fetch("incartdb", raw, manifests, source="s3")
    except FileNotFoundError:  # mirror without SHA256SUMS.txt: record our own hashes of what was read
        root = raw / "incartdb-1.0.0"
        man = {"database": "incartdb", "version": "1.0.0", "source": "s3",
               "url": "s3://physionet-open/incartdb/1.0.0/", "note": "no SHA256SUMS.txt on the mirror",
               "files": {p.name: physionet.sha256_file(p) for p in sorted(root.glob("I*.*"))}}
        manifests.mkdir(parents=True, exist_ok=True)
        (manifests / "incartdb.json").write_text(json.dumps(man, indent=2) + "\n")
        return root


def evaluate(ref: str, root: Path, out: Path):
    import torch
    from omegaconf import OmegaConf

    from src.data import incart
    from src.l1_semantic import data as l1data
    from src.l1_semantic.metrics import assert_consistent, multiclass_metrics
    from src.l1_semantic.train import predict

    sys.path.insert(0, str(REPO / "experiments"))
    from wp1_train import build_method

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    cache: dict = {}
    info = {}
    for run in RUNS:
        for seed in range(5):
            ck = torch.load(io.BytesIO(git_bytes(ref, f"results/wp1/{run}_mitbih/seed{seed}_model.pt")),
                            weights_only=False, map_location="cpu")
            cfg = OmegaConf.create(ck["config"])
            w = cfg.data.window
            key = (w.pre, w.post, float(cfg.features.template_alpha), int(cfg.features.rr_long_window))
            if key not in cache:
                beats, counts, comments = incart.load_all(root, w.pre, w.post)
                cache[key] = l1data.arrays_from_beats(beats, list(cfg.classes), cfg.rr.step_s, (),
                                                      template_alpha=key[2], rr_long_window=key[3])
                info = {"symbol_counts": counts, "header_comments": comments,
                        "n_beats_by_class": dict(zip(list(cfg.classes),
                                                     np.bincount(cache[key].y, minlength=4).tolist()))}
            te = cache[key]
            stub = l1data.SplitArrays(te.x[:4], te.rr[:4], np.arange(len(cfg.classes)), te.record[:4])
            model = build_method(cfg, stub, w.pre + w.post)
            model.load_state_dict(ck["model"])
            model.eval()
            rr_bytes = cfg.rr.bytes + (1 if cfg.features.rr_long_window > 0 else 0)
            grid, preds = [], {}
            with torch.no_grad():  # clean channel: one prediction per level gives the metrics and the labels
                for lv in model.rate_levels:
                    probs = predict(model, te, lv)
                    mt = multiclass_metrics(te.y, probs, list(cfg.classes))
                    rep = assert_consistent(mt, te.y, probs)
                    grid.append({"channels": lv.channels, "bits": lv.bits, "snr_db": None, "loss_p": None,
                                 "metrics": mt, "consistency_warnings": rep["warnings"]})
                    preds[f"k{lv.channels}_b{lv.bits}"] = probs.argmax(1).astype(np.int8)
            d = out / f"{run}_incart"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"seed{seed}.json").write_text(json.dumps(
                {"ref": ref, "commit_of_model": ck.get("commit"), "run": run, "seed": seed,
                 "n": int(len(te.y)), "test_grid": grid}, indent=1) + "\n")
            np.savez_compressed(d / f"seed{seed}_pred.npz", y=te.y.astype(np.int8),
                                record=te.record.astype(np.int16), **preds)
            m = [g for g in grid if g["snr_db"] is None and g["loss_p"] is None]
            print(run, seed, " ".join(f"k{g['channels']}_b{g['bits']}={g['metrics']['macro_f1']:.4f}" for g in m),
                  flush=True)
    (out / "data_info.json").write_text(json.dumps(info, indent=1) + "\n")


def statistics(ref: str, out: Path):
    from scipy import stats as sst

    from src.data import incart

    lvl = json.loads(git_bytes(ref, "results/wp1/stats.json"))["levels"]
    nbytes = json.loads(git_bytes(ref, "results/wp1/stats.json"))["bytes"]
    methods = ["semantic"] + BASELINES
    preds = {m: [dict(np.load(out / f"{m}_incart" / f"seed{s}_pred.npz")) for s in range(5)] for m in methods}
    y0, rec0 = preds["semantic"][0]["y"], preds["semantic"][0]["record"]
    for m in methods:
        for d in preds[m]:
            assert np.array_equal(d["y"], y0) and np.array_equal(d["record"], rec0)
    seed_f1 = {m: np.array([macro_f1(y0, d[lvl[m]]) for d in preds[m]]) for m in methods}
    point = {m: float(seed_f1[m].mean()) for m in methods}

    def bootstrap(units: np.ndarray, rng):
        groups = {u: np.flatnonzero(units == u) for u in np.unique(units)}
        keys = np.array(sorted(groups))
        bt = {m: np.empty(B) for m in methods}
        for b in range(B):
            idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
            for m in methods:
                bt[m][b] = np.mean([macro_f1(y0[idx], d[lvl[m]][idx]) for d in preds[m]])
        return bt, len(keys)

    def tests(bt):
        res, p_ni, p_ni_t, p_sup = {}, {}, {}, {}
        for m in BASELINES:
            dstar = bt["semantic"] - bt[m]
            dseed = seed_f1["semantic"] - seed_f1[m]
            p_ni[m] = float((1 + np.sum(dstar <= -MARGIN)) / (B + 1))
            p_ni_t[m] = float(sst.ttest_1samp(dseed + MARGIN, 0.0, alternative="greater").pvalue)
            p_sup[m] = float(sst.ttest_rel(seed_f1["semantic"], seed_f1[m]).pvalue)
            res[m] = {"level": lvl[m], "bytes": nbytes[m], "diff": point["semantic"] - point[m],
                      "ci95": [float(np.percentile(dstar, 2.5)), float(np.percentile(dstar, 97.5))],
                      "lower_one_sided_97_5": float(np.percentile(dstar, 2.5)),
                      "frac_positive": float(np.mean(dstar > 0)),
                      "p_ni_bootstrap": p_ni[m], "p_ni_t": p_ni_t[m], "p_sup_t": p_sup[m]}
        for name, p in (("p_ni_bootstrap_holm", p_ni), ("p_ni_t_holm", p_ni_t), ("p_sup_t_holm", p_sup)):
            for m, v in holm(p).items():
                res[m][name] = v
        for m in BASELINES:
            res[m]["noninferior"] = bool(res[m]["p_ni_bootstrap_holm"] < ALPHA_NI)
            res[m]["noninferior_t"] = bool(res[m]["p_ni_t_holm"] < ALPHA_NI)
        return res

    bt_rec, n_rec = bootstrap(rec0, np.random.default_rng(RNG_SEED))
    result = {"ref": ref, "margin": MARGIN, "resamples": B, "rng_seed": RNG_SEED, "alpha_ni_one_sided": ALPHA_NI,
              "levels": lvl, "bytes": nbytes, "n_beats": int(len(y0)), "n_records": int(n_rec),
              "class_counts": np.bincount(y0, minlength=4).tolist(),
              "point": point, "seed_f1": {m: v.tolist() for m, v in seed_f1.items()},
              "std": {m: float(v.std(ddof=1)) for m, v in seed_f1.items()},
              "ci95": {m: [float(np.percentile(bt_rec[m], 2.5)), float(np.percentile(bt_rec[m], 97.5))]
                       for m in methods},
              "record_bootstrap": tests(bt_rec)}
    info = json.loads((out / "data_info.json").read_text())
    groups = incart.subject_groups({r: c for r, c in info["header_comments"].items()})
    n_subj = len(set(groups.values()))
    result["n_subjects_from_headers"] = n_subj
    if n_subj < len(groups):
        subj = np.array([groups[f"I{r:02d}"] for r in rec0])
        bt_s, _ = bootstrap(subj, np.random.default_rng(RNG_SEED))
        result["subject_bootstrap"] = tests(bt_s)
    # per-class sensitivity of the proposed encoder and all-level means
    allruns = {}
    for d in sorted(out.glob("*_incart")):
        rows = {}
        for f in sorted(d.glob("seed[0-9].json")):
            for g in json.loads(f.read_text())["test_grid"]:
                if g["snr_db"] is None and g["loss_p"] is None:
                    rows.setdefault(f"k{g['channels']}_b{g['bits']}", []).append(g["metrics"])
        allruns[d.name[:-7]] = {
            k: {"macro_f1_mean": float(np.mean([m["macro_f1"] for m in v])),
                "macro_f1_std": float(np.std([m["macro_f1"] for m in v], ddof=1)),
                "accuracy_mean": float(np.mean([m["accuracy"] for m in v])),
                "auroc_mean": float(np.mean([m["macro_auroc"] for m in v])),
                "se": {c: float(np.mean([m["per_class"][c]["sensitivity"] for m in v])) for c in "NSVF"},
                "ppv": {c: float(np.mean([m["per_class"][c]["ppv"] for m in v])) for c in "NSVF"
                        if all(m["per_class"][c].get("ppv") is not None for m in v)}}
            for k, v in rows.items()}
    result["all_levels"] = allruns
    (out / "incart_stats.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"point": point, "ni": {m: (round(r["diff"], 4), round(r["p_ni_bootstrap_holm"], 4),
                                                  r["noninferior"]) for m, r in result["record_bootstrap"].items()}},
                     indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/r/release-a2a4d9a")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args()
    import os

    raw = Path(a.raw or os.environ.get("SEMPQ_DATA_RAW", REPO / "data" / "raw"))
    out = REPO / "results" / "wp1_ext" / "incart"
    if not a.stats_only:
        root = fetch_incart(raw, REPO / "data" / "manifests")
        evaluate(a.ref, root, out)
    statistics(a.ref, out)


if __name__ == "__main__":
    main()
