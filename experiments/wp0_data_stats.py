"""WP0 gate: download + verify MIT-BIH and PTB-XL, build or re-verify split files, data statistics.

    python experiments/wp0_data_stats.py                 # build splits if absent, else check reproduction
    python experiments/wp0_data_stats.py mode=build      # overwrite committed split files

Outputs: data/manifests/*.json, data/splits/*, results/wp0/data_stats.json,
tables/wp0_mitbih_stats.tex, tables/wp0_ptbxl_stats.tex.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.data import mitbih, physionet, ptbxl, splits  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402
from src.utils.seed import pin_threads, set_seed  # noqa: E402


def _abs(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def build_split_objects(cfg, mit_root: Path, ptb_root: Path):
    counts = mitbih.class_counts_per_record(mit_root, mitbih.DS1)
    vr = cfg.mitbih.val_rule
    val = mitbih.choose_validation_records(counts, target=vr.target, sizes=tuple(vr.sizes),
                                           classes=tuple(vr.classes))
    mit_split = splits.mitbih_split_dict(val, OmegaConf.to_container(vr, resolve=True))
    db, scp = ptbxl.load_metadata(ptb_root)
    folds = ptbxl.build_folds(db, scp)
    return mit_split, folds, db


def mitbih_stats(mit_root: Path, split: dict, pre: int, post: int, include_q: bool) -> dict:
    out = {}
    for name in ("train", "val", "test"):
        ann = np.zeros(len(mitbih.AAMI_CLASSES), int)
        ext = np.zeros(len(mitbih.AAMI_CLASSES), int)
        n_nan = 0
        for r in split[name]:
            _, lab = mitbih.read_annotation_labels(mit_root, r)
            ann += np.bincount(lab, minlength=5)
            b = mitbih.extract_record(mit_root, r, pre, post, include_q)
            ext += np.bincount(b.y, minlength=5)
            n_nan += int(np.isnan(b.x).sum())
            assert b.x.shape[1] == pre + post
        out[name] = {
            "records": split[name],
            "n_records": len(split[name]),
            "annotated_beats": dict(zip(mitbih.AAMI_CLASSES, ann.tolist())),
            "extracted_beats": dict(zip(mitbih.AAMI_CLASSES, ext.tolist())),
            "dropped_at_record_edges": int(ann.sum() - ext.sum()),
            "nan_samples": n_nan,
        }
    return out


def ptbxl_stats(ptb_root: Path, folds: pd.DataFrame, db: pd.DataFrame, cache: Path) -> dict:
    out = {"records_total_in_database": int(len(db)),
           "records_without_superclass": int(len(db) - len(folds))}
    for name in ("train", "val", "test"):
        f = folds[folds["split"] == name]
        x = ptbxl.load_lead_ii(ptb_root, f["filename_lr"])
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / f"ptbxl_{name}_leadII.npy", x)
        lab = f[list(ptbxl.SUPERCLASSES)]
        out[name] = {
            "folds": sorted(f["strat_fold"].unique().tolist()),
            "n_records": int(len(f)),
            "n_patients": int(f["patient_id"].nunique()),
            "positives": {c: int(lab[c].sum()) for c in ptbxl.SUPERCLASSES},
            "multi_label_records": int((lab.sum(axis=1) > 1).sum()),
            "signal_shape": list(x.shape[1:]),
            "nan_samples": int(np.isnan(x).sum()),
            "flat_records": int((np.ptp(x, axis=1) == 0).sum()),
        }
    return out


def write_tables(stats: dict, tables: Path) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    m = stats["mitbih"]
    rows = []
    for name, label in (("train", "DS1 train"), ("val", "DS1 validation"), ("test", "DS2 test")):
        e = m[name]["extracted_beats"]
        rows.append([label, m[name]["n_records"], *[e[c] for c in mitbih.AAMI_CLASSES], sum(e.values())])
    df = pd.DataFrame(rows, columns=["Split", "Records", *mitbih.AAMI_CLASSES, "Total"])
    (tables / "wp0_mitbih_stats.tex").write_text(df.to_latex(index=False, escape=True))
    p = stats["ptbxl"]
    rows = []
    for name, label in (("train", "Train (folds 1--8)"), ("val", "Validation (fold 9)"),
                        ("test", "Test (fold 10)")):
        s = p[name]
        rows.append([label, s["n_records"], s["n_patients"], *[s["positives"][c] for c in ptbxl.SUPERCLASSES]])
    df = pd.DataFrame(rows, columns=["Split", "Records", "Patients", *ptbxl.SUPERCLASSES])
    (tables / "wp0_ptbxl_stats.tex").write_text(df.to_latex(index=False, escape=False))


@hydra.main(config_path="../configs", config_name="wp0_data", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    pin_threads(1)
    raw, man = _abs(cfg.paths.raw), _abs(cfg.paths.manifests)
    mit_root = physionet.fetch("mitdb", raw, man, cfg.source)
    ptb_root = physionet.fetch("ptbxl", raw, man, cfg.source)

    mit_split, folds, db = build_split_objects(cfg, mit_root, ptb_root)
    committed = splits.SPLIT_DIR / splits.SUMS_FILE
    if cfg.mode == "build" or not committed.exists():
        sums = splits.write_splits(mit_split, folds)
        split_status = "written"
    else:
        recorded = splits.verify_splits()
        with tempfile.TemporaryDirectory() as tmp:
            rebuilt = splits.write_splits(mit_split, folds, Path(tmp))
        if rebuilt != recorded:
            raise RuntimeError(f"rebuilt split files differ from committed ones: {rebuilt} vs {recorded}")
        sums, split_status = recorded, "reproduced_identically"

    split = splits.load_mitbih_split()
    w = cfg.mitbih.window
    stats = {
        "provenance": provenance(OmegaConf.to_container(cfg, resolve=True), cfg.seed),
        "split_files_sha256": sums,
        "split_status": split_status,
        "manifests": {n: json.loads((man / f"{n}.json").read_text()) for n in ("mitdb", "ptbxl")},
        "mitbih": mitbih_stats(mit_root, split, w.pre, w.post, include_q=True),  # stats always list Q
        "ptbxl": ptbxl_stats(ptb_root, splits.load_ptbxl_folds(), db, _abs(cfg.paths.cache)),
    }
    res = _abs(cfg.paths.results)
    res.mkdir(parents=True, exist_ok=True)
    (res / "data_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    write_tables(stats, _abs(cfg.paths.tables))
    print(json.dumps({k: stats[k] for k in ("split_files_sha256", "split_status")}, indent=2))


if __name__ == "__main__":
    main()
