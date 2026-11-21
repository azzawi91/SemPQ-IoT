"""PTB-XL (v1.0.3): diagnostic superclasses, official folds, single lead II at 100 Hz.

Labels follow the PTB-XL benchmark convention (Strodthoff et al., 2021; DOI to be verified
in WP5): every SCP code of a record whose ``diagnostic`` flag is 1 in scp_statements.csv
maps to its ``diagnostic_class`` (NORM, MI, STTC, CD, HYP); the task is multi-label and
records with no diagnostic superclass are excluded. Folds: 1-8 train, 9 validation,
10 test (``strat_fold``); PTB-XL assigns all records of a patient to one fold, which
``build_folds`` asserts.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

FS = 100
LEAD = "II"
SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
FOLDS = {"train": tuple(range(1, 9)), "val": (9,), "test": (10,)}


def load_metadata(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    return db, scp


def superclass_labels(scp_codes: dict, scp: pd.DataFrame) -> list[str]:
    diag = scp[scp["diagnostic"] == 1]
    out = {diag.loc[c, "diagnostic_class"] for c in scp_codes if c in diag.index}
    return sorted(out, key=SUPERCLASSES.index)


def build_folds(db: pd.DataFrame, scp: pd.DataFrame) -> pd.DataFrame:
    """One row per labelled record: ecg_id, patient_id, strat_fold, split, filename_lr, 5 label columns."""
    labels = db["scp_codes"].apply(lambda d: superclass_labels(d, scp))
    keep = labels.str.len() > 0
    split_of = {f: s for s, fs in FOLDS.items() for f in fs}
    out = pd.DataFrame({
        "ecg_id": db.index[keep],
        "patient_id": db.loc[keep, "patient_id"].astype(int).values,
        "strat_fold": db.loc[keep, "strat_fold"].astype(int).values,
        "filename_lr": db.loc[keep, "filename_lr"].values,
    })
    out["split"] = out["strat_fold"].map(split_of)
    for c in SUPERCLASSES:
        out[c] = labels[keep].apply(lambda l, c=c: int(c in l)).values
    assert_patient_disjoint(out)
    return out.sort_values("ecg_id").reset_index(drop=True)


def assert_patient_disjoint(folds: pd.DataFrame) -> None:
    groups = {s: set(folds.loc[folds["split"] == s, "patient_id"]) for s in FOLDS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = groups[a] & groups[b]
        if inter:
            raise AssertionError(f"PTB-XL patient leakage {a}/{b}: {len(inter)} patients")


def load_lead_ii(root: Path, filenames, dtype=np.float32) -> np.ndarray:
    """Return (n, 1000) lead-II signals in mV, read with wfdb from the records100 files."""
    import wfdb

    out = []
    for fn in filenames:
        rec = wfdb.rdrecord(str(Path(root) / fn))
        if rec.fs != FS:
            raise ValueError(f"{fn}: fs={rec.fs}, expected {FS}")
        out.append(rec.p_signal[:, rec.sig_name.index(LEAD)])
    return np.asarray(out, dtype=dtype)
