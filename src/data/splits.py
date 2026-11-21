"""Committed, hashed patient-level split files (brief rule 2).

Files under data/splits/:
  mitbih_inter_patient.json   DS1 (train + val records) / DS2 (test), de Chazal
  ptbxl_folds.csv             one row per labelled PTB-XL record with its split
  SHA256SUMS                  sha256 of each split file; checked by tests and loaders
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from . import mitbih

SPLIT_DIR = Path(__file__).resolve().parents[2] / "data" / "splits"
MITBIH_FILE = "mitbih_inter_patient.json"
PTBXL_FILE = "ptbxl_folds.csv"
SUMS_FILE = "SHA256SUMS"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mitbih_split_dict(val_records: tuple[int, ...], val_rule: dict) -> dict:
    ds1 = list(mitbih.DS1)
    val = sorted(val_records)
    if not set(val) <= set(ds1):
        raise ValueError("validation records must come from DS1")
    return {
        "database": "mitdb 1.0.0",
        "scheme": "inter-patient, de Chazal et al. 2004",
        "train": [r for r in ds1 if r not in val],
        "val": val,
        "test": list(mitbih.DS2),
        "excluded_paced": list(mitbih.PACED),
        "same_subject_pairs": [list(p) for p in mitbih.SAME_SUBJECT_PAIRS],
        "val_rule": val_rule,
    }


def write_splits(mitbih_split: dict, ptbxl_folds: pd.DataFrame, out_dir: Path = SPLIT_DIR) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MITBIH_FILE).write_text(json.dumps(mitbih_split, indent=2) + "\n")
    ptbxl_folds.to_csv(out_dir / PTBXL_FILE, index=False, lineterminator="\n")
    sums = {f: _sha(out_dir / f) for f in (MITBIH_FILE, PTBXL_FILE)}
    (out_dir / SUMS_FILE).write_text("".join(f"{h}  {f}\n" for f, h in sorted(sums.items())))
    return sums


def verify_splits(out_dir: Path = SPLIT_DIR) -> dict:
    """Raise if any committed split file does not match its recorded SHA256."""
    out_dir = Path(out_dir)
    recorded = {}
    for line in (out_dir / SUMS_FILE).read_text().splitlines():
        h, f = line.split(maxsplit=1)
        recorded[f.strip()] = h
    for f, h in recorded.items():
        actual = _sha(out_dir / f)
        if actual != h:
            raise RuntimeError(f"split file {f} changed: sha256 {actual} != recorded {h}")
    return recorded


def load_mitbih_split(out_dir: Path = SPLIT_DIR) -> dict:
    verify_splits(out_dir)
    return json.loads((Path(out_dir) / MITBIH_FILE).read_text())


def load_ptbxl_folds(out_dir: Path = SPLIT_DIR) -> pd.DataFrame:
    verify_splits(out_dir)
    return pd.read_csv(Path(out_dir) / PTBXL_FILE)
