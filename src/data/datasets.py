"""Cached arrays and thin torch Dataset wrappers for MIT-BIH beats and PTB-XL lead II.

Split membership always comes from the committed, hash-checked split files; caches under
data/cache/ (gitignored) are keyed by split and window so a config change cannot silently
reuse stale arrays.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import mitbih, ptbxl
from .splits import load_mitbih_split, load_ptbxl_folds


def _atomic_save(path: Path, write) -> None:
    """Write a cache file under a temporary name and rename it into place, so parallel jobs
    sharing one cache directory never read a partly written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        write(fh)
    os.replace(tmp, path)


def mitbih_split_beats(root: Path, split: str, pre: int, post: int, include_q: bool,
                       cache_dir: Path | None = None) -> mitbih.BeatSet:
    records = load_mitbih_split()[split]
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"mitbih_{split}_pre{pre}_post{post}_q{int(include_q)}.npz"
        if cache.exists():
            z = np.load(cache)
            return mitbih.BeatSet(z["x"], z["y"], z["record"], z["r_sample"], z["rr"])
    beats = mitbih.BeatSet.concat(
        [mitbih.extract_record(root, r, pre, post, include_q) for r in records])
    if cache is not None:
        _atomic_save(cache, lambda fh: np.savez(fh, x=beats.x, y=beats.y, record=beats.record,
                                                r_sample=beats.r_sample, rr=beats.rr))
    return beats


def ptbxl_split_arrays(root: Path, split: str, cache_dir: Path | None = None):
    """Return (x: (n,1000) float32 lead II, y: (n,5) multi-hot, ecg_id, patient_id)."""
    folds = load_ptbxl_folds()
    f = folds[folds["split"] == split]
    y = f[list(ptbxl.SUPERCLASSES)].to_numpy(np.float32)
    cache = Path(cache_dir) / f"ptbxl_{split}_leadII.npy" if cache_dir else None
    if cache is not None and cache.exists():
        x = np.load(cache)
    else:
        x = ptbxl.load_lead_ii(root, f["filename_lr"])
        if cache is not None:
            _atomic_save(cache, lambda fh: np.save(fh, x))
    return x, y, f["ecg_id"].to_numpy(), f["patient_id"].to_numpy()


try:
    import torch
    from torch.utils.data import Dataset

    class ArrayDataset(Dataset):
        """(signal, label) pairs; a per-sample transform (e.g. per-beat z-score) is optional."""

        def __init__(self, x: np.ndarray, y: np.ndarray, transform=None):
            self.x = x if transform is None else transform(x)
            self.y = y

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, i):
            return torch.from_numpy(self.x[i][None, :].copy()), torch.as_tensor(self.y[i])

except ImportError:  # pragma: no cover
    ArrayDataset = None
