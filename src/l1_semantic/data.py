"""WP1 data preparation on the committed splits (no dataset-level statistics are fitted).

MIT-BIH beats: per-beat z-score, labels restricted to the configured AAMI classes
(N/S/V/F by decision 2026-10-11), RR features computed within each record and quantised to the
resolution at which they are transmitted (``rr_step_s``, 8 bits each), then expanded to
5 features at the gateway: pre, post, local, pre/local, post/local.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data import mitbih
from src.data.datasets import mitbih_split_beats


@dataclass
class SplitArrays:
    x: np.ndarray        # (n, 1, L) float32
    rr: np.ndarray       # (n, 5) float32
    y: np.ndarray        # (n,) int64
    record: np.ndarray   # (n,) patient key


def quantise_rr(rr: np.ndarray, step: float, bits: int = 8) -> np.ndarray:
    top = step * (2 ** bits - 1)
    return (np.round(np.clip(rr, 0.0, top) / step) * step).astype(np.float32)


def expand_rr(rr3: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    pre, post, loc = rr3[:, 0], rr3[:, 1], rr3[:, 2]
    return np.stack([pre, post, loc, pre / (loc + eps), post / (loc + eps)], axis=1).astype(np.float32)


def causal_record_ops(record: np.ndarray, x: np.ndarray, pre_rr: np.ndarray, alpha: float,
                      long_window: int):
    """Per record, in time order, using only earlier beats (label-free, computable on device):
    template_i = EMA of the z-scored beats before beat i (first beat: itself);
    long_rr_i  = mean of the last ``long_window`` pre-RR intervals up to and including beat i."""
    tmpl = np.empty_like(x)
    long_rr = np.empty(len(x), dtype=np.float32)
    for r in np.unique(record):
        idx = np.flatnonzero(record == r)  # BeatSet keeps each record's beats in time order
        t = x[idx[0]].astype(np.float64)
        csum = np.concatenate([[0.0], np.cumsum(pre_rr[idx])])
        for j, i in enumerate(idx):
            tmpl[i] = t
            t = (1 - alpha) * t + alpha * x[i]
            lo = max(0, j + 1 - long_window) if long_window > 0 else j
            long_rr[i] = (csum[j + 1] - csum[lo]) / (j + 1 - lo)
    return tmpl, long_rr


def prepare_mitbih(root: Path, split: str, window, classes, rr_step_s: float,
                   exclude_records=(), cache_dir: Path | None = None, template_alpha: float = 0.0,
                   rr_long_window: int = 0) -> SplitArrays:
    """``template_alpha`` > 0 adds a second input channel x - template (causal EMA template);
    ``rr_long_window`` > 0 adds a transmitted long-term mean RR and two ratio features."""
    beats = mitbih_split_beats(root, split, window.pre, window.post, include_q=True, cache_dir=cache_dir)
    return arrays_from_beats(beats, classes, rr_step_s, exclude_records, template_alpha, rr_long_window)


def arrays_from_beats(beats, classes, rr_step_s: float, exclude_records=(), template_alpha: float = 0.0,
                      rr_long_window: int = 0) -> SplitArrays:
    """Device-side features of a BeatSet (any database mapped to the MIT-BIH conventions: AAMI labels,
    360 Hz, records in time order); shared by MIT-BIH and the external INCART evaluation."""
    keep_idx = [mitbih.CLASS_INDEX[c] for c in classes]
    remap = {old: new for new, old in enumerate(keep_idx)}
    xz = mitbih.zscore_per_beat(beats.x)
    rr3 = quantise_rr(beats.rr, rr_step_s)
    rr = expand_rr(rr3)
    chans = [xz[:, None, :]]
    if template_alpha > 0 or rr_long_window > 0:  # computed over every beat, before filtering
        tmpl, long_rr = causal_record_ops(beats.record, xz, beats.rr[:, 0], template_alpha, rr_long_window)
        if template_alpha > 0:
            chans.append((xz - tmpl)[:, None, :])
        if rr_long_window > 0:
            lq = quantise_rr(long_rr[:, None], rr_step_s)[:, 0]
            rr = np.concatenate([rr, lq[:, None], (rr3[:, :1] / (lq[:, None] + 1e-3)),
                                 (rr3[:, 1:2] / (lq[:, None] + 1e-3))], axis=1).astype(np.float32)
    sel = np.isin(beats.y, keep_idx) & ~np.isin(beats.record, list(exclude_records))
    x = np.concatenate(chans, axis=1)[sel].astype(np.float32)
    y = np.vectorize(remap.get)(beats.y[sel]).astype(np.int64)
    return SplitArrays(x=x, rr=rr[sel], y=y, record=beats.record[sel])


def concat_splits(parts: list[SplitArrays]) -> SplitArrays:
    return SplitArrays(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("x", "rr", "y", "record")))


def subset_records(d: SplitArrays, records, keep: bool = True) -> SplitArrays:
    sel = np.isin(d.record, list(records))
    if not keep:
        sel = ~sel
    return SplitArrays(d.x[sel], d.rr[sel], d.y[sel], d.record[sel])


def class_weights(y: np.ndarray, n_classes: int, kind: str = "balanced") -> np.ndarray:
    """Weights from the training labels only. 'balanced' = n / (C n_c); 'sqrt' softens it."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    if np.any(counts == 0):
        raise ValueError("a class is absent from the training split")
    w = len(y) / (n_classes * counts)
    if kind == "sqrt":
        w = np.sqrt(w)
    elif kind != "balanced":
        raise ValueError(kind)
    return (w / w.mean()).astype(np.float32)
