"""MIT-BIH Arrhythmia Database: inter-patient DS1/DS2 split and AAMI beat extraction.

Split: de Chazal, O'Dwyer and Reilly, IEEE Trans. Biomed. Eng. 51(7), 2004
(DOI to be verified via Consensus in WP5). The four paced records (102, 104, 107, 217)
are excluded, leaving 22 training (DS1) and 22 test (DS2) records.

Known caveat: PhysioNet documents that records 201 and 202 come from the same subject;
under the standard split 201 is in DS1 and 202 in DS2. ``SAME_SUBJECT_PAIRS`` records this so
a sensitivity analysis without record 202 can be run (decision pending with Dr. Yazan).

Label mapping follows ANSI/AAMI EC57: annotation symbols are grouped into N, S, V, F, Q;
all other annotation symbols (rhythm, noise, comments) are not beats and are skipped.
Signal: the MLII lead, selected by name (it is channel 0 in most records, but not all).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS = 360  # Hz, MIT-BIH sampling rate

DS1 = (101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
       124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230)
DS2 = (100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
       212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234)
PACED = (102, 104, 107, 217)
SAME_SUBJECT_PAIRS = ((201, 202),)

AAMI_CLASSES = ("N", "S", "V", "F", "Q")
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
# EC57 grouping of beat symbols absent from MIT-BIH, used only for external databases (INCART;
# decision 2026-10-13, pre-registered before the INCART data were read)
AAMI_MAP_EXTERNAL = {**AAMI_MAP, "B": "N", "n": "S", "r": "V"}
CLASS_INDEX = {c: i for i, c in enumerate(AAMI_CLASSES)}
LEAD = "MLII"


@dataclass
class BeatSet:
    x: np.ndarray          # (n_beats, window) float32, MLII samples (mV)
    y: np.ndarray          # (n_beats,) int64 index into AAMI_CLASSES
    record: np.ndarray     # (n_beats,) int32 record id (patient key)
    r_sample: np.ndarray   # (n_beats,) int64 annotation sample index
    rr: np.ndarray         # (n_beats, 3) float32: pre-RR, post-RR, local mean pre-RR (s)

    def __len__(self) -> int:
        return len(self.y)

    @staticmethod
    def concat(parts: list["BeatSet"]) -> "BeatSet":
        return BeatSet(*(np.concatenate([getattr(p, f) for p in parts])
                         for f in ("x", "y", "record", "r_sample", "rr")))


def read_annotation_labels(root: Path, record: int):
    """Return (sample indices, AAMI labels) for beat annotations of one record."""
    import wfdb

    ann = wfdb.rdann(str(Path(root) / str(record)), "atr")
    keep = [i for i, s in enumerate(ann.symbol) if s in AAMI_MAP]
    samples = np.asarray(ann.sample, dtype=np.int64)[keep]
    labels = np.array([CLASS_INDEX[AAMI_MAP[ann.symbol[i]]] for i in keep], dtype=np.int64)
    return samples, labels


def rr_features(samples: np.ndarray, fs: int = FS, local: int = 10) -> np.ndarray:
    """Pre-RR, post-RR and the mean of the previous ``local`` pre-RR intervals, in seconds.

    Uses only beats of the same record; the first/last beat reuse the nearest valid interval.
    Post-RR implies a one-beat decision latency, which the streaming setting must account for.
    """
    s = samples.astype(np.float64)
    if len(s) < 2:
        return np.zeros((len(s), 3), dtype=np.float32)
    d = np.diff(s) / fs
    pre = np.concatenate([[d[0]], d])
    post = np.concatenate([d, [d[-1]]])
    csum = np.concatenate([[0.0], np.cumsum(pre)])
    idx = np.arange(len(pre))
    lo = np.maximum(idx - local + 1, 0)
    loc = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)
    return np.stack([pre, post, loc], axis=1).astype(np.float32)


def extract_record(root: Path, record: int, pre: int, post: int, include_q: bool = True) -> BeatSet:
    """Fixed windows [r - pre, r + post) on MLII around each AAMI beat annotation."""
    import wfdb

    rec = wfdb.rdrecord(str(Path(root) / str(record)))
    if LEAD not in rec.sig_name:
        raise ValueError(f"record {record}: lead {LEAD} absent ({rec.sig_name})")
    if rec.fs != FS:
        raise ValueError(f"record {record}: fs={rec.fs}, expected {FS}")
    sig = rec.p_signal[:, rec.sig_name.index(LEAD)].astype(np.float32)
    samples, labels = read_annotation_labels(root, record)
    rr = rr_features(samples)
    ok = (samples - pre >= 0) & (samples + post <= len(sig))
    if not include_q:
        ok &= labels != CLASS_INDEX["Q"]
    samples, labels, rr = samples[ok], labels[ok], rr[ok]
    idx = samples[:, None] + np.arange(-pre, post)[None, :]
    return BeatSet(
        x=sig[idx],
        y=labels,
        record=np.full(len(labels), record, dtype=np.int32),
        r_sample=samples,
        rr=rr,
    )


def class_counts_per_record(root: Path, records) -> dict[int, np.ndarray]:
    out = {}
    for r in records:
        _, lab = read_annotation_labels(root, r)
        out[r] = np.bincount(lab, minlength=len(AAMI_CLASSES))
    return out


def choose_validation_records(counts: dict[int, np.ndarray], target: float = 0.2,
                              sizes=(3, 4, 5, 6), classes=("N", "S", "V", "F")) -> tuple[int, ...]:
    """Deterministic patient-level validation subset of DS1 (no RNG involved).

    Enumerates every subset of DS1 records of the given sizes and picks the one whose
    per-class beat fraction is closest (sum of squared error) to ``target`` for each
    class in ``classes``. Ties break on the lexicographically smallest record tuple.
    """
    recs = sorted(counts)
    cidx = [CLASS_INDEX[c] for c in classes]
    m = np.stack([counts[r][cidx] for r in recs]).astype(np.float64)
    tot = m.sum(axis=0)
    if np.any(tot == 0):
        raise ValueError("a class has zero beats in DS1; cannot stratify on it")
    best, best_err = None, np.inf
    for k in sizes:
        for combo in itertools.combinations(range(len(recs)), k):
            frac = m[list(combo)].sum(axis=0) / tot
            err = float(np.sum((frac - target) ** 2))
            if err < best_err - 1e-12:
                best, best_err = combo, err
    return tuple(recs[i] for i in best)


def group_folds(counts: dict[int, np.ndarray], k: int, classes=("S", "V", "F", "N")) -> list[tuple[int, ...]]:
    """Deterministic patient-level folds of DS1 for development cross-validation.

    Records are taken in descending order of their (S, V, F, N) counts and each goes to the
    fold whose running per-class totals, normalised by class totals, are smallest; ties break
    on fold index. No randomness is involved.
    """
    cidx = [CLASS_INDEX[c] for c in classes]
    tot = np.maximum(np.sum([counts[r][cidx] for r in counts], axis=0), 1)
    order = sorted(counts, key=lambda r: tuple(-counts[r][cidx]) + (r,))
    load = np.zeros((k, len(cidx)))
    folds = [[] for _ in range(k)]
    for r in order:
        share = counts[r][cidx] / tot
        f = int(np.argmin([np.max(load[i] + share) + 1e-9 * i for i in range(k)]))
        load[f] += share
        folds[f].append(r)
    return [tuple(sorted(f)) for f in folds]


def zscore_per_beat(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-beat standardisation: uses no dataset-level statistics, so it cannot leak."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return ((x - mu) / (sd + eps)).astype(np.float32)
