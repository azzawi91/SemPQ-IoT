"""St Petersburg INCART 12-lead Arrhythmia Database (PhysioNet ``incartdb`` 1.0.0) as an external test cohort.

Used only to evaluate frozen models (decision and pre-registration 2026-10-13, docs/DECISIONS.md):
75 records of 30 min, 12 leads at 257 Hz, reference beat annotations ("atr"). Lead II enters the MLII input
slot; signals are resampled 257 -> 360 Hz by polyphase filtering and annotation positions are mapped by
round(n * 360 / 257). Beat symbols follow the ANSI/AAMI EC57 grouping (src.data.mitbih.AAMI_MAP_EXTERNAL);
windows, RR features and edge handling are those of the MIT-BIH extraction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import mitbih

FS_NATIVE = 257
FS = mitbih.FS  # 360 Hz, the rate the models were trained at
LEAD = "II"
RECORDS = tuple(f"I{i:02d}" for i in range(1, 76))


def resample_signal(sig: np.ndarray, fs_in: int = FS_NATIVE, fs_out: int = FS) -> np.ndarray:
    from scipy.signal import resample_poly

    return resample_poly(sig.astype(np.float64), fs_out, fs_in).astype(np.float32)


def map_positions(samples: np.ndarray, fs_in: int = FS_NATIVE, fs_out: int = FS) -> np.ndarray:
    return np.round(np.asarray(samples, dtype=np.float64) * fs_out / fs_in).astype(np.int64)


def beat_annotations(symbols, samples):
    """Keep beat annotations (EC57 grouping); return (samples, labels, per-symbol counts incl. skipped)."""
    amap = mitbih.AAMI_MAP_EXTERNAL
    keep = [i for i, s in enumerate(symbols) if s in amap]
    counts: dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    smp = np.asarray(samples, dtype=np.int64)[keep]
    lab = np.array([mitbih.CLASS_INDEX[amap[symbols[i]]] for i in keep], dtype=np.int64)
    return smp, lab, counts


def windows_from_signal(sig: np.ndarray, samples: np.ndarray, labels: np.ndarray, record_key: int,
                        pre: int, post: int) -> mitbih.BeatSet:
    """Same steps as mitbih.extract_record: RR features over every beat, then edge filtering."""
    rr = mitbih.rr_features(samples, fs=FS)
    ok = (samples - pre >= 0) & (samples + post <= len(sig))
    samples, labels, rr = samples[ok], labels[ok], rr[ok]
    idx = samples[:, None] + np.arange(-pre, post)[None, :]
    return mitbih.BeatSet(x=sig[idx], y=labels, record=np.full(len(labels), record_key, dtype=np.int32),
                          r_sample=samples, rr=rr)


def extract_record(root: Path, name: str, pre: int, post: int):
    """Return (BeatSet at 360 Hz, per-symbol counts, header comments) for one INCART record."""
    import wfdb

    rec = wfdb.rdrecord(str(Path(root) / name))
    if rec.fs != FS_NATIVE:
        raise ValueError(f"{name}: fs={rec.fs}, expected {FS_NATIVE}")
    if LEAD not in rec.sig_name:
        raise ValueError(f"{name}: lead {LEAD} absent ({rec.sig_name})")
    sig = resample_signal(rec.p_signal[:, rec.sig_name.index(LEAD)])
    ann = wfdb.rdann(str(Path(root) / name), "atr")
    smp, lab, counts = beat_annotations(ann.symbol, ann.sample)
    beats = windows_from_signal(sig, map_positions(smp), lab, int(name[1:]), pre, post)
    return beats, counts, list(rec.comments or [])


def load_all(root: Path, pre: int, post: int, records=RECORDS):
    parts, counts, comments = [], {}, {}
    for r in records:
        b, c, com = extract_record(root, r, pre, post)
        parts.append(b)
        counts[r] = c
        comments[r] = com
    return mitbih.BeatSet.concat(parts), counts, comments


def subject_groups(comments: dict[str, list[str]]) -> dict[str, int]:
    """Sensitivity analysis only: record -> subject index. INCART headers carry a comment line "patient <k>";
    records with the same k come from one subject. Records without such a line are their own subject.
    (Fix 2026-10-13: the first implementation compared whole comment blocks, which also contain per-record
    rhythm notes, and therefore found no shared subjects.)"""
    import re

    out, extra = {}, 10 ** 6
    for r in sorted(comments):
        m = next((re.search(r"\bpatient\s+(\d+)", c, re.I) for c in comments[r]
                  if re.search(r"\bpatient\s+(\d+)", c, re.I)), None)
        if m:
            out[r] = int(m.group(1))
        else:
            out[r] = extra
            extra += 1
    return out
