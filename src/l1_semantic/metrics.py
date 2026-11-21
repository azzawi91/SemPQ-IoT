"""Classification metrics and the automated consistency check (brief rule 4).

All metrics are computed from one array of class scores, so accuracy, macro-F1, AUROC and the
confusion matrix always describe the same predictions. ``consistency_report`` recomputes the
headline numbers from the confusion matrix and from the scores independently and returns
hard errors (identity violations: the run must fail) and warnings (implausible combinations,
e.g. a high one-vs-rest AUROC next to a near-chance decision for the same class).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc_binary(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties; NaN if a class is absent."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def confusion_matrix(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(n_classes * np.asarray(y) + np.asarray(pred),
                       minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def _per_class_from_cm(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        f1 = np.where(precision + np.nan_to_num(recall) > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    f1 = np.where(support > 0, f1, np.nan)
    tn = cm.sum() - support - predicted + tp
    neg = cm.sum() - support
    with np.errstate(divide="ignore", invalid="ignore"):
        specificity = np.where(neg > 0, tn / neg, np.nan)
    return {"recall": recall, "precision": precision, "f1": f1,
            "specificity": specificity, "support": support}


def multiclass_metrics(y: np.ndarray, scores: np.ndarray, class_names) -> dict:
    """Single-label metrics. ``scores``: (n, C) probabilities or logits; prediction = argmax."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    c = scores.shape[1]
    pred = scores.argmax(axis=1)
    cm = confusion_matrix(y, pred, c)
    pc = _per_class_from_cm(cm)
    auroc = np.array([auroc_binary(y == k, scores[:, k]) for k in range(c)])
    present = pc["support"] > 0
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(np.nanmean(pc["f1"][present])),
        "macro_auroc": float(np.nanmean(auroc[present])),
        "balanced_accuracy": float(np.nanmean(pc["recall"][present])),
        "per_class": {name: {"sensitivity": _f(pc["recall"][k]), "ppv": _f(pc["precision"][k]),
                             "f1": _f(pc["f1"][k]), "specificity": _f(pc["specificity"][k]),
                             "auroc": _f(auroc[k]), "support": int(pc["support"][k])}
                      for k, name in enumerate(class_names)},
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }


def multilabel_metrics(y: np.ndarray, probs: np.ndarray, class_names, threshold: float = 0.5) -> dict:
    """Multi-label metrics (PTB-XL superclasses): per-label binary decisions at ``threshold``."""
    y = np.asarray(y).astype(int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)
    per = {}
    f1s, aucs = [], []
    for k, name in enumerate(class_names):
        cm = confusion_matrix(y[:, k], pred[:, k], 2)
        pc = _per_class_from_cm(cm)
        auc = auroc_binary(y[:, k], probs[:, k])
        per[name] = {"sensitivity": _f(pc["recall"][1]), "ppv": _f(pc["precision"][1]),
                     "f1": _f(pc["f1"][1]), "specificity": _f(pc["specificity"][1]),
                     "auroc": _f(auc), "support": int(pc["support"][1]),
                     "confusion_matrix": cm.tolist()}
        f1s.append(pc["f1"][1])
        aucs.append(auc)
    return {
        "n": int(len(y)),
        "subset_accuracy": float((pred == y).all(axis=1).mean()),
        "macro_f1": float(np.nanmean(f1s)),
        "macro_auroc": float(np.nanmean(aucs)),
        "per_class": per,
        "class_names": list(class_names),
        "threshold": threshold,
    }


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or np.isnan(x) else float(x)


def consistency_report(m: dict, y: np.ndarray, scores: np.ndarray, auroc_gap_warn: float = 0.35,
                       tol: float = 1e-9) -> dict:
    """Recompute single-label metrics from the confusion matrix and from the scores.

    errors   - identity violations (counts, accuracy, F1, AUROC recomputation, argmax);
    warnings - per-class AUROC far above the balanced accuracy of the argmax decision for that
               class, or a class AUROC below 0.5, which is the pattern behind contradictions such
               as a macro-AUROC of 0.96 reported next to a macro-F1 of 0.28.
    """
    errors, warnings = [], []
    cm = np.asarray(m["confusion_matrix"])
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if cm.sum() != len(y) or m["n"] != len(y):
        errors.append(f"confusion matrix total {cm.sum()} != n {len(y)}")
    if not np.array_equal(cm.sum(axis=1), np.bincount(y, minlength=cm.shape[0])):
        errors.append("confusion matrix row sums != label counts")
    acc_cm = np.trace(cm) / max(cm.sum(), 1)
    if abs(acc_cm - m["accuracy"]) > tol:
        errors.append(f"accuracy {m['accuracy']} != trace/n {acc_cm}")
    pred = scores.argmax(axis=1)
    if not np.array_equal(confusion_matrix(y, pred, cm.shape[0]), cm):
        errors.append("confusion matrix does not match argmax of the scores")
    pc = _per_class_from_cm(cm)
    present = pc["support"] > 0
    f1_cm = float(np.nanmean(pc["f1"][present]))
    if abs(f1_cm - m["macro_f1"]) > tol:
        errors.append(f"macro-F1 {m['macro_f1']} != recomputed {f1_cm}")
    for k, name in enumerate(m["class_names"]):
        rep = m["per_class"][name]
        if not present[k]:
            continue
        auc = auroc_binary(y == k, scores[:, k])
        if rep["auroc"] is None or abs(auc - rep["auroc"]) > 1e-9:
            errors.append(f"{name}: AUROC {rep['auroc']} != recomputed {auc}")
            continue
        bal = (pc["recall"][k] + pc["specificity"][k]) / 2
        if auc - bal > auroc_gap_warn:
            warnings.append(f"{name}: AUROC {auc:.3f} vs argmax balanced accuracy {bal:.3f}")
        if auc < 0.5:
            warnings.append(f"{name}: AUROC {auc:.3f} below chance")
    for key in ("accuracy", "macro_f1", "macro_auroc", "balanced_accuracy"):
        if not 0.0 <= m[key] <= 1.0:
            errors.append(f"{key} {m[key]} outside [0, 1]")
    return {"errors": errors, "warnings": warnings, "ok": not errors}


class InconsistentMetrics(RuntimeError):
    pass


def assert_consistent(m: dict, y, scores, fail_on_warnings: bool = False, **kw) -> dict:
    rep = consistency_report(m, y, scores, **kw)
    if rep["errors"] or (fail_on_warnings and rep["warnings"]):
        raise InconsistentMetrics("; ".join(rep["errors"] + rep["warnings"]))
    return rep
