"""Evaluation metrics in NumPy.

The product decision depends on keeping false positives low: a barrier that
does not exist sends people on longer routes. The binary view treats every
category except 'none' as "barrier present".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from urbiva_ai.labels import CATEGORIES, NONE_CATEGORY

NONE_INDEX = CATEGORIES.index(NONE_CATEGORY)


def _check(y_true: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y_true, dtype=np.int64)
    if y.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if y.size and (y.min() < 0 or y.max() >= n):
        raise ValueError(f"labels must be in [0, {n})")
    return y


def confusion_matrix(y_true, y_pred, num_classes: int = len(CATEGORIES)) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    t = _check(y_true, num_classes)
    p = _check(y_pred, num_classes)
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same length")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (t, p), 1)
    return cm


@dataclass(frozen=True)
class ClassReport:
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray
    macro_f1: float
    accuracy: float


def classification_report(cm: np.ndarray) -> ClassReport:
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    pred = cm.sum(axis=0)
    true = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred > 0, tp / pred, 0.0)
        recall = np.where(true > 0, tp / true, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    present = true > 0
    macro = float(f1[present].mean()) if present.any() else 0.0
    total = cm.sum()
    acc = float(tp.sum() / total) if total else 0.0
    return ClassReport(precision, recall, f1, true.astype(np.int64), macro, acc)


@dataclass(frozen=True)
class BinaryRates:
    true_positive_rate: float
    false_positive_rate: float
    precision: float
    tp: int
    fp: int
    tn: int
    fn: int


def barrier_rates(y_true, y_pred) -> BinaryRates:
    """Rates for the binary question 'is there a barrier?'."""
    t = _check(y_true, len(CATEGORIES)) != NONE_INDEX
    p = _check(y_pred, len(CATEGORIES)) != NONE_INDEX
    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    tn = int(np.sum(~t & ~p))
    fn = int(np.sum(t & ~p))
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return BinaryRates(tpr, fpr, prec, tp, fp, tn, fn)


def barrier_scores(probs: np.ndarray) -> np.ndarray:
    """Probability that any barrier is present: 1 - p(none)."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(CATEGORIES):
        raise ValueError(f"probs must have shape (n, {len(CATEGORIES)})")
    return 1.0 - probs[:, NONE_INDEX]


def threshold_at_fpr(scores: np.ndarray, is_barrier: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose false positive rate on negatives is <= target_fpr.

    A sample is predicted as barrier when score >= threshold. Returns 1.0 + eps
    when no negatives exist or no threshold satisfies the target.
    """
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")
    s = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(is_barrier, dtype=bool)
    neg_scores = np.sort(s[~pos])[::-1]
    n_neg = neg_scores.size
    if n_neg == 0:
        return float(np.nextafter(1.0, 2.0))
    allowed_fp = int(np.floor(target_fpr * n_neg))
    if allowed_fp >= n_neg:
        return float(neg_scores[-1])
    return float(np.nextafter(neg_scores[allowed_fp], np.inf))


def expected_calibration_error(probs: np.ndarray, y_true, n_bins: int = 15) -> float:
    """Expected calibration error of the top-1 prediction with equal-width bins."""
    probs = np.asarray(probs, dtype=np.float64)
    y = _check(y_true, probs.shape[1])
    if probs.shape[0] != y.shape[0]:
        raise ValueError("probs and y_true must have the same length")
    if probs.shape[0] == 0:
        return 0.0
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)
