import numpy as np
import pytest

from urbiva_ai.evaluation import metrics as M
from urbiva_ai.evaluation.report import build_report

NONE = M.NONE_INDEX


def test_confusion_and_report():
    y = np.array([0, 0, 1, 6, 6, 6])
    p = np.array([0, 1, 1, 6, 0, 6])
    cm = M.confusion_matrix(y, p)
    assert cm.sum() == 6 and cm[0, 0] == 1 and cm[6, 0] == 1
    r = M.classification_report(cm)
    assert r.accuracy == pytest.approx(4 / 6)
    assert r.recall[6] == pytest.approx(2 / 3)


def test_barrier_rates():
    y = np.array([0, 2, NONE, NONE, NONE])
    p = np.array([0, NONE, 3, NONE, NONE])
    r = M.barrier_rates(y, p)
    assert (r.tp, r.fn, r.fp, r.tn) == (1, 1, 1, 2)
    assert r.false_positive_rate == pytest.approx(1 / 3)


def test_threshold_respects_target_fpr():
    rng = np.random.default_rng(0)
    scores = rng.random(1000)
    is_b = rng.random(1000) < 0.3
    thr = M.threshold_at_fpr(scores, is_b, 0.05)
    fpr = np.mean(scores[~is_b] >= thr)
    assert fpr <= 0.05


def test_ece_perfect_and_bad():
    probs = np.eye(7)[[0, 1, 2]]
    assert M.expected_calibration_error(probs, [0, 1, 2]) == pytest.approx(0.0)
    assert M.expected_calibration_error(probs, [1, 2, 0]) == pytest.approx(1.0)


def test_report_has_operating_point():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(200, 7))
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    labels = rng.integers(0, 7, 200)
    rep = build_report(probs, labels, 0.05)
    assert rep["samples"] == 200
    assert rep["barrier_operating_point"]["false_positive_rate"] <= 0.05 + 1e-9
    with pytest.raises(ValueError):
        build_report(np.zeros((0, 7)), np.zeros(0), 0.05)
