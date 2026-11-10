"""Evaluation report from predictions. Writes measured numbers only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from urbiva_ai.evaluation import metrics as M
from urbiva_ai.labels import CATEGORIES


def build_report(probs: np.ndarray, labels: np.ndarray, target_fpr: float, bins: int = 15) -> dict:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.shape[0] == 0:
        raise ValueError("cannot build a report from zero predictions")
    cm = M.confusion_matrix(labels, probs.argmax(1))
    rep = M.classification_report(cm)
    scores = M.barrier_scores(probs)
    is_b = labels != M.NONE_INDEX
    thr = M.threshold_at_fpr(scores, is_b, target_fpr)
    pred = np.where(scores >= thr, 0, M.NONE_INDEX)  # 0 stands for any barrier in the binary view
    rates = M.barrier_rates(labels, pred)
    return {
        "samples": int(labels.size),
        "accuracy": rep.accuracy,
        "macro_f1": rep.macro_f1,
        "per_class": {
            c: {"precision": float(rep.precision[i]), "recall": float(rep.recall[i]),
                "f1": float(rep.f1[i]), "support": int(rep.support[i])}
            for i, c in enumerate(CATEGORIES)
        },
        "confusion_matrix": cm.tolist(),
        "barrier_operating_point": {
            "target_false_positive_rate": target_fpr,
            "threshold": thr,
            "false_positive_rate": rates.false_positive_rate,
            "true_positive_rate": rates.true_positive_rate,
            "precision": rates.precision,
        },
        "expected_calibration_error": M.expected_calibration_error(probs, labels, bins),
    }


def write_report(report: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j = out / "report.json"
    j.write_text(json.dumps(report, indent=2), encoding="utf-8")
    op = report["barrier_operating_point"]
    lines = [
        "# Evaluation report", "",
        f"Samples: {report['samples']}  ",
        f"Accuracy: {report['accuracy']:.4f}  ",
        f"Macro F1: {report['macro_f1']:.4f}  ",
        f"ECE: {report['expected_calibration_error']:.4f}", "",
        "## Barrier operating point", "",
        f"Target FPR {op['target_false_positive_rate']:.3f}, threshold {op['threshold']:.4f}: "
        f"FPR {op['false_positive_rate']:.4f}, TPR {op['true_positive_rate']:.4f}, precision {op['precision']:.4f}", "",
        "## Per class", "", "| Class | Precision | Recall | F1 | Support |", "| --- | --- | --- | --- | --- |",
    ]
    for c, v in report["per_class"].items():
        lines.append(f"| {c} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |")
    md = out / "report.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return j, md
