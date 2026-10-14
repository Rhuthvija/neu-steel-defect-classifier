"""Metrics that only need numpy / scikit-learn (so they are easy to unit-test)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: average gap between confidence and accuracy, weighted by bin size.

    If the model says "90% sure" on 100 images, about 90 of them should be right.
    ECE = 0 means confidence matches reality perfectly.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-12)))


def classification_metrics(labels: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    preds = probs.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "ece": expected_calibration_error(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
        "per_class": classification_report(
            labels, preds, labels=list(range(len(classes))), target_names=classes,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(classes)))).tolist(),
    }
