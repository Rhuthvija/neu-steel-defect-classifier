"""Metrics and temperature scaling (numpy only)."""
import numpy as np

from src.calibrate import fit_temperature
from src.metrics import classification_metrics, expected_calibration_error, softmax


def test_softmax_rows_sum_to_one_and_temperature_keeps_argmax():
    logits = np.array([[2.0, 1.0, 0.1], [0.0, 5.0, 1.0]])
    for t in (0.5, 1.0, 3.0):
        p = softmax(logits, t)
        assert np.allclose(p.sum(1), 1)
        assert (p.argmax(1) == logits.argmax(1)).all()
    assert softmax(logits, 3.0).max() < softmax(logits, 1.0).max()  # higher T = softer


def test_ece_zero_when_confidence_matches_accuracy():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(probs, np.array([0, 1])) == 0.0


def test_ece_large_when_confidently_wrong():
    probs = np.array([[0.99, 0.01]] * 10)
    assert expected_calibration_error(probs, np.zeros(10, int) + 1) > 0.9


def test_classification_metrics_perfect():
    labels = np.array([0, 1, 2, 2])
    probs = np.eye(3)[labels]
    m = classification_metrics(labels, probs, ["x", "y", "z"])
    assert m["accuracy"] == 1.0 and m["f1_macro"] == 1.0
    assert m["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]


def test_temperature_scaling_softens_overconfident_model():
    rng = np.random.default_rng(0)
    n, k = 2000, 6
    labels = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), labels] += 2.0      # mostly right...
    logits *= 4.0                            # ...but far too confident
    t = fit_temperature(logits, labels)
    assert t > 1.5
    assert expected_calibration_error(softmax(logits, t), labels) < expected_calibration_error(softmax(logits), labels)
