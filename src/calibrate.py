"""Optional step: temperature scaling, fitted on the VAL set.

Neural nets are often over-confident: they say 99% when they are right ~95% of
the time. Temperature scaling divides the logits by a single number T before the
softmax. T > 1 softens over-confident predictions; T < 1 sharpens them.
It never changes which class wins, so accuracy stays exactly the same; only the
confidence numbers become more honest.

Usage:
    python -m src.calibrate --model resnet18
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import minimize_scalar

from src.config import ROOT
from src.metrics import expected_calibration_error, negative_log_likelihood, softmax


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find T that minimises negative log-likelihood on held-out data."""
    def nll_at(log_t: float) -> float:
        return negative_log_likelihood(softmax(logits, float(np.exp(log_t))), labels)

    res = minimize_scalar(nll_at, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "resnet18"], required=True)
    args = ap.parse_args()

    model_dir = ROOT / "models" / args.model
    data = np.load(model_dir / "val_logits.npz")
    logits, labels = data["logits"], data["labels"]

    n_errors = int((logits.argmax(1) != labels).sum())
    if n_errors == 0:
        # With zero mistakes, NLL keeps falling as T -> 0, so the "best" T would make the
        # model absurdly over-confident. There is nothing to learn from, so keep T = 1.
        t, note = 1.0, "Model made no errors on val, so temperature cannot be fitted; T kept at 1.0"
    else:
        t, note = fit_temperature(logits, labels), f"Fitted on {n_errors} val errors"
    print(note)
    before, after = softmax(logits), softmax(logits, t)
    result = {
        "temperature": t,
        "note": note,
        "fitted_on": "val",
        "val_ece_before": expected_calibration_error(before, labels),
        "val_ece_after": expected_calibration_error(after, labels),
        "val_nll_before": negative_log_likelihood(before, labels),
        "val_nll_after": negative_log_likelihood(after, labels),
    }
    (model_dir / "calibration.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
