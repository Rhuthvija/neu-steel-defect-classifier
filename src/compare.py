"""Compare both models side by side and pick the one to deploy.

The pick is made on VALIDATION macro-F1 (ties broken by val loss), never on the
test set. Test numbers are reported next to it for information only.

Usage:
    python -m src.compare
"""
from __future__ import annotations

import json

from src.config import ROOT

MODELS = ["baseline", "resnet18"]


def load(name: str) -> dict | None:
    train_p = ROOT / "models" / name / "train_summary.json"
    test_p = ROOT / "reports" / name / "metrics.json"
    if not train_p.exists():
        return None
    out = {"train": json.loads(train_p.read_text())}
    out["test"] = json.loads(test_p.read_text()) if test_p.exists() else None
    return out


def main() -> None:
    rows = {m: r for m in MODELS if (r := load(m)) is not None}
    if not rows:
        raise SystemExit("No trained models found. Train first.")

    winner = min(rows, key=lambda m: (-rows[m]["train"]["val"]["f1_macro"], rows[m]["train"]["val"]["nll"]))

    def fmt(v, pct=True):
        return "-" if v is None else (f"{v:.2%}" if pct else f"{v}")

    lines = ["# Model comparison", "",
             f"Selected for deployment (by **validation** macro-F1): **{winner}**", "",
             "| Metric | " + " | ".join(rows) + " |",
             "|---|" + "---|" * len(rows)]

    def add(label, getter, pct=True):
        lines.append(f"| {label} | " + " | ".join(fmt(getter(r), pct) for r in rows.values()) + " |")

    add("Val accuracy", lambda r: r["train"]["val"]["accuracy"])
    add("Val macro-F1", lambda r: r["train"]["val"]["f1_macro"])
    add("**Test accuracy**", lambda r: r["test"] and r["test"]["accuracy"])
    add("Test macro precision", lambda r: r["test"] and r["test"]["precision_macro"])
    add("Test macro recall", lambda r: r["test"] and r["test"]["recall_macro"])
    add("**Test macro-F1**", lambda r: r["test"] and r["test"]["f1_macro"])
    add("Test errors (of 360)", lambda r: r["test"] and r["test"]["n_misclassified"], pct=False)
    add("Test ECE (calibrated)", lambda r: r["test"] and round(r["test"]["ece_calibrated"], 4), pct=False)
    add("Parameters", lambda r: f"{r['train']['n_params'] / 1e6:.2f} M", pct=False)
    add("Model size (MB)", lambda r: r["test"] and r["test"]["model_size_mb"], pct=False)
    add("CPU latency (ms/image)", lambda r: r["test"] and r["test"]["cpu_latency_ms"], pct=False)
    add("Training time (s)", lambda r: r["train"]["train_time_s"], pct=False)
    add("Best epoch / epochs run",
        lambda r: f"{r['train']['best_epoch']} / {r['train']['epochs_run']}", pct=False)

    text = "\n".join(lines) + "\n"
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
