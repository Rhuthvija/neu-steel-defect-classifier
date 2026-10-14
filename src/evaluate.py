"""Score a trained model on the held-out TEST set (run once per final model).

Usage:
    python -m src.evaluate --model resnet18
    python -m src.evaluate --model baseline

Writes reports/<model>/: metrics.json, confusion_matrix.png,
misclassified.png, misclassified.csv. Also logs everything to MLflow.

The model and preprocessing are loaded through the same Predictor class the app
uses, so the test score reflects exactly what users get.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader

from src.config import ROOT, get_device, load_params
from src.dataset import NEUDataset
from src.metrics import classification_metrics, softmax
from src.models import count_parameters
from src.predict import Predictor
from src.train import collect_logits, git_commit


def plot_confusion(cm: np.ndarray, classes: list[str], title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"), ax.set_ylabel("True"), ax.set_title(title)
    thresh = cm.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_examples(rows: pd.DataFrame, raw_dir: Path, title: str, out: Path, max_n: int = 24) -> None:
    rows = rows.head(max_n)
    n = max(len(rows), 1)
    cols = min(6, n)
    nrows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(2.4 * cols, 2.7 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, rows.iterrows()):
        ax.imshow(Image.open(raw_dir / r["path"]).convert("L"), cmap="gray")
        ax.set_title(f"true: {r['true']}\npred: {r['pred']} ({r['confidence']:.0%})", fontsize=8,
                     color="crimson" if r["true"] != r["pred"] else "black")
    if len(rows) == 0:
        axes.flat[0].text(0.5, 0.5, "No examples", ha="center")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def measure_latency(predictor: Predictor, paths: list[Path], n: int = 50) -> float:
    """Average single-image CPU time (ms) including preprocessing, as in the app."""
    images = [Image.open(p).convert("RGB") for p in paths[:n]]
    predictor.predict(images[0])  # warm-up
    t0 = time.perf_counter()
    for img in images:
        predictor.predict(img)
    return (time.perf_counter() - t0) / len(images) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "resnet18"], required=True)
    args = ap.parse_args()

    params = load_params()
    classes = params["data"]["classes"]
    raw_dir = ROOT / params["data"]["raw_dir"]
    test_csv = ROOT / params["data"]["splits_dir"] / "test.csv"
    model_dir = ROOT / "models" / args.model
    out_dir = ROOT / "reports" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(params["train"]["device"])
    predictor = Predictor(model_dir, device=str(device))
    loader = DataLoader(NEUDataset(test_csv, raw_dir, predictor.transform),
                        batch_size=params["train"]["batch_size"], shuffle=False)
    logits, labels = collect_logits(predictor.model, loader, device)

    t = predictor.temperature
    m_raw = classification_metrics(labels, softmax(logits), classes)
    probs = softmax(logits, t)
    m = classification_metrics(labels, probs, classes)

    # ---- error analysis
    test_df = pd.read_csv(test_csv)
    pred_idx = probs.argmax(1)
    results = pd.DataFrame({
        "path": test_df["path"], "true": [classes[i] for i in labels],
        "pred": [classes[i] for i in pred_idx], "confidence": probs.max(1),
    })
    wrong = results[results["true"] != results["pred"]].sort_values("confidence", ascending=False)
    wrong.to_csv(out_dir / "misclassified.csv", index=False)
    plot_examples(wrong, raw_dir, f"{args.model}: {len(wrong)} misclassified test images",
                  out_dir / "misclassified.png")
    least_sure = results[results["true"] == results["pred"]].sort_values("confidence").head(12)
    plot_examples(least_sure, raw_dir, f"{args.model}: least confident CORRECT predictions",
                  out_dir / "least_confident_correct.png")
    plot_confusion(np.array(m["confusion_matrix"]), classes,
                   f"{args.model}: test confusion matrix (n={len(labels)})", out_dir / "confusion_matrix.png")

    # ---- cost of the model
    cpu_predictor = Predictor(model_dir, device="cpu")
    latency_ms = measure_latency(cpu_predictor, [raw_dir / p for p in test_df["path"]])
    size_mb = (model_dir / "model.pt").stat().st_size / 1e6

    summary = {
        "model": args.model, "split": "test", "n_images": int(len(labels)), "temperature": t,
        "accuracy": m["accuracy"], "precision_macro": m["precision_macro"],
        "recall_macro": m["recall_macro"], "f1_macro": m["f1_macro"],
        "ece_uncalibrated": m_raw["ece"], "ece_calibrated": m["ece"],
        "n_misclassified": int(len(wrong)), "cpu_latency_ms": round(latency_ms, 2),
        "model_size_mb": round(size_mb, 2), "n_params": count_parameters(predictor.model),
        "per_class": m["per_class"], "confusion_matrix": m["confusion_matrix"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment"])
    with mlflow.start_run(run_name=f"evaluate-{args.model}"):
        mlflow.set_tags({"stage": "evaluate", "arch": args.model, "git_commit": git_commit()})
        mlflow.log_params({"arch": args.model, "temperature": t, "split": "test"})
        mlflow.log_metrics({f"test_{k}": v for k, v in summary.items()
                            if isinstance(v, (int, float)) and k not in ("n_images", "temperature")})
        for cls in classes:
            mlflow.log_metric(f"test_f1_{cls}", m["per_class"][cls]["f1-score"])
        mlflow.log_artifacts(str(out_dir), artifact_path="evaluation")

    print(f"\n{args.model} on TEST ({len(labels)} images)")
    print(f"  accuracy {m['accuracy']:.4f} | macro P {m['precision_macro']:.4f} "
          f"R {m['recall_macro']:.4f} F1 {m['f1_macro']:.4f}")
    print(f"  ECE {m_raw['ece']:.4f} -> {m['ece']:.4f} (T={t:.2f}) | {len(wrong)} errors | "
          f"{latency_ms:.1f} ms/image on CPU | {size_mb:.1f} MB")
    print(f"  Reports in {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
