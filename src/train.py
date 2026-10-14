"""Train one model and log everything to MLflow.

Usage:
    python -m src.train --model baseline
    python -m src.train --model resnet18
    python -m src.train --model resnet18 --quick     # 1-2 epochs, just to check it runs

Model selection uses the VAL set only (lowest val loss). The test set is not
touched here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import ROOT, get_device, load_params, seed_everything
from src.dataset import NEUDataset
from src.metrics import classification_metrics, softmax
from src.models import build_model, count_parameters, set_backbone_trainable
from src.transforms import build_eval_transform, build_train_transform


# ---------------------------------------------------------------- helpers
def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over ``loader``. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device)).float().cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def file_md5(*paths: Path) -> str:
    h = hashlib.md5()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:12]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def plot_history(history: list[dict], out_path: Path, title: str) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for key, label in (("train_loss", "train"), ("val_loss", "val")):
        axes[0].plot(epochs, [h[key] for h in history], label=label)
    for key, label in (("train_acc", "train"), ("val_acc", "val")):
        axes[1].plot(epochs, [h[key] for h in history], label=label)
    axes[0].set_title("Loss"), axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("epoch"), ax.legend(), ax.grid(alpha=0.3)
    stage_change = [h["epoch"] for h in history if h.get("stage") == "finetune"]
    if stage_change and stage_change[0] > 1:
        for ax in axes:
            ax.axvline(stage_change[0] - 0.5, color="grey", ls="--", lw=1)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "resnet18"], required=True)
    ap.add_argument("--quick", action="store_true", help="1-2 epochs, for a smoke test")
    args = ap.parse_args()

    params = load_params()
    arch = args.model
    tp = params["train"]
    mp = dict(tp[arch])
    if args.quick:
        if arch == "baseline":
            mp["epochs"] = 1
        else:
            mp["frozen_epochs"], mp["finetune_epochs"] = 1, 1

    seed_everything(params["seed"])
    device = get_device(tp["device"])
    classes = params["data"]["classes"]
    pre_cfg = params["preprocess"]
    raw_dir = ROOT / params["data"]["raw_dir"]
    splits = ROOT / params["data"]["splits_dir"]

    train_ds = NEUDataset(splits / "train.csv", raw_dir, build_train_transform(pre_cfg))
    val_ds = NEUDataset(splits / "val.csv", raw_dir, build_eval_transform(pre_cfg))
    g = torch.Generator().manual_seed(params["seed"])
    train_dl = DataLoader(train_ds, batch_size=tp["batch_size"], shuffle=True,
                          num_workers=tp["num_workers"], generator=g)
    val_dl = DataLoader(val_ds, batch_size=tp["batch_size"], shuffle=False, num_workers=tp["num_workers"])

    model = build_model(arch, len(classes), mp["dropout"], pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    # Stages: baseline = one stage; resnet18 = frozen head training, then full fine-tuning.
    if arch == "baseline":
        stages = [("full", mp["epochs"], mp["lr"], False)]
    else:
        stages = [("frozen", mp["frozen_epochs"], mp["lr_head"], True),
                  ("finetune", mp["finetune_epochs"], mp["lr_finetune"], False)]

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment"])
    out_dir = ROOT / "models" / arch
    out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"train-{arch}" + ("-quick" if args.quick else "")):
        mlflow.set_tags({"stage": "train", "arch": arch, "git_commit": git_commit(),
                         "device": str(device), "quick": str(args.quick)})
        mlflow.log_params({"arch": arch, "seed": params["seed"], "batch_size": tp["batch_size"],
                           "image_size": pre_cfg["image_size"], "optimizer": "AdamW",
                           "early_stopping_patience": tp["early_stopping_patience"],
                           "n_params": count_parameters(model),
                           "data_version": file_md5(splits / "train.csv", splits / "val.csv"),
                           **{k: v for k, v in mp.items()}})

        history, best = [], {"val_loss": float("inf"), "epoch": 0, "state": None}
        epoch, patience_left = 0, tp["early_stopping_patience"]
        t0 = time.time()
        for stage_name, n_epochs, lr, freeze in stages:
            if arch == "resnet18":
                set_backbone_trainable(model, not freeze)
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=mp["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))
            patience_left = tp["early_stopping_patience"]  # reset per stage

            for _ in range(n_epochs):
                epoch += 1
                tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer)
                va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
                scheduler.step()
                row = {"epoch": epoch, "stage": stage_name, "train_loss": tr_loss, "train_acc": tr_acc,
                       "val_loss": va_loss, "val_acc": va_acc, "lr": optimizer.param_groups[0]["lr"]}
                history.append(row)
                mlflow.log_metrics({k: v for k, v in row.items() if k not in ("epoch", "stage")}, step=epoch)
                print(f"[{arch}] epoch {epoch:2d} ({stage_name}) "
                      f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

                if va_loss < best["val_loss"]:
                    best = {"val_loss": va_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
                    patience_left = tp["early_stopping_patience"]
                else:
                    patience_left -= 1
                    if patience_left == 0:
                        print(f"Early stopping: val loss has not improved for {tp['early_stopping_patience']} epochs")
                        break
        train_time = time.time() - t0

        # Restore the best epoch and score it on VAL (used for model selection + calibration).
        model.load_state_dict(best["state"])
        val_logits, val_labels = collect_logits(model, val_dl, device)
        val_m = classification_metrics(val_labels, softmax(val_logits), classes)

        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_dir / "model.pt")
        np.savez(out_dir / "val_logits.npz", logits=val_logits, labels=val_labels)
        preprocess = {"arch": arch, "classes": classes, "num_classes": len(classes),
                      "dropout": mp["dropout"], **pre_cfg}
        (out_dir / "preprocess.json").write_text(json.dumps(preprocess, indent=2))
        summary = {"arch": arch, "best_epoch": best["epoch"], "epochs_run": epoch,
                   "train_time_s": round(train_time, 1), "n_params": count_parameters(model),
                   "device": str(device), "seed": params["seed"],
                   "val": {k: val_m[k] for k in ("accuracy", "precision_macro", "recall_macro",
                                                 "f1_macro", "ece", "nll")},
                   "history": history}
        (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
        plot_history(history, ROOT / "reports" / arch / "training_curves.png", f"{arch} training")

        mlflow.log_metrics({f"best_val_{k}": v for k, v in summary["val"].items()})
        mlflow.log_metrics({"best_epoch": best["epoch"], "train_time_s": train_time})
        mlflow.log_artifacts(str(out_dir), artifact_path="model")
        mlflow.log_artifact(str(ROOT / "reports" / arch / "training_curves.png"))

    print(f"\nBest epoch {best['epoch']}: val acc {val_m['accuracy']:.4f}, "
          f"val macro-F1 {val_m['f1_macro']:.4f}. Saved to {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
