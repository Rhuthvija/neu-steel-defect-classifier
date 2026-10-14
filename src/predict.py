"""Inference: load a saved model + its preprocessing config and classify an image.

Used by the Streamlit app, the tests and from the command line:
    python -m src.predict samples/scratches_example.jpg
    python -m src.predict path/to/image.png --model-dir models/baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import ROOT
from src.gradcam import GradCAM, overlay_heatmap
from src.models import build_model, gradcam_target_layer
from src.transforms import build_eval_transform

REQUIRED_CONFIG_KEYS = {"arch", "classes", "num_classes", "image_size", "mean", "std"}


class Predictor:
    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        model_dir = Path(model_dir)
        cfg_path, weights_path = model_dir / "preprocess.json", model_dir / "model.pt"
        if not cfg_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"No trained model in {model_dir}. Expected preprocess.json and model.pt; "
                "run `python -m src.train --model resnet18` first."
            )
        self.cfg = json.loads(cfg_path.read_text())
        missing = REQUIRED_CONFIG_KEYS - self.cfg.keys()
        if missing:
            raise ValueError(f"preprocess.json is missing keys: {sorted(missing)}")

        self.classes: list[str] = self.cfg["classes"]
        self.device = torch.device(device)
        self.model = build_model(self.cfg["arch"], self.cfg["num_classes"],
                                 self.cfg.get("dropout", 0.0), pretrained=False)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.transform = build_eval_transform(self.cfg)

        cal_path = model_dir / "calibration.json"
        self.temperature = json.loads(cal_path.read_text())["temperature"] if cal_path.exists() else 1.0
        self._gradcam = None

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        logits = self.model(self.preprocess(image))[0]
        probs = torch.softmax(logits / self.temperature, dim=0).cpu().numpy()
        order = np.argsort(probs)[::-1]
        return {
            "label": self.classes[order[0]],
            "confidence": float(probs[order[0]]),
            "top_k": [(self.classes[i], float(probs[i])) for i in order[:top_k]],
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            "temperature": self.temperature,
        }

    def explain(self, image: Image.Image, class_idx: int | None = None) -> Image.Image:
        """Grad-CAM overlay for the predicted (or a chosen) class."""
        if self._gradcam is None:
            self._gradcam = GradCAM(self.model, gradcam_target_layer(self.model, self.cfg["arch"]))
        cam = self._gradcam(self.preprocess(image), class_idx)
        return overlay_heatmap(image, cam)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "resnet18"))
    args = ap.parse_args()

    from src.validation import load_and_validate_path

    image, warnings = load_and_validate_path(args.image)
    for w in warnings:
        print("Warning:", w)
    result = Predictor(args.model_dir).predict(image)
    print(f"Prediction: {result['label']}  (confidence {result['confidence']:.1%})")
    for cls, p in result["top_k"]:
        print(f"  {cls:<16} {p:.1%}")


if __name__ == "__main__":
    main()
