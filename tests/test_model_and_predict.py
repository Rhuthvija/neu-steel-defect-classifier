"""Preprocessing, model shapes and the Predictor (needs torch, but no trained weights)."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from src.config import ROOT, load_params  # noqa: E402
from src.models import build_model  # noqa: E402
from src.predict import Predictor  # noqa: E402
from src.transforms import build_eval_transform, build_train_transform  # noqa: E402

PARAMS = load_params()
PRE = PARAMS["preprocess"]
CLASSES = PARAMS["data"]["classes"]
SAMPLE = sorted((ROOT / "samples").glob("*.jpg"))[0]


def test_eval_transform_shape_and_normalisation():
    x = build_eval_transform(PRE)(Image.open(SAMPLE))
    assert x.shape == (3, PRE["image_size"], PRE["image_size"])
    assert x.dtype == torch.float32
    assert -3 < x.mean() < 3  # normalised, not 0..255


def test_eval_transform_is_deterministic():
    t = build_eval_transform(PRE)
    img = Image.open(SAMPLE)
    assert torch.equal(t(img), t(img))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_any_input_mode_becomes_3_channels(mode):
    img = Image.open(SAMPLE).convert(mode).resize((317, 251))  # odd size on purpose
    x = build_eval_transform(PRE)(img)
    assert x.shape == (3, PRE["image_size"], PRE["image_size"])


def test_train_transform_shape():
    x = build_train_transform(PRE)(Image.open(SAMPLE))
    assert x.shape == (3, PRE["image_size"], PRE["image_size"])


@pytest.mark.parametrize("arch", ["baseline", "resnet18"])
def test_models_output_six_logits(arch):
    model = build_model(arch, num_classes=6, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, PRE["image_size"], PRE["image_size"]))
    assert out.shape == (2, 6)


def _save_untrained(tmp_path: Path, arch: str) -> Path:
    model = build_model(arch, num_classes=6, pretrained=False)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    cfg = {"arch": arch, "classes": CLASSES, "num_classes": 6, "dropout": 0.3, **PRE}
    (tmp_path / "preprocess.json").write_text(json.dumps(cfg))
    return tmp_path


@pytest.mark.parametrize("arch", ["baseline", "resnet18"])
def test_predictor_output_format(tmp_path, arch):
    pred = Predictor(_save_untrained(tmp_path, arch))
    r = pred.predict(Image.open(SAMPLE))
    assert r["label"] in CLASSES
    assert 0 <= r["confidence"] <= 1
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-5
    assert [p for _, p in r["top_k"]] == sorted([p for _, p in r["top_k"]], reverse=True)
    assert r["top_k"][0] == (r["label"], r["confidence"])


def test_predictor_applies_temperature(tmp_path):
    d = _save_untrained(tmp_path, "baseline")
    r1 = Predictor(d).predict(Image.open(SAMPLE))
    (d / "calibration.json").write_text(json.dumps({"temperature": 5.0}))
    r5 = Predictor(d).predict(Image.open(SAMPLE))
    assert r5["label"] == r1["label"]           # temperature never changes the class
    assert r5["confidence"] <= r1["confidence"]  # but softens confidence


def test_gradcam_heatmap(tmp_path):
    pred = Predictor(_save_untrained(tmp_path, "resnet18"))
    overlay = pred.explain(Image.open(SAMPLE))
    assert overlay.size == (PRE["image_size"], PRE["image_size"])
    pred.predict(Image.open(SAMPLE))  # normal prediction still works after Grad-CAM


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No trained model"):
        Predictor(tmp_path)


@pytest.mark.skipif(not (ROOT / "models" / "resnet18" / "model.pt").exists(), reason="model not trained yet")
def test_trained_model_gets_samples_right():
    pred = Predictor(ROOT / "models" / "resnet18")
    correct = sum(pred.predict(Image.open(p))["label"] == p.stem.rsplit("_", 1)[0]
                  for p in sorted((ROOT / "samples").glob("*.jpg")))
    assert correct >= 5  # at least 5 of the 6 sample images
