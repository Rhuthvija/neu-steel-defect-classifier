"""Streamlit app: upload a steel-surface image -> defect class + confidence.

Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import ROOT, load_params
from src.validation import InvalidImageError, validate_image_bytes

PARAMS = load_params()
APP_CFG = PARAMS["app"]
SAMPLES_DIR = ROOT / "samples"

DESCRIPTIONS = {
    "crazing": "A network of fine surface cracks.",
    "inclusion": "Foreign material (e.g. slag) trapped in the steel surface.",
    "patches": "Irregular darker or lighter areas on the surface.",
    "pitted_surface": "Small pits or craters scattered over the surface.",
    "rolled-in_scale": "Oxide scale pressed into the surface during rolling.",
    "scratches": "Linear marks, usually from contact with equipment.",
}

st.set_page_config(page_title="Steel Defect Classifier", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_predictor(model_dir: str):
    from src.predict import Predictor  # imported lazily so the page renders fast

    return Predictor(model_dir, device="cpu")


def available_models() -> dict[str, str]:
    found = {}
    for name, label in (("resnet18", "ResNet18 (pretrained, fine-tuned)"), ("baseline", "Baseline CNN (from scratch)")):
        if (ROOT / "models" / name / "model.pt").exists():
            found[label] = str(ROOT / "models" / name)
    return found


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    models = available_models()
    if not models:
        st.error("No trained model found in `models/`. Run `python -m src.train --model resnet18` first.")
        st.stop()
    model_label = st.selectbox("Model", list(models))
    threshold = st.slider("Low-confidence warning below", 0.3, 0.95,
                          float(APP_CFG["low_confidence_threshold"]), 0.05)
    show_cam = st.toggle("Show Grad-CAM heatmap", value=True)
    st.divider()
    st.caption("Classes")
    for cls, text in DESCRIPTIONS.items():
        st.markdown(f"**{cls}**: {text}")

# ---------------------------------------------------------------- main
st.title("🔍 Steel Surface Defect Classifier")
st.write("Upload a close-up image of a steel surface. The model predicts one of six "
         "defect types from the NEU Surface Defect dataset and shows how confident it is.")

uploaded = st.file_uploader("Upload an image (JPG, PNG, BMP or TIFF, up to "
                            f"{APP_CFG['max_upload_mb']} MB)")
samples = sorted(p.name for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".png", ".bmp"})
sample_choice = st.selectbox("...or try a sample image", ["(none)"] + samples) if samples else "(none)"

if uploaded is not None:
    data, name = uploaded.getvalue(), uploaded.name
elif sample_choice != "(none)":
    data, name = (SAMPLES_DIR / sample_choice).read_bytes(), sample_choice
else:
    st.info("Waiting for an image.")
    st.stop()

try:
    image, warnings = validate_image_bytes(data, name, max_mb=APP_CFG["max_upload_mb"])
except InvalidImageError as e:
    st.error(f"**Can't use this file.** {e}")
    st.stop()

try:
    predictor = get_predictor(models[model_label])
    result = predictor.predict(image)
except Exception as e:  # never show a raw traceback to the user
    st.error("Something went wrong while running the model. Please try another image.")
    with st.expander("Technical details"):
        st.code(repr(e))
    st.stop()

for w in warnings:
    st.warning(w)

col_img, col_res = st.columns([1, 1.3], gap="large")
with col_img:
    st.subheader("Input")
    st.image(image, caption=f"{name} ({image.width}x{image.height})")
    if show_cam:
        try:
            st.image(predictor.explain(image),
                     caption="Grad-CAM: red areas influenced the prediction most")
        except Exception:
            st.caption("Grad-CAM is not available for this image.")

with col_res:
    st.subheader("Prediction")
    label, conf = result["label"], result["confidence"]
    st.markdown(f"### {label.replace('_', ' ').title()}")
    st.caption(DESCRIPTIONS.get(label, ""))
    st.metric("Confidence", f"{conf:.1%}")
    st.progress(min(max(conf, 0.0), 1.0))
    if conf < threshold:
        st.warning(f"Low confidence (below {threshold:.0%}). Treat this as uncertain and "
                   "inspect the part manually.")
    else:
        st.success("Confident prediction.")

    st.markdown("**All classes**")
    probs = pd.DataFrame({"probability": result["probabilities"]})
    st.bar_chart(probs)
    with st.expander("Details"):
        st.write(pd.DataFrame(result["top_k"], columns=["class", "probability"])
                 .style.format({"probability": "{:.2%}"}))
        st.caption(f"Model: {model_label} · temperature T = {result['temperature']:.2f} "
                   "(T = 1 means no calibration applied)")

st.divider()
st.caption("Prototype for the PRADRIX AI/ML assignment. Trained on 200x200 grayscale "
           "NEU images; results on other kinds of images are not reliable.")
