"""Image preprocessing. The SAME eval transform is used in evaluation and in the app,
built from the preprocess.json saved next to each model, so training and serving
can never drift apart (train/serve skew)."""
from __future__ import annotations

from PIL import Image
from torchvision import transforms as T


def to_rgb(img: Image.Image) -> Image.Image:
    """NEU images are grayscale; uploads may be L, RGBA, P, I;16... -> always 3-channel RGB."""
    if img.mode == "RGB":
        return img
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):  # 16-bit TIFF/PNG
        img = img.point(lambda v: v * (1 / 256)).convert("L")
    return img.convert("RGB")


def build_eval_transform(cfg: dict) -> T.Compose:
    """cfg = {"image_size": 224, "mean": [...], "std": [...]}"""
    return T.Compose(
        [
            T.Lambda(to_rgb),
            T.Resize((cfg["image_size"], cfg["image_size"])),
            T.ToTensor(),
            T.Normalize(cfg["mean"], cfg["std"]),
        ]
    )


def build_train_transform(cfg: dict) -> T.Compose:
    """Light augmentation that keeps defects realistic.

    * Flips: a scratch or patch is still the same defect when mirrored.
    * Small rotation/shift: camera position on a line is never perfect.
    * Brightness/contrast: lighting on a production line varies.
    No colour jitter (images are grayscale) and no heavy crops (could cut the defect out).
    """
    size = cfg["image_size"]
    return T.Compose(
        [
            T.Lambda(to_rgb),
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(cfg["mean"], cfg["std"]),
        ]
    )


def denormalize(tensor, cfg: dict):
    """Undo Normalize for display (C,H,W tensor -> H,W,C numpy in [0,1])."""
    import torch

    mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["std"]).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
