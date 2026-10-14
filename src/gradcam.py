"""Grad-CAM: which part of the image pushed the model towards its answer?

Steps:
1. Run the image forward and keep the feature maps of the last conv block.
2. Back-propagate the chosen class score to get the gradient for each feature map.
3. Average each map's gradient over space -> one importance weight per channel.
4. Weighted sum of the maps, keep the positive part (ReLU), resize, scale to 0-1.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._acts = None
        self._grads = None
        target_layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self._acts = output.detach()
        # A tensor hook (not a module backward hook) is safe with in-place ReLUs.
        # Skipped during normal no_grad predictions, where there is no gradient to catch.
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "_grads", g.detach()))

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """x: (1, 3, H, W) preprocessed tensor. Returns an (H, W) heatmap in [0, 1]."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(1))
            logits[0, class_idx].backward()

        weights = self._grads.mean(dim=(2, 3), keepdim=True)          # (1, C, 1, 1)
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a jet-coloured heatmap on top of the (resized) input image."""
    import matplotlib

    base = image.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    blended = (1 - alpha) * (np.asarray(base) / 255.0) + alpha * heat
    return Image.fromarray((blended * 255).astype(np.uint8))
