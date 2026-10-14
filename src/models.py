"""The two models we compare.

1. ``baseline``  - a small CNN trained from scratch (about 1.2M parameters).
   Tells us how hard the task is without any outside knowledge.
2. ``resnet18``  - ImageNet-pretrained ResNet18 (about 11M parameters) with a new
   6-class head. Tells us how much transfer learning adds.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    """4 conv blocks (32-64-128-256) -> global average pool -> dropout -> linear."""

    def __init__(self, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_resnet18(num_classes: int = 6, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    return model


def build_model(arch: str, num_classes: int = 6, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    if arch == "baseline":
        return BaselineCNN(num_classes, dropout)
    if arch == "resnet18":
        return build_resnet18(num_classes, dropout, pretrained)
    raise ValueError(f"Unknown architecture '{arch}'. Use 'baseline' or 'resnet18'.")


def gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Last convolutional block: where Grad-CAM reads activations and gradients."""
    if arch == "baseline":
        return model.features[-1]
    if arch == "resnet18":
        return model.layer4[-1]
    raise ValueError(arch)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except ResNet's new ``fc`` head."""
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = trainable


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
