"""Shared helpers: read params.yaml, fix random seeds, pick a device."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"


def load_params(path: str | Path = PARAMS_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    """Make runs repeatable: same seed -> same split, same init, same batch order.

    Note: on GPU/MPS a few kernels are non-deterministic, so metrics can still
    differ in the last decimal between runs. The split itself is always identical.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # data preparation does not need torch
        pass


def get_device(preference: str = "auto"):
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
