"""The split must be stratified, disjoint, reproducible and leak-free."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data import make_splits

CLASSES = ["a", "b"]


def _make_fake_dataset(root: Path, per_class_train=10, per_class_test=4):
    rng = np.random.default_rng(0)
    for split, n in (("train", per_class_train), ("validation", per_class_test)):
        for c in CLASSES:
            d = root / split / "images" / c
            d.mkdir(parents=True)
            for i in range(n):
                Image.fromarray(rng.integers(0, 255, (16, 16), dtype=np.uint8)).save(d / f"{c}_{split}_{i}.png")
    return root


def test_split_sizes_and_stratification(tmp_path):
    raw = _make_fake_dataset(tmp_path)
    train, val, test, report = make_splits(raw, CLASSES, 0.2, seed=42)
    assert len(train) == 16 and len(val) == 4 and len(test) == 8
    assert val["label"].value_counts().to_dict() == {"a": 2, "b": 2}
    assert report["totals"] == {"train": 16, "val": 4, "test": 8}


def test_splits_are_disjoint(tmp_path):
    raw = _make_fake_dataset(tmp_path)
    train, val, test, _ = make_splits(raw, CLASSES, 0.2, seed=42)
    assert not set(train["path"]) & set(val["path"])
    assert not set(train["path"]) & set(test["path"])
    assert all(p.startswith("validation/") for p in test["path"])  # test = Kaggle validation only


def test_same_seed_same_split_different_seed_different_split(tmp_path):
    raw = _make_fake_dataset(tmp_path)
    v1 = make_splits(raw, CLASSES, 0.2, seed=1)[1]["path"].tolist()
    v1_again = make_splits(raw, CLASSES, 0.2, seed=1)[1]["path"].tolist()
    v2 = make_splits(raw, CLASSES, 0.2, seed=2)[1]["path"].tolist()
    assert v1 == v1_again
    assert v1 != v2


def test_duplicates_stay_together_and_test_copies_are_dropped(tmp_path):
    raw = _make_fake_dataset(tmp_path)
    d = raw / "train" / "images" / "a"
    (d / "a_dup.png").write_bytes((d / "a_train_0.png").read_bytes())          # duplicate inside train
    test_img = raw / "validation" / "images" / "b" / "b_validation_0.png"
    (raw / "train" / "images" / "b" / "b_leak.png").write_bytes(test_img.read_bytes())  # copy of a test image

    train, val, test, report = make_splits(raw, CLASSES, 0.2, seed=42)
    dup_paths = {"train/images/a/a_dup.png", "train/images/a/a_train_0.png"}
    assert dup_paths <= set(train["path"]) or dup_paths <= set(val["path"])
    assert "train/images/b/b_leak.png" not in set(train["path"]) | set(val["path"])
    assert report["train_images_identical_to_test_dropped"] == 1


def test_wrong_class_folders_raise(tmp_path):
    raw = _make_fake_dataset(tmp_path)
    with pytest.raises(ValueError, match="do not match"):
        make_splits(raw, ["a", "b", "c"], 0.2, seed=42)
