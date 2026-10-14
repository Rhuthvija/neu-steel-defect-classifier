"""Dataset scanning and the train / val / test split. No torch needed here.

Split policy
------------
* Kaggle ``validation/`` (60 images per class)  -> our TEST set. Never used for
  training, early stopping, calibration or picking the best model.
* Kaggle ``train/`` (240 per class) -> stratified 80/20 into TRAIN and VAL.
* Exact duplicate images (same bytes) are kept in the same split, and any
  training image that is byte-identical to a test image is dropped, so the
  test score cannot be inflated by leakage.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def scan_split_folder(folder: Path, classes: list[str]) -> pd.DataFrame:
    """Read ``<folder>/images/<class>/*.jpg`` into a table and check the labels."""
    images_dir = folder / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Expected {images_dir} (is the dataset unzipped into data/raw?)")

    found = sorted(p.name for p in images_dir.iterdir() if p.is_dir())
    if sorted(found) != sorted(classes):
        raise ValueError(f"Class folders {found} do not match params.yaml classes {classes}")

    rows = []
    for label_idx, cls in enumerate(classes):
        for p in sorted((images_dir / cls).iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                rows.append({"path": str(p), "label": cls, "label_idx": label_idx, "md5": md5_of(p)})
    return pd.DataFrame(rows)


def stratified_group_split(
    df: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per class, shuffle groups of identical images and move whole groups to val."""
    rng = random.Random(seed)
    train_parts, val_parts = [], []
    for cls in sorted(df["label"].unique()):
        cls_df = df[df["label"] == cls]
        groups = defaultdict(list)
        for idx, h in zip(cls_df.index, cls_df["md5"]):
            groups[h].append(idx)
        keys = sorted(groups)  # sorted first so the shuffle is reproducible
        rng.shuffle(keys)

        target = round(len(cls_df) * val_fraction)
        val_idx: list[int] = []
        for k in keys:
            if len(val_idx) >= target:
                break
            val_idx.extend(groups[k])
        val_parts.append(cls_df.loc[val_idx])
        train_parts.append(cls_df.drop(index=val_idx))
    return pd.concat(train_parts), pd.concat(val_parts)


def make_splits(raw_dir: Path, classes: list[str], val_fraction: float, seed: int):
    """Return (train, val, test, report). Paths in the tables are relative to raw_dir."""
    raw_dir = Path(raw_dir)
    pool = scan_split_folder(raw_dir / "train", classes)
    test = scan_split_folder(raw_dir / "validation", classes)

    leaked = pool["md5"].isin(set(test["md5"]))
    pool_clean = pool[~leaked]

    dup_groups = pool_clean.groupby("md5").filter(lambda g: len(g) > 1)
    train, val = stratified_group_split(pool_clean, val_fraction, seed)

    for frame in (train, val, test):
        frame["path"] = frame["path"].map(lambda p: str(Path(p).relative_to(raw_dir)))

    report = {
        "seed": seed,
        "val_fraction": val_fraction,
        "classes": classes,
        "counts": {
            name: frame["label"].value_counts().reindex(classes).fillna(0).astype(int).to_dict()
            for name, frame in (("train", train), ("val", val), ("test", test))
        },
        "totals": {"train": len(train), "val": len(val), "test": len(test)},
        "exact_duplicates_within_train_pool": sorted(
            Path(p).name for p in dup_groups["path"]
        ),
        "train_images_identical_to_test_dropped": int(leaked.sum()),
    }
    # Safety net: no image may appear in two splits.
    assert not (set(train["md5"]) & set(val["md5"]))
    assert not (set(train["md5"]) & set(test["md5"]))
    assert not (set(val["md5"]) & set(test["md5"]))
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), report


def read_split(splits_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(splits_dir) / f"{name}.csv")


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
