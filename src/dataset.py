"""PyTorch Dataset that reads one of the split CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class NEUDataset(Dataset):
    def __init__(self, csv_path: str | Path, raw_dir: str | Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.raw_dir = Path(raw_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        with Image.open(self.raw_dir / row["path"]) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label_idx"])
