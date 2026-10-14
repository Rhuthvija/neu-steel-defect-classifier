"""Step 1 of the pipeline: create data/splits/{train,val,test}.csv.

Usage:
    python -m src.prepare_data
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config import ROOT, load_params
from src.data import make_splits, save_report


def main() -> None:
    params = load_params()
    d = params["data"]
    raw_dir = ROOT / d["raw_dir"]
    out_dir = ROOT / d["splits_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test, report = make_splits(raw_dir, d["classes"], d["val_fraction"], params["seed"])
    for name, frame in (("train", train), ("val", val), ("test", test)):
        frame.to_csv(out_dir / f"{name}.csv", index=False)

    save_report(report, ROOT / "reports" / "data_report.json")
    print(json.dumps(report["totals"], indent=2))
    if report["exact_duplicates_within_train_pool"]:
        print("Exact duplicates kept in the same split:", report["exact_duplicates_within_train_pool"])
    print(f"Splits written to {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
