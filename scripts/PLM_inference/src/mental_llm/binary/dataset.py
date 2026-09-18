"""Binary dataset loading and positive-class weighting."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_positive_alpha(values: Iterable[object]) -> float:
    observed = [
        value
        for value in values
        if value not in (None, "NA", -100) and not pd.isna(value)
    ]
    if not observed:
        return 0.5
    numeric = np.asarray(observed, dtype=float)
    positive_ratio = float(np.sum(numeric == 1.0) / len(numeric))
    return max(0.01, min(0.99, 1.0 - positive_ratio))


def load_binary_dataset(data_dir: str | Path, include_test: bool = False):
    """Load staged CSV files through Hugging Face Datasets."""
    from datasets import load_dataset

    root = Path(data_dir).expanduser().resolve()
    splits = ["train", "validation"]
    if include_test:
        splits.append("test")
    data_files = {split: str(root / f"{split}.csv") for split in splits}
    return load_dataset("csv", data_files=data_files)

