"""Privacy-safe tabular validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mental_llm.constants import ID_COLUMN, LABEL_COLUMNS, TEXT_COLUMN


class DataValidationError(ValueError):
    """Raised when a dataset does not satisfy the data contract."""


def read_csv_compatible(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise DataValidationError(f"CSV file not found: {csv_path}")

    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            frame = pd.read_csv(csv_path, encoding=encoding, keep_default_na=True)
            frame.columns = frame.columns.str.strip()
            return frame
        except UnicodeDecodeError as error:
            last_error = error
    raise DataValidationError(f"Unable to decode CSV file: {csv_path}") from last_error


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DataValidationError(f"{source_name} is missing required columns: {missing}")


def normalize_ids(frame: pd.DataFrame, id_column: str = ID_COLUMN) -> pd.DataFrame:
    require_columns(frame, [id_column], "dataset")
    if frame[id_column].isna().any():
        raise DataValidationError("The ID column contains missing values.")

    normalized = frame.copy()
    normalized[id_column] = normalized[id_column].astype(str).str.strip().str.upper()
    if (normalized[id_column] == "").any():
        raise DataValidationError("The ID column contains blank values after normalization.")
    return normalized


def validate_binary_labels(
    frame: pd.DataFrame,
    label_columns: Iterable[str] = LABEL_COLUMNS,
    source_name: str = "dataset",
) -> None:
    labels = list(label_columns)
    require_columns(frame, labels, source_name)
    for label in labels:
        numeric = pd.to_numeric(frame[label], errors="coerce")
        invalid_text = frame[label].notna() & numeric.isna()
        invalid_domain = numeric.notna() & ~numeric.isin([0, 1])
        if invalid_text.any() or invalid_domain.any():
            raise DataValidationError(
                f"{source_name} column {label!r} contains values outside 0, 1, or missing."
            )


def validate_model_data_dir(
    data_dir: str | Path,
    text_column: str = TEXT_COLUMN,
    label_columns: Iterable[str] = LABEL_COLUMNS,
) -> dict[str, int]:
    root = Path(data_dir).expanduser().resolve()
    counts: dict[str, int] = {}
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        frame = read_csv_compatible(root / f"{split}.csv")
        require_columns(frame, [ID_COLUMN, text_column, *label_columns], split)
        if frame.empty:
            raise DataValidationError(f"{split}.csv is empty.")
        normalized = normalize_ids(frame)
        if normalized[ID_COLUMN].duplicated().any():
            raise DataValidationError(f"{split}.csv contains duplicate IDs.")
        validate_binary_labels(frame, label_columns, split)
        frames[split] = normalized
        counts[split] = len(frame)

    split_ids = {split: set(frame[ID_COLUMN]) for split, frame in frames.items()}
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for left, right in pairs:
        if split_ids[left] & split_ids[right]:
            raise DataValidationError(f"{left} and {right} contain overlapping IDs.")
    return counts

