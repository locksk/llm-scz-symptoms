"""Synthetic tests for preparation and staging."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from mental_llm.constants import ID_COLUMN, LABEL_COLUMNS, SCHEME_1, SCHEME_2
from mental_llm.data.group_split import prepare_schemes
from mental_llm.data.stage import stage_prepared_data
from mental_llm.data.validation import DataValidationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY_ROOT / "data/samples/source.synthetic.csv"
STRATA = REPOSITORY_ROOT / "data/samples/strata.synthetic.csv"


def test_both_schemes_are_deterministic_and_stage_validation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first"
        second = root / "second"
        summary = prepare_schemes(SOURCE, STRATA, first, seed=42)
        repeated = prepare_schemes(SOURCE, STRATA, second, seed=42)

        assert summary == repeated
        assert summary.source_rows == 13
        assert summary.stratum_rows == 13
        assert summary.source_duplicates_removed == 1
        assert summary.stratum_duplicates_removed == 1
        assert summary.joined_rows == 12
        assert (summary.scheme_1_train, summary.scheme_1_dev, summary.scheme_1_test) == (
            6,
            4,
            2,
        )
        assert (summary.scheme_2_train, summary.scheme_2_dev, summary.scheme_2_test) == (
            6,
            3,
            3,
        )

        for scheme in (SCHEME_1, SCHEME_2):
            split_ids: dict[str, set[str]] = {}
            for split in ("train", "dev", "test"):
                left = first / scheme / f"{split}.csv"
                right = second / scheme / f"{split}.csv"
                assert left.read_bytes() == right.read_bytes()
                frame = pd.read_csv(left, encoding="utf-8-sig")
                assert {ID_COLUMN, *LABEL_COLUMNS}.issubset(frame.columns)
                assert not frame[ID_COLUMN].duplicated().any()
                assert frame[ID_COLUMN].str.fullmatch(r"SYN-[0-9]{3}").all()
                split_ids[split] = set(frame[ID_COLUMN])
            assert not split_ids["train"] & split_ids["dev"]
            assert not split_ids["train"] & split_ids["test"]
            assert not split_ids["dev"] & split_ids["test"]
            assert len(set().union(*split_ids.values())) == 12

        staged = root / "staged"
        result = stage_prepared_data(first, staged)
        assert result["files"] == 6
        for scheme in (SCHEME_1, SCHEME_2):
            assert (staged / scheme / "train.csv").is_file()
            assert (staged / scheme / "validation.csv").is_file()
            assert (staged / scheme / "test.csv").is_file()
            assert not (staged / scheme / "dev.csv").exists()
            assert (first / scheme / "dev.csv").read_bytes() == (
                staged / scheme / "validation.csv"
            ).read_bytes()
        manifest = json.loads(
            (staged / "staging_manifest.json").read_text(encoding="utf-8")
        )
        assert "reconstructed portability step" in manifest["staging_status"]


def test_validation_only_writes_nothing() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "not-created"
        summary = prepare_schemes(SOURCE, STRATA, output, seed=42, write_outputs=False)
        assert not summary.wrote_outputs
        assert not output.exists()


def test_missing_columns_fail_clearly() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bad_source = root / "bad.csv"
        pd.DataFrame({ID_COLUMN: ["SYN-X"]}).to_csv(bad_source, index=False)
        with pytest.raises(DataValidationError, match="missing required columns"):
            prepare_schemes(bad_source, STRATA, root / "output", write_outputs=False)

