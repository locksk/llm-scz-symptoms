"""Construct the two historical schemes from authorized inputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from mental_llm.constants import (
    FIXED_GROUP,
    ID_COLUMN,
    LABEL_COLUMNS,
    RANDOM_SEED,
    SCHEME_1,
    SCHEME_2,
    STRATUM_COLUMN,
    TEXT_COLUMN,
)
from mental_llm.data.validation import (
    DataValidationError,
    normalize_ids,
    read_csv_compatible,
    require_columns,
    validate_binary_labels,
)


@dataclass(frozen=True)
class PreparationSummary:
    source_rows: int
    stratum_rows: int
    source_duplicates_removed: int
    stratum_duplicates_removed: int
    source_only_rows: int
    stratum_only_rows: int
    joined_rows: int
    scheme_1_train: int
    scheme_1_dev: int
    scheme_1_test: int
    scheme_2_train: int
    scheme_2_dev: int
    scheme_2_test: int
    seed: int
    wrote_outputs: bool


def find_best_split_by_count(
    frame: pd.DataFrame,
    group_column: str,
    target_dev_count: int,
    target_test_count: int,
    seed: int = RANDOM_SEED,
    trials: int = 10_000,
) -> tuple[list[str], list[str], list[str]]:
    """Match the recovered randomized greedy assignment by row count."""
    group_sizes = frame[group_column].value_counts().to_dict()
    all_groups = list(group_sizes)
    if len(all_groups) < 2:
        raise DataValidationError(
            f"At least two groups are required to split {group_column!r}."
        )

    random_state = np.random.RandomState(seed)
    best_split: tuple[list[str], list[str], list[str]] | None = None
    minimum_error = float("inf")

    for _ in range(trials):
        shuffled = all_groups.copy()
        random_state.shuffle(shuffled)
        dev_groups: list[str] = []
        test_groups: list[str] = []
        dev_count = 0
        test_count = 0

        for group in shuffled:
            size = group_sizes[group]
            improves = abs(test_count + size - target_test_count) < abs(
                test_count - target_test_count
            )
            if improves or test_count == 0:
                test_groups.append(group)
                test_count += size

        remaining = [group for group in shuffled if group not in test_groups]
        for group in remaining:
            size = group_sizes[group]
            improves = abs(dev_count + size - target_dev_count) < abs(
                dev_count - target_dev_count
            )
            if improves or dev_count == 0:
                dev_groups.append(group)
                dev_count += size

        train_groups = [group for group in remaining if group not in dev_groups]
        error = abs(dev_count - target_dev_count) + abs(test_count - target_test_count)
        if error < minimum_error:
            minimum_error = error
            best_split = (train_groups, dev_groups, test_groups)
            if error == 0:
                break

    if best_split is None:
        raise DataValidationError("The split search did not produce an assignment.")
    return best_split


def _validate_partition(
    scheme: str,
    subsets: dict[str, pd.DataFrame],
    expected_ids: set[str],
) -> None:
    for split, frame in subsets.items():
        if frame.empty:
            raise DataValidationError(f"{scheme} produced an empty {split} split.")
        if frame[ID_COLUMN].duplicated().any():
            raise DataValidationError(f"{scheme} {split} contains duplicate IDs.")

    observed = {split: set(frame[ID_COLUMN]) for split, frame in subsets.items()}
    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    if any(observed[left] & observed[right] for left, right in pairs):
        raise DataValidationError(f"{scheme} contains overlapping split IDs.")
    if set().union(*observed.values()) != expected_ids:
        raise DataValidationError(f"{scheme} does not cover every joined record exactly once.")


def _statistics(
    schemes: dict[str, dict[str, pd.DataFrame]],
    label_columns: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scheme, split_frames in schemes.items():
        for split, frame in split_frames.items():
            for label in label_columns:
                numeric = pd.to_numeric(frame[label], errors="coerce")
                rows.append(
                    {
                        "Scheme": scheme,
                        "Dataset": split.capitalize(),
                        "Total_Rows": len(frame),
                        "Attribute": label,
                        "Pos": int((numeric == 1).sum()),
                        "Neg": int((numeric == 0).sum()),
                        "NA": int(numeric.isna().sum()),
                    }
                )
    return pd.DataFrame(rows)


def _prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    known_files = {
        output_dir / "split_statistics_detailed.csv",
        output_dir / "split_statistics_report.txt",
    }
    known_files.update(
        output_dir / scheme / filename
        for scheme in (SCHEME_1, SCHEME_2)
        for filename in ("train.csv", "dev.csv", "test.csv")
    )

    if output_dir.exists():
        unknown = [
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path not in known_files
        ]
        if unknown:
            raise DataValidationError("Output directory contains files not owned by this command.")
        existing = [path for path in known_files if path.exists()]
        if existing and not overwrite:
            raise DataValidationError(
                "Output files exist. Pass --overwrite to replace known outputs."
            )
        for path in existing:
            path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    for scheme in (SCHEME_1, SCHEME_2):
        (output_dir / scheme).mkdir(parents=True, exist_ok=True)


def _write_outputs(
    output_dir: Path,
    schemes: dict[str, dict[str, pd.DataFrame]],
    statistics: pd.DataFrame,
    overwrite: bool,
) -> None:
    _prepare_output_directory(output_dir, overwrite)
    for scheme, split_frames in schemes.items():
        for split, frame in split_frames.items():
            frame.to_csv(
                output_dir / scheme / f"{split}.csv",
                index=False,
                encoding="utf-8-sig",
            )

    statistics.to_csv(
        output_dir / "split_statistics_detailed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (output_dir / "split_statistics_report.txt").open("w", encoding="utf-8") as handle:
        for scheme in (SCHEME_1, SCHEME_2):
            for split in ("Train", "Dev", "Test"):
                subset = statistics[
                    (statistics["Scheme"] == scheme) & (statistics["Dataset"] == split)
                ]
                if subset.empty:
                    continue
                total = int(subset.iloc[0]["Total_Rows"])
                handle.write(f"--- {scheme} - {split} (Total Rows: {total}) ---\n")
                handle.write(f"{'Attribute':<42} {'Pos':<8} {'Neg':<8} {'NA':<8}\n")
                for row in subset.itertuples(index=False):
                    handle.write(
                        f"{row.Attribute:<42} {row.Pos:<8} {row.Neg:<8} {row.NA:<8}\n"
                    )
                handle.write("\n")


def prepare_schemes(
    source_path: str | Path,
    strata_path: str | Path,
    output_dir: str | Path,
    *,
    label_columns: Iterable[str] = LABEL_COLUMNS,
    id_column: str = ID_COLUMN,
    text_column: str = TEXT_COLUMN,
    stratum_column: str = STRATUM_COLUMN,
    fixed_group: str = FIXED_GROUP,
    seed: int = RANDOM_SEED,
    write_outputs: bool = True,
    overwrite: bool = False,
) -> PreparationSummary:
    """Normalize, deduplicate, join, and construct both schemes."""
    labels = tuple(label_columns)
    source = read_csv_compatible(source_path)
    strata = read_csv_compatible(strata_path)
    require_columns(source, [id_column, text_column, *labels], "source dataset")
    require_columns(strata, [id_column, stratum_column], "stratum metadata")
    if stratum_column in source.columns:
        raise DataValidationError(
            f"Source already contains {stratum_column!r}; join provenance is ambiguous."
        )

    source = normalize_ids(source, id_column)
    strata = normalize_ids(strata, id_column)
    source_duplicates = int(source.duplicated(id_column).sum())
    stratum_duplicates = int(strata.duplicated(id_column).sum())
    source_unique = source.drop_duplicates(id_column, keep="first")
    strata_unique = strata[[id_column, stratum_column]].drop_duplicates(
        id_column, keep="first"
    )

    source_ids = set(source_unique[id_column])
    stratum_ids = set(strata_unique[id_column])
    source_only = len(source_ids - stratum_ids)
    stratum_only = len(stratum_ids - source_ids)

    joined = pd.merge(source_unique, strata_unique, on=id_column, how="inner")
    if joined.empty:
        raise DataValidationError("The source-to-stratum join produced no records.")
    if joined[id_column].duplicated().any():
        raise DataValidationError("The joined dataset contains duplicate IDs.")
    if joined[stratum_column].isna().any():
        raise DataValidationError("The joined dataset has missing stratum values.")
    validate_binary_labels(joined, labels, "joined dataset")

    fixed = joined[joined[stratum_column] == fixed_group].copy()
    remainder = joined[joined[stratum_column] != fixed_group].copy()
    if fixed.empty or remainder.empty:
        raise DataValidationError(
            "Both the fixed group and complementary group must contain records."
        )

    remainder_dev_target = int(len(remainder) * (2 / 3))
    remainder_test_target = len(remainder) - remainder_dev_target
    _, dev_groups, test_groups = find_best_split_by_count(
        remainder,
        stratum_column,
        remainder_dev_target,
        remainder_test_target,
        seed,
    )
    scheme_1 = {
        "train": fixed,
        "dev": remainder[remainder[stratum_column].isin(dev_groups)].copy(),
        "test": remainder[remainder[stratum_column].isin(test_groups)].copy(),
    }

    fixed_dev_target = int(len(fixed) * 0.5)
    fixed_test_target = len(fixed) - fixed_dev_target
    _, dev_ids, test_ids = find_best_split_by_count(
        fixed, id_column, fixed_dev_target, fixed_test_target, seed
    )
    scheme_2 = {
        "train": remainder,
        "dev": fixed[fixed[id_column].isin(dev_ids)].copy(),
        "test": fixed[fixed[id_column].isin(test_ids)].copy(),
    }

    expected_ids = set(joined[id_column])
    _validate_partition(SCHEME_1, scheme_1, expected_ids)
    _validate_partition(SCHEME_2, scheme_2, expected_ids)
    schemes = {SCHEME_1: scheme_1, SCHEME_2: scheme_2}
    statistics = _statistics(schemes, labels)
    if write_outputs:
        _write_outputs(Path(output_dir).expanduser().resolve(), schemes, statistics, overwrite)

    return PreparationSummary(
        source_rows=len(source),
        stratum_rows=len(strata),
        source_duplicates_removed=source_duplicates,
        stratum_duplicates_removed=stratum_duplicates,
        source_only_rows=source_only,
        stratum_only_rows=stratum_only,
        joined_rows=len(joined),
        scheme_1_train=len(scheme_1["train"]),
        scheme_1_dev=len(scheme_1["dev"]),
        scheme_1_test=len(scheme_1["test"]),
        scheme_2_train=len(scheme_2["train"]),
        scheme_2_dev=len(scheme_2["dev"]),
        scheme_2_test=len(scheme_2["test"]),
        seed=seed,
        wrote_outputs=write_outputs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--strata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-cols", nargs="+", default=list(LABEL_COLUMNS))
    parser.add_argument("--fixed-group", default=FIXED_GROUP)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Construct both schemes in memory without writing outputs.",
    )
    args = parser.parse_args()
    summary = prepare_schemes(
        args.source,
        args.strata,
        args.output_dir,
        label_columns=args.label_cols,
        fixed_group=args.fixed_group,
        seed=args.seed,
        write_outputs=not args.validation_only,
        overwrite=args.overwrite,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

