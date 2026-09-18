"""Map prepared development files to the validation filename."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

from mental_llm.constants import ID_COLUMN, LABEL_COLUMNS, SCHEMES, STRATUM_COLUMN, TEXT_COLUMN
from mental_llm.data.validation import (
    DataValidationError,
    normalize_ids,
    read_csv_compatible,
    require_columns,
    validate_binary_labels,
)

SOURCE_TO_STAGED = {
    "train.csv": "train.csv",
    "dev.csv": "validation.csv",
    "test.csv": "test.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_prepared_data(
    prepared_dir: str | Path,
    staged_dir: str | Path,
    *,
    label_columns: Iterable[str] = LABEL_COLUMNS,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Validate and stage both schemes without changing prepared inputs."""
    source_root = Path(prepared_dir).expanduser().resolve()
    destination_root = Path(staged_dir).expanduser().resolve()
    labels = tuple(label_columns)
    plan: list[dict[str, object]] = []

    for scheme in SCHEMES:
        for source_name, staged_name in SOURCE_TO_STAGED.items():
            source_path = source_root / scheme / source_name
            frame = read_csv_compatible(source_path)
            require_columns(
                frame,
                [ID_COLUMN, TEXT_COLUMN, STRATUM_COLUMN, *labels],
                f"{scheme}/{source_name}",
            )
            if frame.empty:
                raise DataValidationError(f"{scheme}/{source_name} is empty.")
            normalized = normalize_ids(frame)
            if normalized[ID_COLUMN].duplicated().any():
                raise DataValidationError(f"{scheme}/{source_name} contains duplicate IDs.")
            validate_binary_labels(frame, labels, f"{scheme}/{source_name}")

            destination_path = destination_root / scheme / staged_name
            if destination_path.exists() and not overwrite:
                raise DataValidationError(
                    f"Staged file exists: {destination_path}. Use --overwrite."
                )
            plan.append(
                {
                    "scheme": scheme,
                    "source_name": source_name,
                    "staged_name": staged_name,
                    "rows": len(frame),
                    "source": source_path,
                    "destination": destination_path,
                }
            )

    if not dry_run:
        for item in plan:
            destination = item["destination"]
            source = item["source"]
            assert isinstance(destination, Path) and isinstance(source, Path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination)

        manifest = {
            "staging_status": "reconstructed portability step; not an archived historical step",
            "mapping": SOURCE_TO_STAGED,
            "files": [
                {
                    "scheme": item["scheme"],
                    "source_name": item["source_name"],
                    "staged_name": item["staged_name"],
                    "rows": item["rows"],
                    "sha256": _sha256(item["destination"]),
                }
                for item in plan
            ],
        }
        destination_root.mkdir(parents=True, exist_ok=True)
        with (destination_root / "staging_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    return {
        "dry_run": dry_run,
        "files": len(plan),
        "rows_by_file": {
            f"{item['scheme']}/{item['staged_name']}": item["rows"] for item in plan
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--staged-dir", type=Path, required=True)
    parser.add_argument("--label-cols", nargs="+", default=list(LABEL_COLUMNS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = stage_prepared_data(
        args.prepared_dir,
        args.staged_dir,
        label_columns=args.label_cols,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

