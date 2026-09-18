"""Build binary reports and final leaderboard tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mental_llm.config import load_config
from mental_llm.constants import safe_model_name
from mental_llm.evaluation.metrics import metrics_from_prediction_frame

TABLE_COLUMNS = (
    "Model",
    "Macro-F1",
    "BAC",
    "Positive F1",
    "Accuracy",
    "Recall/Sensitivity",
    "Specificity",
    "Precision",
    "AUC",
    "CI_L",
    "CI_U",
    "MCC",
    "Predicted Cases",
    "Actual Cases",
    "N",
)


def _prediction_paths(
    evaluation_root: Path,
    schemes: list[str],
    model_name: str,
) -> list[Path]:
    safe_name = safe_model_name(model_name)
    return [
        evaluation_root / scheme / safe_name / f"{split}_predictions_details.csv"
        for scheme in schemes
        for split in ("validation", "test")
    ]


def build_scope_report(
    config: dict,
    evaluation_root: Path,
    output_dir: Path,
    scope: str,
    schemes: list[str],
) -> pd.DataFrame:
    reports_dir = output_dir / "per_model_reports" / scope
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    bootstrap = config["bootstrap"]
    digits = int(config["round_digits"])

    for model_name in config["models"]:
        paths = _prediction_paths(evaluation_root, schemes, model_name)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Prediction files are missing: {missing}")
        combined = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
        metrics = metrics_from_prediction_frame(
            combined,
            config["labels"],
            n_bootstraps=int(bootstrap["n_resamples"]),
            bootstrap_seed=int(bootstrap["seed"]),
            lower=float(bootstrap["lower"]),
            upper=float(bootstrap["upper"]),
        )
        metrics.round(digits).to_csv(
            reports_dir / f"{safe_model_name(model_name)}.csv", index=False
        )
        overall = metrics[metrics["Symptom"] == "OVERALL (Macro)"].copy()
        overall.insert(0, "Model", model_name)
        rows.append(overall)

    leaderboard = pd.concat(rows, ignore_index=True)
    selected = [column for column in TABLE_COLUMNS if column in leaderboard]
    leaderboard = leaderboard[selected].round(digits)
    leaderboard.to_csv(output_dir / f"{scope}_leaderboard.csv", index=False)
    return leaderboard


def build_reports(config: dict, evaluation_root: Path, output_dir: Path) -> dict[str, int]:
    evaluation_root = evaluation_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for scheme in config["schemes"]:
        build_scope_report(config, evaluation_root, output_dir, scheme, [scheme])
    final_table = build_scope_report(
        config,
        evaluation_root,
        output_dir,
        "combined",
        list(config["schemes"]),
    )
    final_table.to_csv(output_dir / "binary_final_table.csv", index=False)
    return {
        "models": len(config["models"]),
        "schemes": len(config["schemes"]),
        "leaderboards": len(config["schemes"]) + 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and show the report plan without reading predictions.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "evaluation_root": str(args.evaluation_root.resolve()),
                    "output_dir": str(args.output_dir.resolve()),
                    "models": list(config["models"]),
                    "schemes": config["schemes"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(json.dumps(build_reports(config, args.evaluation_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

