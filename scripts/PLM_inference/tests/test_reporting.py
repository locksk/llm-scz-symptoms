"""Synthetic report and final-table test."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from mental_llm.config import load_config
from mental_llm.constants import ID_COLUMN, safe_model_name
from mental_llm.reporting.tables import build_reports

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPOSITORY_ROOT / "configs/smoke/binary.json"


def test_build_reports_from_synthetic_predictions() -> None:
    config = load_config(SMOKE_CONFIG)
    model = next(iter(config["models"]))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        evaluation_root = root / "outputs"
        report_root = root / "reports"
        for scheme_index, scheme in enumerate(config["schemes"]):
            model_dir = evaluation_root / scheme / safe_model_name(model)
            model_dir.mkdir(parents=True)
            for split_index, split in enumerate(("validation", "test")):
                frame: dict[str, object] = {
                    ID_COLUMN: [
                        f"SYN-{scheme_index}{split_index}01",
                        f"SYN-{scheme_index}{split_index}02",
                        f"SYN-{scheme_index}{split_index}03",
                        f"SYN-{scheme_index}{split_index}04",
                    ]
                }
                for label in config["labels"]:
                    frame[f"{label}_Actual"] = [0, 0, 1, 1]
                    frame[f"{label}_Predicted"] = [0, 0, 1, 1]
                    frame[f"{label}_Prob"] = [0.1, 0.2, 0.8, 0.9]
                pd.DataFrame(frame).to_csv(
                    model_dir / f"{split}_predictions_details.csv", index=False
                )
        result = build_reports(config, evaluation_root, report_root)
        assert result == {"models": 1, "schemes": 2, "leaderboards": 3}
        final_table = pd.read_csv(report_root / "binary_final_table.csv")
        assert final_table["Model"].tolist() == [model]
        assert final_table.iloc[0]["AUC"] == 1.0

