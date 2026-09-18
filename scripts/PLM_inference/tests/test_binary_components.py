"""Synthetic unit tests for independent binary components."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mental_llm.binary.checkpoints import final_checkpoint_dir
from mental_llm.binary.dataset import calculate_positive_alpha
from mental_llm.binary.inference import aggregate_label_predictions
from mental_llm.binary.thresholds import apply_threshold, select_f1_threshold
from mental_llm.config import load_config, training_parameters
from mental_llm.constants import ID_COLUMN, LABEL_COLUMNS, SCHEME_1, SCHEME_2
from mental_llm.evaluation.metrics import calculate_auc_ci, metrics_from_prediction_frame

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_CONFIG = REPOSITORY_ROOT / "configs/historical/binary.json"


def test_historical_configuration_is_complete() -> None:
    config = load_config(HISTORICAL_CONFIG)
    assert tuple(config["schemes"]) == (SCHEME_1, SCHEME_2)
    assert len(config["models"]) == 5
    assert config["models"]["AIMH/mental-longformer-base-4096"]["max_length"] == 4096
    bigbird = config["models"]["google/bigbird-roberta-large"]
    assert bigbird["max_length"] == 4096
    assert bigbird["tokenizer_use_fast"] is False
    for scheme in config["schemes"]:
        for model in config["models"]:
            for label in config["labels"]:
                parameters = training_parameters(config, scheme, model, label)
                assert {
                    "learning_rate",
                    "num_train_epochs",
                    "gradient_accumulation_steps",
                    "warmup_ratio",
                    "weight_decay",
                    "per_device_train_batch_size",
                    "per_device_eval_batch_size",
                }.issubset(parameters)


def test_checkpoint_path_is_label_specific() -> None:
    path = final_checkpoint_dir(
        Path("/tmp/synthetic-checkpoints"),
        SCHEME_1,
        "mental/mental-bert-base-uncased",
        LABEL_COLUMNS[0],
    )
    assert path.parts[-4:] == (
        SCHEME_1,
        "mental_mental-bert-base-uncased",
        LABEL_COLUMNS[0],
        "final_model",
    )
    assert path.name == "final_model"


def test_positive_alpha_ignores_missing_values_and_clips() -> None:
    assert calculate_positive_alpha([1, 0, None, "NA", np.nan, -100]) == 0.5
    assert calculate_positive_alpha([1] * 100) == 0.01
    assert calculate_positive_alpha([0] * 100) == 0.99


def test_threshold_selection_and_application() -> None:
    actual = np.array([0, 0, 1, 1, -100])
    probability = np.array([0.1, 0.3, 0.4, 0.8, 0.99])
    threshold = select_f1_threshold(actual, probability)
    assert threshold == pytest.approx(0.30)
    assert apply_threshold(probability[:4], threshold).tolist() == [0, 0, 1, 1]


def test_independent_prediction_aggregation_preserves_label_order() -> None:
    ids = ["SYN-001", "SYN-002"]
    frames = {}
    for index, label in enumerate(LABEL_COLUMNS[:2]):
        frames[label] = pd.DataFrame(
            {
                ID_COLUMN: ids,
                f"{label}_Actual": [0, 1],
                f"{label}_Predicted": [index, 1],
                f"{label}_Prob": [0.2, 0.8],
            }
        )
    combined = aggregate_label_predictions(frames, list(LABEL_COLUMNS[:2]))
    assert combined.columns.tolist() == [
        ID_COLUMN,
        f"{LABEL_COLUMNS[0]}_Actual",
        f"{LABEL_COLUMNS[0]}_Predicted",
        f"{LABEL_COLUMNS[0]}_Prob",
        f"{LABEL_COLUMNS[1]}_Actual",
        f"{LABEL_COLUMNS[1]}_Predicted",
        f"{LABEL_COLUMNS[1]}_Prob",
    ]
    frames[LABEL_COLUMNS[1]][ID_COLUMN] = ids[::-1]
    with pytest.raises(ValueError, match="misaligned"):
        aggregate_label_predictions(frames, list(LABEL_COLUMNS[:2]))


def test_metrics_and_bootstrap_are_deterministic() -> None:
    label = LABEL_COLUMNS[0]
    frame = pd.DataFrame(
        {
            f"{label}_Actual": [0, 0, 1, 1, -100],
            f"{label}_Predicted": [0, 1, 1, 1, 0],
            f"{label}_Prob": [0.1, 0.7, 0.8, 0.9, 0.2],
        }
    )
    metrics = metrics_from_prediction_frame(frame, [label], n_bootstraps=100)
    row = metrics.iloc[0]
    assert row["Recall/Sensitivity"] == 1.0
    assert row["Specificity"] == 0.5
    assert row["BAC"] == 0.75
    first = calculate_auc_ci(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.8, 0.9]), n_bootstraps=100
    )
    second = calculate_auc_ci(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.8, 0.9]), n_bootstraps=100
    )
    assert first == second

