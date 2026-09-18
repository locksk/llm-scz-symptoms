"""Recovered binary metrics and bootstrap AUC intervals."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from mental_llm.binary.thresholds import sigmoid


def metric_row(
    label: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | int | str]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        roc_auc_score,
    )

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()
    sensitivity = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )
    positive_f1 = f1_score(actual, predicted, pos_label=1, zero_division=0)
    negative_f1 = f1_score(actual, predicted, pos_label=0, zero_division=0)
    both_classes = len(np.unique(actual)) > 1
    return {
        "Symptom": label,
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall/Sensitivity": sensitivity,
        "Specificity": specificity,
        "Positive F1": positive_f1,
        "Macro-F1": (positive_f1 + negative_f1) / 2,
        "Accuracy": accuracy_score(actual, predicted),
        "BAC": (sensitivity + specificity) / 2,
        "AUC": roc_auc_score(actual, probability) if both_classes else 0.5,
        "MCC": matthews_corrcoef(actual, predicted) if both_classes else 0.0,
        "Predicted Cases": int(true_positive + false_positive),
        "Actual Cases": int(true_positive + false_negative),
        "N": len(actual),
    }


def metrics_from_arrays(
    logits: np.ndarray,
    labels: np.ndarray,
    label_columns: Iterable[str],
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    columns = list(label_columns)
    probabilities = sigmoid(logits)
    threshold_vector = np.asarray([thresholds.get(label, 0.5) for label in columns])
    predictions = (probabilities >= threshold_vector).astype(int)
    rows: list[dict[str, float | int | str]] = []
    for index, label in enumerate(columns):
        valid = labels[:, index] != -100
        if valid.sum() == 0:
            continue
        row = metric_row(
            label,
            labels[:, index][valid],
            predictions[:, index][valid],
            probabilities[:, index][valid],
        )
        row["Threshold"] = thresholds.get(label, 0.5)
        rows.append(row)
    if not rows:
        raise ValueError("No evaluable labels were found.")
    frame = pd.DataFrame(rows)
    macro = frame.mean(numeric_only=True).to_dict()
    macro["Symptom"] = "OVERALL (Macro)"
    macro["Threshold"] = "N/A"
    return pd.concat([frame, pd.DataFrame([macro])], ignore_index=True), probabilities, predictions


def calculate_auc_ci(
    actual: np.ndarray,
    probability: np.ndarray,
    n_bootstraps: int = 1_000,
    seed: int = 42,
    lower: float = 0.025,
    upper: float = 0.975,
) -> tuple[float, float, float]:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(actual)) < 2:
        return 0.5, 0.5, 0.5
    base_auc = roc_auc_score(actual, probability)
    random_state = np.random.RandomState(seed)
    indices = np.arange(len(actual))
    scores: list[float] = []
    for _ in range(n_bootstraps):
        sampled = random_state.choice(indices, len(indices), replace=True)
        if len(np.unique(actual[sampled])) < 2:
            continue
        scores.append(roc_auc_score(actual[sampled], probability[sampled]))
    if not scores:
        return base_auc, base_auc, base_auc
    ordered = np.sort(scores)
    return (
        base_auc,
        ordered[int(lower * len(ordered))],
        ordered[int(upper * len(ordered))],
    )


def metrics_from_prediction_frame(
    frame: pd.DataFrame,
    label_columns: Iterable[str],
    *,
    n_bootstraps: int = 1_000,
    bootstrap_seed: int = 42,
    lower: float = 0.025,
    upper: float = 0.975,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for label in label_columns:
        actual_column = f"{label}_Actual"
        predicted_column = f"{label}_Predicted"
        probability_column = f"{label}_Prob"
        required = {actual_column, predicted_column, probability_column}
        if not required.issubset(frame.columns):
            continue
        valid = frame[frame[actual_column] != -100]
        if valid.empty:
            continue
        actual = valid[actual_column].to_numpy()
        predicted = valid[predicted_column].to_numpy()
        probability = valid[probability_column].to_numpy()
        row = metric_row(label, actual, predicted, probability)
        _, ci_lower, ci_upper = calculate_auc_ci(
            actual,
            probability,
            n_bootstraps=n_bootstraps,
            seed=bootstrap_seed,
            lower=lower,
            upper=upper,
        )
        row["CI_L"] = ci_lower
        row["CI_U"] = ci_upper
        rows.append(row)
    if not rows:
        raise ValueError("No complete prediction columns were found.")
    result = pd.DataFrame(rows)
    macro = result.mean(numeric_only=True).to_dict()
    macro["Symptom"] = "OVERALL (Macro)"
    return pd.concat([result, pd.DataFrame([macro])], ignore_index=True)

