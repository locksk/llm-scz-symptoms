"""Validation-set threshold selection and application."""

from __future__ import annotations

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(logits)))


def apply_threshold(probability: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(probability) >= threshold).astype(int)


def select_f1_threshold(
    actual: np.ndarray,
    probability: np.ndarray,
    *,
    start: float = 0.1,
    stop: float = 0.95,
    step: float = 0.05,
) -> float:
    """Select the first threshold with the highest positive-class F1."""
    from sklearn.metrics import f1_score

    actual = np.asarray(actual)
    probability = np.asarray(probability)
    valid = actual != -100
    if valid.sum() == 0:
        return 0.5
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in np.arange(start, stop, step):
        predicted = apply_threshold(probability[valid], float(threshold))
        score = f1_score(actual[valid], predicted, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold

