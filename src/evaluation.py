"""Compute binary classification metrics from class-1 probabilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def evaluate_predictions(
    target: pd.Series, probabilities: np.ndarray, threshold: float = 0.5,
) -> dict[str, float]:
    """Score a fixed threshold without fitting or changing model parameters."""
    target = pd.Series(target)
    probabilities = np.asarray(probabilities, dtype=float)
    if target.isna().any() or not target.isin([0, 1]).all() or target.nunique() != 2:
        raise ValueError("Evaluation requires both binary classes 0 and 1.")
    if probabilities.ndim != 1 or len(probabilities) != len(target):
        raise ValueError("Provide one probability per target row.")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("Probabilities must be finite values between zero and one.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The decision threshold must be between zero and one.")
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "accuracy": float(accuracy_score(target, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }
