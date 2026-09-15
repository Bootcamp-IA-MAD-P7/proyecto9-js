"""Evaluate a trained classifier's predictions (accuracy, precision,
recall, F1, confusion matrix — all on the hateful class) and check
train/test metrics for overfitting against the project constitution's
5-point threshold. See specs/051-model-evaluation/.
"""
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1")


def compute_metrics(y_true, y_pred) -> dict[str, Any]:
    """Return accuracy, precision, recall, F1 (on the hateful class,
    label == 1) and the confusion matrix for a set of predictions."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "f1": f1_score(y_true, y_pred, pos_label=1),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def check_overfitting(
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Compare train vs. test metrics and flag overfitting if any of
    accuracy/precision/recall/F1 differs by more than `threshold`
    (5 percentage points by default, per the project constitution)."""
    gaps = {name: train_metrics[name] - test_metrics[name] for name in METRIC_NAMES}
    flagged_metrics = [name for name, gap in gaps.items() if abs(gap) > threshold]
    return {
        "gaps": gaps,
        "is_overfit": bool(flagged_metrics),
        "flagged_metrics": flagged_metrics,
    }
