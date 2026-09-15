import pytest

from src.evaluation.evaluate import check_overfitting, compute_metrics


def test_compute_metrics_matches_hand_computed_values():
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(0.6)
    assert metrics["recall"] == pytest.approx(0.6)
    assert metrics["f1"] == pytest.approx(0.6)
    assert metrics["confusion_matrix"].tolist() == [[5, 2], [2, 3]]


def test_compute_metrics_perfect_predictions():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 0]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_check_overfitting_flags_a_gap_above_threshold():
    train_metrics = {"accuracy": 0.95, "precision": 0.90, "recall": 0.90, "f1": 0.90}
    test_metrics = {"accuracy": 0.94, "precision": 0.80, "recall": 0.89, "f1": 0.89}

    result = check_overfitting(train_metrics, test_metrics, threshold=0.05)

    assert result["is_overfit"] is True
    assert "precision" in result["flagged_metrics"]
    assert "accuracy" not in result["flagged_metrics"]
    assert result["gaps"]["precision"] == pytest.approx(0.10)


def test_check_overfitting_passes_within_threshold():
    train_metrics = {"accuracy": 0.80, "precision": 0.75, "recall": 0.75, "f1": 0.75}
    test_metrics = {"accuracy": 0.78, "precision": 0.73, "recall": 0.74, "f1": 0.73}

    result = check_overfitting(train_metrics, test_metrics, threshold=0.05)

    assert result["is_overfit"] is False
    assert result["flagged_metrics"] == []
