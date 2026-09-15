"""Evaluate the persisted baseline model on both the training and
held-out test splits, report accuracy/precision/recall/F1/confusion
matrix, and flag overfitting if the train/test gap on any metric exceeds
the project constitution's 5-point threshold.

Requires data/processed/baseline_model.joblib and
data/processed/baseline_vectorizer.joblib (see
scripts/train_baseline_model.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.evaluate import check_overfitting, compute_metrics  # noqa: E402
from src.models.train import split_dataset  # noqa: E402

INPUT_PATH = Path("data/processed/enriched_comments_es_augmented_preprocessed.csv")
MODEL_PATH = Path("data/processed/baseline_model.joblib")
VECTORIZER_PATH = Path("data/processed/baseline_vectorizer.joblib")


def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n{label}:")
    for name in ("accuracy", "precision", "recall", "f1"):
        print(f"  {name}: {metrics[name]:.4f}")
    print(f"  confusion_matrix (rows=true, cols=pred, [[TN,FP],[FN,TP]]):")
    print(f"  {metrics['confusion_matrix'].tolist()}")


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    # Same split_dataset call (same random_state) as
    # scripts/train_baseline_model.py, so this reconstructs the exact
    # train/test membership the persisted model was trained/evaluated on.
    X_train, X_test, y_train, y_test = split_dataset(df)

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)

    X_train_vec = vectorizer.transform(X_train.fillna(""))
    X_test_vec = vectorizer.transform(X_test.fillna(""))

    train_metrics = compute_metrics(y_train, model.predict(X_train_vec))
    test_metrics = compute_metrics(y_test, model.predict(X_test_vec))

    print_metrics("Train metrics", train_metrics)
    print_metrics("Test metrics", test_metrics)

    overfit_result = check_overfitting(train_metrics, test_metrics)
    print("\nTrain/test gaps (percentage points):")
    for name, gap in overfit_result["gaps"].items():
        print(f"  {name}: {gap * 100:+.2f}")

    if overfit_result["is_overfit"]:
        print(
            f"\nVERDICT: OVERFIT - gap exceeds 5 points on: "
            f"{', '.join(overfit_result['flagged_metrics'])}"
        )
    else:
        print("\nVERDICT: not overfit - all train/test gaps within 5 points")


if __name__ == "__main__":
    main()
