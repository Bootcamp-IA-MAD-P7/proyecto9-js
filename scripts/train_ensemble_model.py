"""Train a hard-voting ensemble over the three baseline algorithm types
and compare it against the persisted single-model baseline
(specs/050-baseline-model-training/spec.md,
specs/051-model-evaluation/spec.md) on the identical held-out test
split.

Requires data/processed/enriched_comments_es_augmented_preprocessed.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.evaluate import check_overfitting, compute_metrics  # noqa: E402
from src.features.vectorize import build_tfidf_vectorizer  # noqa: E402
from src.models.ensemble import build_ensemble  # noqa: E402
from src.models.train import split_dataset  # noqa: E402

INPUT_PATH = Path("data/processed/enriched_comments_es_augmented_preprocessed.csv")
MODEL_OUTPUT_PATH = Path("data/processed/ensemble_model.joblib")
VECTORIZER_OUTPUT_PATH = Path("data/processed/ensemble_vectorizer.joblib")

# From specs/051-model-evaluation/spec.md's Result section (test split).
BASELINE_TEST_METRICS = {
    "accuracy": 0.7914,
    "precision": 0.7104,
    "recall": 0.7438,
    "f1": 0.7267,
}


def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n{label}:")
    for name in ("accuracy", "precision", "recall", "f1"):
        print(f"  {name}: {metrics[name]:.4f}")
    print("  confusion_matrix (rows=true, cols=pred, [[TN,FP],[FN,TP]]):")
    print(f"  {metrics['confusion_matrix'].tolist()}")


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    X_train, X_test, y_train, y_test = split_dataset(df)

    vectorizer = build_tfidf_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train.fillna(""))
    X_test_vec = vectorizer.transform(X_test.fillna(""))

    ensemble = build_ensemble()
    ensemble.fit(X_train_vec, y_train)

    train_metrics = compute_metrics(y_train, ensemble.predict(X_train_vec))
    test_metrics = compute_metrics(y_test, ensemble.predict(X_test_vec))

    print_metrics("Ensemble train metrics", train_metrics)
    print_metrics("Ensemble test metrics", test_metrics)

    overfit_result = check_overfitting(train_metrics, test_metrics)
    if overfit_result["is_overfit"]:
        print(f"\nOverfit gap exceeds 5 points on: {overfit_result['flagged_metrics']}")
    else:
        print("\nNot overfit - all train/test gaps within 5 points")

    print("\nEnsemble vs. persisted baseline (test split):")
    for name in ("accuracy", "precision", "recall", "f1"):
        ensemble_value = test_metrics[name]
        baseline_value = BASELINE_TEST_METRICS[name]
        delta = ensemble_value - baseline_value
        verdict = "improved" if delta > 0 else ("regressed" if delta < 0 else "tied")
        print(
            f"  {name}: ensemble={ensemble_value:.4f} baseline={baseline_value:.4f} "
            f"delta={delta:+.4f} ({verdict})"
        )

    joblib.dump(ensemble, MODEL_OUTPUT_PATH)
    joblib.dump(vectorizer, VECTORIZER_OUTPUT_PATH)
    print(f"\nPersisted ensemble to {MODEL_OUTPUT_PATH}")
    print(f"Persisted matched vectorizer to {VECTORIZER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
