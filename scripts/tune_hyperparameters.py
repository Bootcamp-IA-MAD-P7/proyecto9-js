"""Optuna hyperparameter search for the served Logistic Regression
baseline: cross-validate candidates on the training split only, then
retrain and evaluate the best one exactly once on the held-out test
split. See specs/055-optuna-hyperparameter-tuning/.

Requires data/processed/enriched_comments_es_augmented_preprocessed.csv.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.evaluation.evaluate import compute_metrics  # noqa: E402
from src.features.vectorize import build_tfidf_vectorizer  # noqa: E402
from src.models.train import split_dataset  # noqa: E402
from src.models.tuning import objective  # noqa: E402

INPUT_PATH = Path("data/processed/enriched_comments_es_augmented_preprocessed.csv")
MODEL_OUTPUT_PATH = Path("data/processed/tuned_model.joblib")
VECTORIZER_OUTPUT_PATH = Path("data/processed/tuned_vectorizer.joblib")
N_TRIALS = 20

BASELINE_TEST_METRICS = {
    "accuracy": 0.7914,
    "precision": 0.7104,
    "recall": 0.7438,
    "f1": 0.7267,
}


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    X_train, X_test, y_train, y_test = split_dataset(df)

    vectorizer = build_tfidf_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train.fillna(""))
    X_test_vec = vectorizer.transform(X_test.fillna(""))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda trial: objective(trial, X_train_vec, y_train, cv=3),
        n_trials=N_TRIALS,
    )

    print(f"\nBest CV F1 (training split, {N_TRIALS} trials): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    final_model = LogisticRegression(
        solver="liblinear", random_state=42, **study.best_params
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        final_model.fit(X_train_vec, y_train)

    test_metrics = compute_metrics(y_test, final_model.predict(X_test_vec))
    print("\nTuned model test metrics:")
    for name in ("accuracy", "precision", "recall", "f1"):
        print(f"  {name}: {test_metrics[name]:.4f}")
    print(f"  confusion_matrix: {test_metrics['confusion_matrix'].tolist()}")

    print("\nTuned vs. baseline (test split):")
    for name in ("accuracy", "precision", "recall", "f1"):
        tuned_value = test_metrics[name]
        baseline_value = BASELINE_TEST_METRICS[name]
        delta = tuned_value - baseline_value
        verdict = "improved" if delta > 0 else ("regressed" if delta < 0 else "tied")
        print(
            f"  {name}: tuned={tuned_value:.4f} baseline={baseline_value:.4f} "
            f"delta={delta:+.4f} ({verdict})"
        )

    joblib.dump(final_model, MODEL_OUTPUT_PATH)
    joblib.dump(vectorizer, VECTORIZER_OUTPUT_PATH)
    print(f"\nPersisted tuned model to {MODEL_OUTPUT_PATH}")
    print(f"Persisted matched vectorizer to {VECTORIZER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
