"""Train and compare three classic baseline classifiers (Logistic
Regression, Linear SVM, Multinomial Naive Bayes) on the preprocessed,
Spanish-augmented corpus, and persist the best-performing one alongside
its matched, training-only-fit TF-IDF vectorizer.

Requires data/processed/enriched_comments_es_augmented_preprocessed.csv
(see scripts/preprocess_enriched_dataset.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src.features.vectorize import build_tfidf_vectorizer  # noqa: E402
from src.models.train import build_models, split_dataset, train_and_compare  # noqa: E402

INPUT_PATH = Path("data/processed/enriched_comments_es_augmented_preprocessed.csv")
MODEL_OUTPUT_PATH = Path("data/processed/baseline_model.joblib")
VECTORIZER_OUTPUT_PATH = Path("data/processed/baseline_vectorizer.joblib")


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    X_train, X_test, y_train, y_test = split_dataset(df)
    print(f"Train: {len(X_train)} rows ({y_train.mean():.1%} hateful)")
    print(f"Test:  {len(X_test)} rows ({y_test.mean():.1%} hateful)")

    # Fit fresh on the training split only — never reuse
    # data/processed/tfidf_vectorizer.joblib here, which was fit on the
    # full corpus for spec 049's feature-space comparison and would leak
    # test-set vocabulary/IDF statistics into training.
    vectorizer = build_tfidf_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train.fillna(""))
    X_test_vec = vectorizer.transform(X_test.fillna(""))

    results, winner_name, winner_model = train_and_compare(
        build_models(), X_train_vec, y_train, X_test_vec, y_test
    )

    print("\nModel comparison (held-out test split):")
    for name, metrics in results.items():
        marker = " <- winner" if name == winner_name else ""
        print(
            f"  {name}: accuracy={metrics['accuracy']:.4f}, "
            f"f1_hateful={metrics['f1_hateful']:.4f}{marker}"
        )

    joblib.dump(winner_model, MODEL_OUTPUT_PATH)
    joblib.dump(vectorizer, VECTORIZER_OUTPUT_PATH)
    print(f"\nPersisted winning model ({winner_name}) to {MODEL_OUTPUT_PATH}")
    print(f"Persisted matched vectorizer to {VECTORIZER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
