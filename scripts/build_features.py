"""Vectorize the preprocessed, Spanish-augmented corpus with TF-IDF and
Bag of Words, report feature-space size/sparsity for each, and persist
the chosen TF-IDF baseline vectorizer for reuse at inference time.

Requires data/processed/enriched_comments_es_augmented_preprocessed.csv
(see scripts/preprocess_enriched_dataset.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src.features.vectorize import (  # noqa: E402
    build_bow_vectorizer,
    build_tfidf_vectorizer,
    fit_transform_corpus,
)

INPUT_PATH = Path("data/processed/enriched_comments_es_augmented_preprocessed.csv")
VECTORIZER_OUTPUT_PATH = Path("data/processed/tfidf_vectorizer.joblib")


def report(name: str, matrix) -> None:
    rows, vocab_size = matrix.shape
    nnz = matrix.nnz
    sparsity = 1 - nnz / (rows * vocab_size)
    print(
        f"{name}: vocab_size={vocab_size}, nnz={nnz}, "
        f"sparsity={sparsity:.4%}, matrix_shape={matrix.shape}"
    )


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    tfidf_matrix, tfidf_vectorizer = fit_transform_corpus(df, build_tfidf_vectorizer())
    report("tfidf", tfidf_matrix)

    bow_matrix, _ = fit_transform_corpus(df, build_bow_vectorizer())
    report("bow", bow_matrix)

    joblib.dump(tfidf_vectorizer, VECTORIZER_OUTPUT_PATH)
    print(f"Wrote baseline TF-IDF vectorizer to {VECTORIZER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
