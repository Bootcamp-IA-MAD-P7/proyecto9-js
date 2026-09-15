"""Train and compare classic baseline classifiers (Logistic Regression,
Linear SVM, Multinomial Naive Bayes) on TF-IDF features, picking the
best-performing one by F1-score on the hateful class — see
specs/050-baseline-model-training/.
"""
from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def split_dataset(
    df: pd.DataFrame,
    text_column: str = "clean_comment",
    label_column: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified train/test split on `label_column`, so both splits keep
    the same hateful/not-hateful class balance as the full dataset."""
    return train_test_split(
        df[text_column],
        df[label_column],
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_column],
    )


def build_models() -> dict[str, Any]:
    """Return the three untrained baseline estimators. `class_weight`
    is balanced on the two models that support it, to counter the
    ~63/37 not-hate/hate class imbalance; MultinomialNB has no such
    parameter and is evaluated as-is."""
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=42),
        "naive_bayes": MultinomialNB(),
    }


def train_and_compare(
    models: dict[str, Any],
    X_train: csr_matrix,
    y_train: pd.Series,
    X_test: csr_matrix,
    y_test: pd.Series,
) -> tuple[dict[str, dict[str, float]], str, Any]:
    """Fit every model in `models` and score it on the held-out test
    split. Returns (per-model metrics, winning model name, winning
    fitted model) — the winner is whichever model scores highest on
    F1 for the hateful class (label == 1), not raw accuracy, since the
    dataset is imbalanced."""
    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1_hateful": f1_score(y_test, predictions, pos_label=1),
        }
        fitted[name] = model

    winner_name = max(results, key=lambda name: results[name]["f1_hateful"])
    return results, winner_name, fitted[winner_name]
