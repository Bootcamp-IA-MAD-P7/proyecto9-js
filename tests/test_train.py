import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.models.train import build_models, split_dataset, train_and_compare


def _sample_df(n_per_class: int = 15) -> pd.DataFrame:
    hate_texts = [f"hate slur insult bad_{i}" for i in range(n_per_class)]
    not_hate_texts = [f"nice friendly kind good_{i}" for i in range(n_per_class * 2)]
    comments = hate_texts + not_hate_texts
    labels = [1] * len(hate_texts) + [0] * len(not_hate_texts)
    return pd.DataFrame({"clean_comment": comments, "label": labels})


def test_split_dataset_preserves_class_balance():
    df = _sample_df()
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.2, random_state=42)

    full_ratio = df["label"].mean()
    train_ratio = np.mean(y_train)
    test_ratio = np.mean(y_test)

    assert abs(train_ratio - full_ratio) < 0.05
    assert abs(test_ratio - full_ratio) < 0.1
    assert len(X_train) + len(X_test) == len(df)


def test_split_dataset_is_reproducible_with_same_random_state():
    df = _sample_df()
    split_a = split_dataset(df, test_size=0.2, random_state=42)
    split_b = split_dataset(df, test_size=0.2, random_state=42)

    assert list(split_a[0]) == list(split_b[0])
    assert list(split_a[2]) == list(split_b[2])


def test_build_models_returns_three_distinct_unfitted_estimators():
    models = build_models()

    assert set(models.keys()) == {"logistic_regression", "linear_svm", "naive_bayes"}
    assert isinstance(models["logistic_regression"], LogisticRegression)
    assert isinstance(models["linear_svm"], LinearSVC)
    assert isinstance(models["naive_bayes"], MultinomialNB)


def test_train_and_compare_picks_the_highest_f1_model():
    df = _sample_df(n_per_class=20)
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.25, random_state=42)

    vectorizer = TfidfVectorizer(token_pattern=r"\S+", min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results, winner_name, winner_model = train_and_compare(
        build_models(), X_train_vec, y_train, X_test_vec, y_test
    )

    assert set(results.keys()) == {"logistic_regression", "linear_svm", "naive_bayes"}
    for metrics in results.values():
        assert "accuracy" in metrics
        assert "f1_hateful" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_hateful"] <= 1.0

    best_by_f1 = max(results, key=lambda name: results[name]["f1_hateful"])
    assert winner_name == best_by_f1
    assert winner_model is not None
