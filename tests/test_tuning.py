import optuna
import pandas as pd

from src.features.vectorize import build_tfidf_vectorizer, fit_transform_corpus
from src.models.tuning import objective, suggest_hyperparameters


def test_suggest_hyperparameters_returns_expected_keys_and_values():
    trial = optuna.trial.FixedTrial({"C": 1.0, "penalty": "l2", "class_weight": None})

    params = suggest_hyperparameters(trial)

    assert set(params.keys()) == {"C", "penalty", "class_weight"}
    assert params["C"] == 1.0
    assert params["penalty"] in ("l1", "l2")
    assert params["class_weight"] in (None, "balanced")


def test_objective_returns_a_valid_f1_score():
    df = pd.DataFrame(
        {
            "clean_comment": [
                "hate you disgust",
                "hate you all",
                "nice day friend",
                "have a nice day",
                "wonderful people",
                "you are disgust",
                "hate speech bad",
                "kind gentle soul",
                "awful hateful word",
                "great lovely time",
            ]
        }
    )
    labels = [1, 1, 0, 0, 0, 1, 1, 0, 1, 0]
    matrix, _ = fit_transform_corpus(df, build_tfidf_vectorizer(min_df=1))

    trial = optuna.trial.FixedTrial({"C": 1.0, "penalty": "l2", "class_weight": "balanced"})
    score = objective(trial, matrix, labels, cv=2)

    assert 0.0 <= score <= 1.0
