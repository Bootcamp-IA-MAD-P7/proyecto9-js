"""Optuna hyperparameter search for the served Logistic Regression
baseline — see specs/055-optuna-hyperparameter-tuning/.
"""
import warnings
from typing import Any

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def suggest_hyperparameters(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest the three knobs most likely to move a linear TF-IDF
    baseline: regularization strength, penalty type, and class
    weighting — see specs/055-optuna-hyperparameter-tuning/research.md
    for why these three and not others."""
    return {
        "C": trial.suggest_float("C", 0.01, 100, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def objective(trial: optuna.Trial, X_train, y_train, cv: int = 3) -> float:
    """Score one trial via stratified cross-validation on the training
    split only (never the held-out test split), returning mean F1 on
    the hateful class for Optuna to maximize."""
    params = suggest_hyperparameters(trial)
    y_train = np.asarray(y_train)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = []
    for train_idx, val_idx in splitter.split(X_train, y_train):
        model = LogisticRegression(solver="liblinear", random_state=42, **params)
        with warnings.catch_warnings():
            # `penalty` still works correctly on scikit-learn 1.8+ (verified:
            # produces the same coefficients as the replacement `l1_ratio`
            # API) but emits deprecation noise on every fit — silence it.
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            model.fit(X_train[train_idx], y_train[train_idx])
        predictions = model.predict(X_train[val_idx])
        scores.append(f1_score(y_train[val_idx], predictions, pos_label=1))

    return float(np.mean(scores))
