"""Hard-voting ensemble over the three baseline algorithm types
(src/models/train.py::build_models) — see specs/054-ensemble-model/.
"""
from sklearn.ensemble import VotingClassifier

from src.models.train import build_models


def build_ensemble() -> VotingClassifier:
    """Build an unfitted hard-voting classifier wrapping Logistic
    Regression, Linear SVM, and Multinomial Naive Bayes. Hard voting
    (majority label vote) is used instead of soft voting because
    LinearSVC has no predict_proba without extra calibration overhead
    — see specs/054-ensemble-model/research.md."""
    models = build_models()
    estimators = [(name, model) for name, model in models.items()]
    return VotingClassifier(estimators=estimators, voting="hard")
