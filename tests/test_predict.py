from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.models.predict import predict_label
from src.preprocessing.pipeline import preprocess


def _fake_model_and_vectorizer():
    # Fit on already-preprocessed text, matching how the real baseline
    # vectorizer was fit on clean_comment (specs/050-baseline-model-training).
    raw_corpus = ["hate you disgusting", "hate you all", "nice day friend", "have a nice day"]
    corpus = [preprocess(text, "en") for text in raw_corpus]
    labels = [1, 1, 0, 0]
    vectorizer = TfidfVectorizer(token_pattern=r"\S+", min_df=1)
    X = vectorizer.fit_transform(corpus)
    model = LogisticRegression()
    model.fit(X, labels)
    return model, vectorizer


def test_predict_label_returns_expected_shape():
    model, vectorizer = _fake_model_and_vectorizer()

    result = predict_label("hate you disgusting", model, vectorizer)

    assert result["label"] in (0, 1)
    assert result["prediction"] == ("hate" if result["label"] == 1 else "not_hate")
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_label_matches_underlying_model_prediction():
    model, vectorizer = _fake_model_and_vectorizer()
    text = "nice day friend"

    result = predict_label(text, model, vectorizer)

    expected_label = int(model.predict(vectorizer.transform([preprocess(text, "en")]))[0])
    assert result["label"] == expected_label


class _NoProbaEstimator:
    """Minimal stand-in for an estimator with no predict_proba (e.g. LinearSVC)."""

    def predict(self, X):
        return [1] * X.shape[0]


def test_predict_label_handles_model_without_predict_proba():
    _, vectorizer = _fake_model_and_vectorizer()
    model = _NoProbaEstimator()

    result = predict_label("hate you disgusting", model, vectorizer)

    assert result["label"] == 1
    assert result["prediction"] == "hate"
    assert result["probability"] is None
