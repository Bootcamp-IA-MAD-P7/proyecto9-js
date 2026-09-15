from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.features.vectorize import build_tfidf_vectorizer, fit_transform_corpus
from src.models.ensemble import build_ensemble


def test_build_ensemble_wraps_the_three_baseline_estimator_types():
    ensemble = build_ensemble()

    assert isinstance(ensemble, VotingClassifier)
    assert ensemble.voting == "hard"
    estimator_types = {type(model) for _, model in ensemble.estimators}
    assert estimator_types == {LogisticRegression, LinearSVC, MultinomialNB}


def test_ensemble_fits_and_predicts_on_a_small_corpus():
    import pandas as pd

    df = pd.DataFrame(
        {
            "clean_comment": [
                "hate you disgust",
                "hate you all",
                "nice day friend",
                "have a nice day",
                "wonderful people",
                "you are disgust",
            ]
        }
    )
    labels = [1, 1, 0, 0, 0, 1]
    matrix, _ = fit_transform_corpus(df, build_tfidf_vectorizer(min_df=1))

    ensemble = build_ensemble()
    ensemble.fit(matrix, labels)
    predictions = ensemble.predict(matrix)

    assert len(predictions) == len(labels)
    assert set(predictions).issubset({0, 1})
