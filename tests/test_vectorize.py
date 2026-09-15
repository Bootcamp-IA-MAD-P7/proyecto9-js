import pandas as pd
from scipy.sparse import csr_matrix

from src.features.vectorize import (
    build_bow_vectorizer,
    build_tfidf_vectorizer,
    fit_transform_corpus,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clean_comment": [
                "odi inmigr",
                "odi inmigr pais",
                "buen dia amig",
                "hate speech bad",
                "hate speech word",
                "nice day friend",
            ]
        }
    )


def test_fit_transform_corpus_returns_sparse_matrix_and_fitted_vectorizer():
    matrix, vectorizer = fit_transform_corpus(_sample_df(), build_tfidf_vectorizer(min_df=1))

    assert isinstance(matrix, csr_matrix)
    assert matrix.shape[0] == len(_sample_df())
    assert hasattr(vectorizer, "vocabulary_")


def test_tfidf_and_bow_share_the_same_vocabulary_for_equal_min_df():
    tfidf_matrix, tfidf_vec = fit_transform_corpus(_sample_df(), build_tfidf_vectorizer(min_df=1))
    bow_matrix, bow_vec = fit_transform_corpus(_sample_df(), build_bow_vectorizer(min_df=1))

    assert tfidf_vec.vocabulary_ == bow_vec.vocabulary_
    assert tfidf_matrix.shape == bow_matrix.shape


def test_bow_matrix_holds_raw_token_counts():
    df = pd.DataFrame({"clean_comment": ["odi odi inmigr"]})
    matrix, vectorizer = fit_transform_corpus(df, build_bow_vectorizer(min_df=1))

    odi_index = vectorizer.vocabulary_["odi"]
    assert matrix[0, odi_index] == 2


def test_fitted_vectorizer_transform_does_not_refit():
    _, vectorizer = fit_transform_corpus(_sample_df(), build_tfidf_vectorizer(min_df=1))
    vocab_before = dict(vectorizer.vocabulary_)

    new_matrix = vectorizer.transform(["a completely unseen sentence never trained on"])

    assert vectorizer.vocabulary_ == vocab_before
    assert new_matrix.shape[1] == len(vocab_before)


def test_min_df_prunes_singleton_tokens():
    df = pd.DataFrame(
        {
            "clean_comment": [
                "common word",
                "common word",
                "common word",
                "rare_singleton_token",
            ]
        }
    )
    _, vectorizer = fit_transform_corpus(df, build_tfidf_vectorizer(min_df=2))

    assert "common" in vectorizer.vocabulary_
    assert "rare_singleton_token" not in vectorizer.vocabulary_
