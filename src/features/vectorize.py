"""Classic text vectorization (TF-IDF / Bag of Words) over the already
preprocessed `clean_comment` column (src/preprocessing/pipeline.py).

Both vectorizers share the same tokenizer and `min_df` vocabulary-pruning
threshold, so their feature spaces are directly comparable (same token
set, different cell values) — see specs/049-classic-text-vectorization/.
"""
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# clean_comment is already lowercased, stemmed, stopword-free tokens
# joined by whitespace, so a plain whitespace split is enough — no need
# for scikit-learn's default punctuation-aware token pattern.
TOKEN_PATTERN = r"\S+"


def build_tfidf_vectorizer(min_df: int = 5) -> TfidfVectorizer:
    """Build an unfitted TF-IDF vectorizer for the `clean_comment` corpus."""
    return TfidfVectorizer(token_pattern=TOKEN_PATTERN, min_df=min_df)


def build_bow_vectorizer(min_df: int = 5) -> CountVectorizer:
    """Build an unfitted Bag-of-Words vectorizer for the `clean_comment`
    corpus, sharing TF-IDF's tokenizer/min_df for a fair comparison."""
    return CountVectorizer(token_pattern=TOKEN_PATTERN, min_df=min_df)


def fit_transform_corpus(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer | CountVectorizer,
    text_column: str = "clean_comment",
) -> tuple[csr_matrix, TfidfVectorizer | CountVectorizer]:
    """Fit `vectorizer` on `df[text_column]` and return the resulting
    sparse feature matrix alongside the now-fitted vectorizer, so it can
    be persisted and reused unchanged at inference time."""
    matrix = vectorizer.fit_transform(df[text_column].fillna(""))
    return matrix, vectorizer
