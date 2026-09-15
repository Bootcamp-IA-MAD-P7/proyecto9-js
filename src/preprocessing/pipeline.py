"""End-to-end text preprocessing pipeline: clean -> tokenize -> remove
stopwords -> stem. Combines src/preprocessing/{clean,tokenize,stem}.py
into the single entry point later tasks (vectorization, modeling) build on.
"""
import pandas as pd

from src.preprocessing.clean import clean_text
from src.preprocessing.stem import stem_tokens
from src.preprocessing.tokenize import remove_stopwords, tokenize


def preprocess(text: str, language: str = "en") -> str:
    """Run the full pipeline on one comment, returning the processed text
    as a whitespace-joined string of stemmed, stopword-free tokens."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens, language)
    tokens = stem_tokens(tokens, language)
    return " ".join(tokens)


def preprocess_dataframe(
    df: pd.DataFrame, text_column: str = "comment", language_column: str = "language"
) -> pd.DataFrame:
    """Add a `clean_comment` column, preprocessing each row with its own
    language (the dataset is bilingual: English and Spanish)."""
    df = df.copy()
    df["clean_comment"] = [
        preprocess(text, lang) for text, lang in zip(df[text_column], df[language_column])
    ]
    return df
