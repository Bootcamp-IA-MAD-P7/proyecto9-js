"""Stemming, aware of the dataset's `language` column ("en"/"es"). Uses
NLTK's Snowball stemmer, which needs no downloaded corpora (pure
algorithm), unlike stopword lists.

Implements the EARS criterion in specs/001-hate-speech-detection/spec.md
(section 2): "WHEN cleaned text is tokenized, THE SYSTEM SHALL return a
list of lemmatized/stemmed tokens excluding stopwords."
"""
from nltk.stem.snowball import SnowballStemmer

from src.preprocessing.tokenize import LANGUAGE_NAMES

_stemmer_cache: dict[str, SnowballStemmer] = {}


def _stemmer(language: str) -> SnowballStemmer:
    nltk_language = LANGUAGE_NAMES.get(language, "english")
    if nltk_language not in _stemmer_cache:
        _stemmer_cache[nltk_language] = SnowballStemmer(nltk_language)
    return _stemmer_cache[nltk_language]


def stem_tokens(tokens: list[str], language: str) -> list[str]:
    """Stem each token using a language-appropriate Snowball stemmer."""
    stemmer = _stemmer(language)
    return [stemmer.stem(t) for t in tokens]
