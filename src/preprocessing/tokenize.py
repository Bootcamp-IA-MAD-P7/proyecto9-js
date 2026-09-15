"""Tokenization and stopword removal, aware of the dataset's two languages
(English and Spanish, from the `language` column produced by
src/data/loader.py and src/data/harmonize.py).

Implements the EARS criterion in specs/001-hate-speech-detection/spec.md
(section 2): tokenize cleaned text and exclude stopwords.
"""
import re

import nltk

TOKEN_RE = re.compile(r"[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]+")

LANGUAGE_NAMES = {"en": "english", "es": "spanish"}
_stopword_cache: dict[str, frozenset] = {}


def _stopwords(language: str) -> frozenset:
    nltk_language = LANGUAGE_NAMES.get(language, "english")
    if nltk_language not in _stopword_cache:
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        _stopword_cache[nltk_language] = frozenset(stopwords.words(nltk_language))
    return _stopword_cache[nltk_language]


def tokenize(text: str) -> list[str]:
    """Split cleaned text into word tokens (letters only, Unicode-aware for
    Spanish accented characters)."""
    return TOKEN_RE.findall(text)


def remove_stopwords(tokens: list[str], language: str) -> list[str]:
    """Drop tokens that are stopwords for the given language ("en"/"es")."""
    stop = _stopwords(language)
    return [t for t in tokens if t not in stop]
