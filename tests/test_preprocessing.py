import pandas as pd

from src.preprocessing.clean import clean_text
from src.preprocessing.pipeline import preprocess, preprocess_dataframe
from src.preprocessing.stem import stem_tokens
from src.preprocessing.tokenize import remove_stopwords, tokenize


def test_clean_text_removes_urls_mentions_and_control_chars():
    raw = "Check this out @someuser!! http://example.com/x\t\x07 &amp; more"
    cleaned = clean_text(raw)

    assert "http" not in cleaned
    assert "@someuser" not in cleaned
    assert "\x07" not in cleaned
    assert "&amp;" not in cleaned
    assert cleaned == cleaned.lower()


def test_clean_text_removes_emoji():
    cleaned = clean_text("I love this video 😂😂🔥")
    assert "😂" not in cleaned
    assert "🔥" not in cleaned
    assert "love" in cleaned


def test_tokenize_splits_words_including_spanish_accents():
    assert tokenize("odio a los inmigrantes según él") == [
        "odio",
        "a",
        "los",
        "inmigrantes",
        "según",
        "él",
    ]


def test_remove_stopwords_filters_english_and_spanish():
    en_tokens = ["this", "is", "a", "hateful", "comment"]
    assert remove_stopwords(en_tokens, "en") == ["hateful", "comment"]

    es_tokens = ["esto", "es", "un", "comentario", "odioso"]
    assert remove_stopwords(es_tokens, "es") == ["comentario", "odioso"]


def test_stem_tokens_reduces_related_words_to_common_root():
    en_stems = stem_tokens(["running", "runs"], "en")
    assert en_stems[0] == en_stems[1]

    es_stems = stem_tokens(["corriendo", "corre"], "es")
    assert es_stems[0] == es_stems[1]


def test_preprocess_end_to_end_pipeline_english():
    result = preprocess("Check out http://x.com @user THIS IS SO HATEFUL!!", language="en")
    assert "http" not in result
    assert "@user" not in result
    assert "hate" in result  # stemmed form of "hateful"


def test_preprocess_end_to_end_pipeline_spanish():
    result = preprocess("Odio a los inmigrantes, que se vayan ya!", language="es")
    assert "odi" in result  # stemmed form of "odio"
    assert "inmigr" in result  # stemmed form of "inmigrantes"


def test_preprocess_dataframe_adds_clean_comment_column():
    df = pd.DataFrame(
        {
            "comment": ["I HATE this @user http://x.com", "Odio esto también"],
            "language": ["en", "es"],
        }
    )

    result = preprocess_dataframe(df)

    assert "clean_comment" in result.columns
    assert len(result) == 2
    assert "http" not in result.loc[0, "clean_comment"]
