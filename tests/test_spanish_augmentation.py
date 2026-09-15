import pandas as pd

from src.data.spanish_augmentation import (
    build_spanish_translations,
    select_translation_candidates,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "comment": [
                "stop being such a fag",
                "not hateful at all",
                "hola como estas",
                "get out of my country",
                "already covered category",
            ],
            "language": ["en", "en", "es", "en", "en"],
            "label": [1, 0, 0, 1, 1],
            "categories": ["homophobia", "other", "other", "racism", "misogyny"],
            "source": ["ethos", "ethos", "haternet", "hatexplain", "ethos"],
        }
    )


def test_select_translation_candidates_filters_english_hate_in_target_categories():
    result = select_translation_candidates(_sample_df(), categories={"homophobia", "racism"})

    assert list(result["comment"]) == ["stop being such a fag", "get out of my country"]


def test_select_translation_candidates_excludes_non_hate_and_non_english():
    result = select_translation_candidates(_sample_df(), categories={"homophobia"})

    assert "not hateful at all" not in result["comment"].tolist()
    assert "hola como estas" not in result["comment"].tolist()


def test_build_spanish_translations_marks_language_and_source():
    fake_translate = lambda texts: [f"[es]{t}" for t in texts]  # noqa: E731

    result = build_spanish_translations(_sample_df(), fake_translate, categories={"homophobia"})

    assert len(result) == 1
    row = result.iloc[0]
    assert row["language"] == "es"
    assert row["comment"] == "[es]stop being such a fag"
    assert row["categories"] == "homophobia"
    assert row["source"] == "ethos_es_mt"


def test_build_spanish_translations_preserves_schema_columns():
    fake_translate = lambda texts: texts  # noqa: E731

    result = build_spanish_translations(_sample_df(), fake_translate, categories={"racism"})

    assert list(result.columns) == ["comment", "language", "label", "categories", "source"]
