"""Machine-translate English hateful comments into Spanish for categories
that have zero or near-zero native Spanish coverage in the enriched dataset
(homophobia, racism, violence, religion, disability, transphobia, classism).

Reuses the same local MarianMT pipeline built for the sarcasm dataset
(src/data/translate.py) via an injectable `translate_fn`, so tests can pass
a fake translator instead of requiring torch/transformers.

Implements the EARS criteria in specs/048-spanish-category-augmentation/spec.md.
"""
from collections.abc import Callable

import pandas as pd

from src.data.harmonize import SCHEMA_COLUMNS

ORPHAN_CATEGORIES = {
    "homophobia",
    "racism",
    "violence",
    "religion",
    "disability",
    "transphobia",
    "classism",
}


def select_translation_candidates(
    df: pd.DataFrame, categories: set[str] = ORPHAN_CATEGORIES
) -> pd.DataFrame:
    """Return the English hateful rows whose categories intersect `categories`
    (a row's `categories` field may hold several "|"-joined tags)."""
    is_english_hate = (df["language"] == "en") & (df["label"] == 1)
    tags = df["categories"].str.split("|")
    matches = tags.apply(lambda row_tags: bool(categories.intersection(row_tags)))
    return df[is_english_hate & matches]


def build_spanish_translations(
    df: pd.DataFrame,
    translate_fn: Callable[[list[str]], list[str]],
    categories: set[str] = ORPHAN_CATEGORIES,
) -> pd.DataFrame:
    """Translate the selected English rows to Spanish, keeping their label
    and categories but marking `source` as machine-translated so this
    synthetic data stays distinguishable from native Spanish rows."""
    candidates = select_translation_candidates(df, categories)
    translated = pd.DataFrame(
        {
            "comment": translate_fn(candidates["comment"].tolist()),
            "language": "es",
            "label": candidates["label"].to_numpy(),
            "categories": candidates["categories"].to_numpy(),
            "source": candidates["source"].astype(str) + "_es_mt",
        }
    )
    return translated[SCHEMA_COLUMNS]
