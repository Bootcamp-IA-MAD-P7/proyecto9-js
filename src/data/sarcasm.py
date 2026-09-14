"""Load and build the auxiliary sarcasm dataset (English + machine-translated
Spanish). Kept separate from src/data/harmonize.py's hate-speech schema:
sarcasm is a different annotation dimension over a different corpus (news
headlines, not the YouTube/Twitter comments in the hate datasets), so rows
cannot be merged 1:1 with the hate dataset. See
specs/005-sarcasm-dataset/spec.md.
"""
import json
from collections.abc import Callable

import pandas as pd

SARCASM_SCHEMA_COLUMNS = ["comment", "language", "is_sarcastic", "source"]


def load_sarcasm_en(path: str) -> pd.DataFrame:
    """Load the News Headlines Dataset For Sarcasm Detection (JSON Lines:
    one {is_sarcastic, headline, article_link} object per line)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.append(
                {
                    "comment": record["headline"],
                    "language": "en",
                    "is_sarcastic": int(record["is_sarcastic"]),
                    "source": "sarcasm_headlines",
                }
            )

    return pd.DataFrame(rows, columns=SARCASM_SCHEMA_COLUMNS)


def build_spanish_sarcasm(
    english_df: pd.DataFrame, translate_fn: Callable[[list[str]], list[str]]
) -> pd.DataFrame:
    """Machine-translate an English sarcasm DataFrame's comments to Spanish,
    keeping the same is_sarcastic labels. `translate_fn` is injected so
    tests don't need the real (heavy) translation model — see
    src/data/translate.py::translate_texts for the real implementation."""
    translated = english_df.copy()
    translated["comment"] = translate_fn(english_df["comment"].tolist())
    translated["language"] = "es"
    translated["source"] = english_df["source"] + "_es_mt"
    return translated[SARCASM_SCHEMA_COLUMNS]
