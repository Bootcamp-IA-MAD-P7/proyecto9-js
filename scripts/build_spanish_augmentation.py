"""Augment the enriched dataset with machine-translated Spanish rows for
categories that had zero or near-zero native Spanish coverage: homophobia,
racism, violence, religion, disability, transphobia, classism.

Requires data/processed/enriched_comments.csv (see build_enriched_dataset.py)
and the heavy deps in requirements-nlp.txt. Translation runs locally via
Helsinki-NLP/opus-mt-en-es (no rate limits, but expect roughly the same
order of magnitude as the sarcasm dataset run, ~80-90 minutes on CPU for
~27,000 comments).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.spanish_augmentation import build_spanish_translations  # noqa: E402
from src.data.translate import translate_texts  # noqa: E402

ENRICHED_PATH = Path("data/processed/enriched_comments.csv")
OUTPUT_PATH = Path("data/processed/enriched_comments_es_augmented.csv")


def main() -> None:
    enriched = pd.read_csv(ENRICHED_PATH)
    print(f"Loaded {len(enriched)} rows from {ENRICHED_PATH}")

    spanish = build_spanish_translations(enriched, translate_fn=translate_texts)
    print(f"Translated {len(spanish)} rows to Spanish")

    combined = pd.concat([enriched, spanish], ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(combined)} rows to {OUTPUT_PATH}")
    print(combined["language"].value_counts())
    print(spanish["categories"].value_counts())


if __name__ == "__main__":
    main()
