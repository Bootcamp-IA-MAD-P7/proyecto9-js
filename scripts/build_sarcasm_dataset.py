"""Build the auxiliary sarcasm dataset (English + machine-translated Spanish).

Requires data/raw/external/sarcasm_headlines_en.json (see README.md) and
the heavy deps in requirements-nlp.txt (`pip install -r requirements-nlp.txt`).
Translation runs locally via Helsinki-NLP/opus-mt-en-es (no rate limits,
but ~80-90 minutes on CPU for the full ~28,600 headlines).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.sarcasm import build_spanish_sarcasm, load_sarcasm_en  # noqa: E402
from src.data.translate import translate_texts  # noqa: E402

EN_PATH = Path("data/raw/external/sarcasm_headlines_en.json")
OUTPUT_PATH = Path("data/processed/sarcasm_comments.csv")


def main() -> None:
    english = load_sarcasm_en(str(EN_PATH))
    print(f"Loaded {len(english)} English headlines")

    spanish = build_spanish_sarcasm(english, translate_fn=translate_texts)
    print(f"Translated {len(spanish)} headlines to Spanish")

    combined = pd.concat([english, spanish], ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(combined)} rows to {OUTPUT_PATH}")
    print(combined["language"].value_counts())
    print(combined["is_sarcastic"].value_counts())


if __name__ == "__main__":
    main()
