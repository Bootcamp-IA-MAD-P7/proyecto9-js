"""Apply the text preprocessing pipeline (src/preprocessing/pipeline.py) to
the enriched dataset, adding a `clean_comment` column for downstream
vectorization/modeling tasks and for the EDA notebook.

Requires data/processed/enriched_comments.csv (see
scripts/build_enriched_dataset.py).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.preprocessing.pipeline import preprocess_dataframe  # noqa: E402

INPUT_PATH = Path("data/processed/enriched_comments.csv")
OUTPUT_PATH = Path("data/processed/enriched_comments_preprocessed.csv")


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    t0 = time.time()
    result = preprocess_dataframe(df)
    print(f"Preprocessed {len(result)} rows in {time.time() - t0:.1f}s")

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
