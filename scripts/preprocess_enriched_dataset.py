"""Apply the text preprocessing pipeline (src/preprocessing/pipeline.py) to
the enriched dataset, adding a `clean_comment` column for downstream
vectorization/modeling tasks and for the EDA notebook.

Requires data/processed/enriched_comments.csv (see
scripts/build_enriched_dataset.py), or another harmonized CSV passed via
--input (e.g. the Spanish-augmented dataset from
scripts/build_spanish_augmentation.py).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.preprocessing.pipeline import preprocess_dataframe  # noqa: E402

DEFAULT_INPUT_PATH = Path("data/processed/enriched_comments.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/enriched_comments_preprocessed.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    t0 = time.time()
    result = preprocess_dataframe(df)
    print(f"Preprocessed {len(result)} rows in {time.time() - t0:.1f}s")

    result.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
