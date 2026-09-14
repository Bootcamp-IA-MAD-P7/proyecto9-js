"""Loading and basic exploration utilities for the YouTube comments dataset."""
import pandas as pd

REQUIRED_COLUMNS = ["comment", "label"]

# Known source dataset columns mapped to this project's normalized schema
# (comment/label), so different raw formats can share the same loader.
COLUMN_ALIASES = {
    "text": "comment",  # youtoxic_english_1000.csv (briefing dataset)
    "ishatespeech": "label",  # youtoxic_english_1000.csv (briefing dataset)
}


def load_comments(path: str) -> pd.DataFrame:
    """Load a raw comments CSV into a DataFrame with consistent column names.

    Implements the EARS criterion in specs/001-hate-speech-detection/spec.md:
    "WHEN a raw comments dataset is provided, THE SYSTEM SHALL produce a
    cleaned, preprocessed dataset ready for vectorization" (loading step).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=COLUMN_ALIASES)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["label"] = df["label"].astype(int)
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return basic exploration stats: size, duplicates, nulls, class balance."""
    return {
        "n_rows": len(df),
        "n_duplicates": int(df.duplicated(subset=["comment"]).sum()),
        "n_nulls": int(df["comment"].isna().sum()),
        "label_counts": df["label"].value_counts().to_dict(),
    }
