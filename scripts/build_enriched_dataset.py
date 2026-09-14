"""Build the enriched, multi-category hate-speech dataset.

Combines the briefing dataset with HateXplain, ETHOS, Measuring Hate Speech,
and HatEval into data/processed/enriched_comments.csv. Requires the source
files to already be in data/raw/ and data/raw/external/ (see README.md for
download instructions).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.harmonize import (  # noqa: E402
    combine_datasets,
    load_ethos,
    load_haternet,
    load_hateval,
    load_hatexplain,
    load_measuring_hate_speech,
    load_offendes,
)
from src.data.loader import load_comments  # noqa: E402

RAW_DIR = Path("data/raw")
EXTERNAL_DIR = RAW_DIR / "external"
OUTPUT_PATH = Path("data/processed/enriched_comments.csv")


def main() -> None:
    youtoxic = load_comments(str(RAW_DIR / "youtoxic_english_1000.csv"))
    youtoxic["language"] = "en"
    youtoxic["categories"] = "other"
    youtoxic["source"] = "youtoxic"
    youtoxic = youtoxic[["comment", "language", "label", "categories", "source"]]

    hatexplain = load_hatexplain(str(EXTERNAL_DIR / "hatexplain_dataset.json"))
    ethos = load_ethos(
        str(EXTERNAL_DIR / "ethos_binary.csv"),
        str(EXTERNAL_DIR / "ethos_multilabel.csv"),
    )
    measuring = load_measuring_hate_speech(str(EXTERNAL_DIR / "measuring_hate_speech.parquet"))
    hateval = load_hateval(
        str(EXTERNAL_DIR / "hateval" / "train.parquet"),
        str(EXTERNAL_DIR / "hateval" / "dev.parquet"),
        str(EXTERNAL_DIR / "hateval" / "test.parquet"),
    )
    haternet = load_haternet(str(EXTERNAL_DIR / "haternet_labeled_corpus_6k.txt"))
    offendes = load_offendes(
        str(EXTERNAL_DIR / "offendes" / "training_set.tsv"),
        str(EXTERNAL_DIR / "offendes" / "dev_set.tsv"),
        str(EXTERNAL_DIR / "offendes" / "test_set.tsv"),
    )

    combined = combine_datasets(
        youtoxic, hatexplain, ethos, measuring, hateval, haternet, offendes
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(combined)} rows to {OUTPUT_PATH}")
    print(combined["source"].value_counts())
    print(combined["language"].value_counts())
    print(combined["label"].value_counts())


if __name__ == "__main__":
    main()
