"""Normalize and combine external hate-speech datasets into a shared schema.

Every loader below returns a DataFrame with the same columns:
    comment    : str  - the raw text
    language   : str  - "en" or "es"
    label      : int  - 1 if hateful, 0 otherwise
    categories : str  - "|"-joined tags from CATEGORY_VOCAB, or "other"
    source     : str  - short dataset identifier

Implements the EARS criteria in specs/004-dataset-enrichment/spec.md.
"""
import json

import pandas as pd

SCHEMA_COLUMNS = ["comment", "language", "label", "categories", "source"]

CATEGORY_VOCAB = {
    "racism",
    "xenophobia",
    "religion",
    "misogyny",
    "homophobia",
    "transphobia",
    "disability",
    "classism",
    "violence",
    "other",
}

# HateXplain annotators tag a "target" community per post; map each known
# target to this project's category vocabulary. Anything not listed here
# (including "None") falls back to "other".
HATEXPLAIN_TARGET_MAP = {
    "African": "racism",
    "Arab": "racism",
    "Caucasian": "racism",
    "Asian": "racism",
    "Hispanic": "racism",
    "Indigenous": "racism",
    "Indian": "racism",
    "Islam": "religion",
    "Jewish": "religion",
    "Christian": "religion",
    "Hindu": "religion",
    "Buddhism": "religion",
    "Women": "misogyny",
    "Homosexual": "homophobia",
    "Bisexual": "homophobia",
    "Refugee": "xenophobia",
    "Disability": "disability",
    "Economic": "classism",
}


def _majority_label(annotators: list[dict]) -> str:
    votes = [a["label"] for a in annotators]
    return max(set(votes), key=votes.count)


def _majority_categories(annotators: list[dict]) -> str:
    votes: dict[str, int] = {}
    for a in annotators:
        for target in a["target"]:
            votes[target] = votes.get(target, 0) + 1
    majority_targets = [t for t, count in votes.items() if count > len(annotators) / 2]
    mapped = {HATEXPLAIN_TARGET_MAP.get(t, "other") for t in majority_targets}
    if len(mapped) > 1:
        mapped.discard("other")
    return "|".join(sorted(mapped)) if mapped else "other"


def load_hatexplain(path: str) -> pd.DataFrame:
    """Load HateXplain's dataset.json, resolving each post's label and
    target categories by majority vote across its 3 annotators. Only a
    "hatespeech" majority counts as hate (see spec's open questions)."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for post in raw.values():
        is_hate = _majority_label(post["annotators"]) == "hatespeech"
        rows.append(
            {
                "comment": " ".join(post["post_tokens"]),
                "language": "en",
                "label": int(is_hate),
                "categories": _majority_categories(post["annotators"]) if is_hate else "other",
                "source": "hatexplain",
            }
        )

    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS)


# ETHOS's multi-label columns mapped to this project's category vocabulary.
ETHOS_CATEGORY_MAP = {
    "violence": "violence",
    "gender": "misogyny",
    "race": "racism",
    "national_origin": "xenophobia",
    "disability": "disability",
    "religion": "religion",
    "sexual_orientation": "homophobia",
}


def load_ethos(binary_path: str, multilabel_path: str, threshold: float = 0.5) -> pd.DataFrame:
    """Load ETHOS's binary + multi-label CSVs, binarizing continuous scores
    at `threshold` and joining categories onto the rows that have them."""
    binary = pd.read_csv(binary_path, sep=";")
    multilabel = pd.read_csv(multilabel_path, sep=";")

    category_columns = [c for c in ETHOS_CATEGORY_MAP if c in multilabel.columns]
    multilabel = multilabel.copy()

    def _ethos_categories(row):
        tags = {ETHOS_CATEGORY_MAP[c] for c in category_columns if row[c] >= threshold}
        return "|".join(sorted(tags)) or "other"

    multilabel["categories"] = multilabel[category_columns].apply(_ethos_categories, axis=1)

    merged = binary.merge(multilabel[["comment", "categories"]], on="comment", how="left")
    merged["categories"] = merged["categories"].fillna("other")
    merged["label"] = (merged["isHate"] >= threshold).astype(int)
    merged["language"] = "en"
    merged["source"] = "ethos"

    return merged[SCHEMA_COLUMNS]


# Measuring Hate Speech's per-annotation target_* flags mapped to this
# project's category vocabulary (aggregated per comment before mapping).
MEASURING_HATE_SPEECH_TARGET_MAP = {
    "target_race": "racism",
    "target_religion": "religion",
    "target_origin": "xenophobia",
    "target_gender_transgender_men": "transphobia",
    "target_gender_transgender_women": "transphobia",
    "target_gender_transgender_unspecified": "transphobia",
    "target_gender_women": "misogyny",
    "target_sexuality_gay": "homophobia",
    "target_sexuality_lesbian": "homophobia",
    "target_sexuality_bisexual": "homophobia",
    "target_disability": "disability",
    "violence": "violence",
}


def load_measuring_hate_speech(path: str, threshold: float = 0.5) -> pd.DataFrame:
    """Load Measuring Hate Speech's parquet, aggregating its per-annotator
    annotation rows into one row per comment (mean of numeric columns)."""
    df = pd.read_parquet(path)

    target_columns = [c for c in MEASURING_HATE_SPEECH_TARGET_MAP if c in df.columns]
    agg = df.groupby("comment_id").agg(
        comment=("text", "first"),
        hatespeech=("hatespeech", "mean"),
        **{c: (c, "mean") for c in target_columns},
    )

    agg["label"] = (agg["hatespeech"] >= 1).astype(int)

    def _measuring_categories(row):
        tags = {MEASURING_HATE_SPEECH_TARGET_MAP[c] for c in target_columns if row[c] >= threshold}
        return "|".join(sorted(tags)) or "other"

    agg["categories"] = agg[target_columns].apply(_measuring_categories, axis=1)
    agg["language"] = "en"
    agg["source"] = "measuring_hate_speech"

    return agg.reset_index()[SCHEMA_COLUMNS]


# HatEval's `target` column: "mig" (hate against migrants/immigrants) or
# "mis" (hate against women, i.e. misogyny).
HATEVAL_TARGET_MAP = {
    "mig": "xenophobia",
    "mis": "misogyny",
}


def load_hateval(train_path: str, dev_path: str, test_path: str) -> pd.DataFrame:
    """Load HatEval's train/dev/test parquet splits (English and Spanish
    tweets about hate towards immigrants and women) into the shared schema."""
    df = pd.concat(
        [pd.read_parquet(train_path), pd.read_parquet(dev_path), pd.read_parquet(test_path)],
        ignore_index=True,
    )

    df["label"] = df["HS"].astype(int)
    df["categories"] = df.apply(
        lambda row: HATEVAL_TARGET_MAP.get(row["target"], "other") if row["label"] else "other",
        axis=1,
    )
    df["source"] = "hateval"
    df = df.rename(columns={"text": "comment"})

    return df[SCHEMA_COLUMNS]


def combine_datasets(*dataframes: pd.DataFrame) -> pd.DataFrame:
    """Concatenate normalized source DataFrames into a single dataset."""
    for df in dataframes:
        missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

    return pd.concat(dataframes, ignore_index=True)[SCHEMA_COLUMNS]
