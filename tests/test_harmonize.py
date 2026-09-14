import json

import pandas as pd
import pytest

from src.data.harmonize import (
    SCHEMA_COLUMNS,
    combine_datasets,
    load_ethos,
    load_hatexplain,
    load_measuring_hate_speech,
)


def test_load_hatexplain_resolves_majority_label_and_categories(tmp_path):
    path = tmp_path / "hatexplain.json"
    data = {
        "post1": {
            "post_tokens": ["i", "hate", "immigrants"],
            "annotators": [
                {"label": "hatespeech", "target": ["Refugee"]},
                {"label": "hatespeech", "target": ["Refugee"]},
                {"label": "normal", "target": ["None"]},
            ],
        },
        "post2": {
            "post_tokens": ["nice", "weather", "today"],
            "annotators": [
                {"label": "normal", "target": ["None"]},
                {"label": "normal", "target": ["None"]},
                {"label": "offensive", "target": ["None"]},
            ],
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    df = load_hatexplain(str(path))

    assert list(df.columns) == SCHEMA_COLUMNS
    hate_row = df[df["comment"] == "i hate immigrants"].iloc[0]
    assert hate_row["label"] == 1
    assert hate_row["categories"] == "xenophobia"

    normal_row = df[df["comment"] == "nice weather today"].iloc[0]
    assert normal_row["label"] == 0
    assert normal_row["categories"] == "other"


def test_load_ethos_binarizes_score_and_joins_categories(tmp_path):
    binary_path = tmp_path / "ethos_binary.csv"
    multilabel_path = tmp_path / "ethos_multilabel.csv"

    pd.DataFrame(
        {"comment": ["you are trash", "have a nice day"], "isHate": [0.8, 0.0]}
    ).to_csv(binary_path, sep=";", index=False)

    pd.DataFrame(
        {
            "comment": ["you are trash", "have a nice day"],
            "directed_vs_generalized": [1.0, 0.0],
            "gender": [1.0, 0.0],
            "race": [0.0, 0.0],
        }
    ).to_csv(multilabel_path, sep=";", index=False)

    df = load_ethos(str(binary_path), str(multilabel_path))

    assert list(df.columns) == SCHEMA_COLUMNS
    hate_row = df[df["comment"] == "you are trash"].iloc[0]
    assert hate_row["label"] == 1
    assert hate_row["categories"] == "misogyny"

    normal_row = df[df["comment"] == "have a nice day"].iloc[0]
    assert normal_row["label"] == 0
    assert normal_row["categories"] == "other"


def test_load_measuring_hate_speech_aggregates_annotations(tmp_path):
    path = tmp_path / "measuring.parquet"
    pd.DataFrame(
        {
            "comment_id": [1, 1, 2],
            "text": ["go back home", "go back home", "have a nice day"],
            "hatespeech": [2, 1, 0],
            "target_race": [0, 0, 0],
            "target_religion": [0, 0, 0],
            "target_origin": [1, 1, 0],
            "target_gender_transgender_men": [0, 0, 0],
            "target_gender_transgender_women": [0, 0, 0],
            "target_gender_transgender_unspecified": [0, 0, 0],
            "target_gender_women": [0, 0, 0],
            "target_sexuality_gay": [0, 0, 0],
            "target_sexuality_lesbian": [0, 0, 0],
            "target_sexuality_bisexual": [0, 0, 0],
            "target_disability": [0, 0, 0],
            "violence": [0, 0, 0],
        }
    ).to_parquet(path)

    df = load_measuring_hate_speech(str(path))

    assert list(df.columns) == SCHEMA_COLUMNS
    hate_row = df[df["comment"] == "go back home"].iloc[0]
    assert hate_row["label"] == 1
    assert hate_row["categories"] == "xenophobia"


def test_combine_datasets_concatenates_and_validates_schema():
    a = pd.DataFrame(
        [{"comment": "x", "language": "en", "label": 1, "categories": "racism", "source": "a"}]
    )
    b = pd.DataFrame(
        [{"comment": "y", "language": "es", "label": 0, "categories": "other", "source": "b"}]
    )

    combined = combine_datasets(a, b)

    assert len(combined) == 2
    assert list(combined.columns) == SCHEMA_COLUMNS


def test_combine_datasets_rejects_missing_columns():
    bad = pd.DataFrame([{"comment": "x", "label": 1}])
    with pytest.raises(ValueError):
        combine_datasets(bad)
