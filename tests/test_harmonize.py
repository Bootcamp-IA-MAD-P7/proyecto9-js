import json

import pandas as pd
import pytest

from src.data.harmonize import (
    SCHEMA_COLUMNS,
    combine_datasets,
    load_detests,
    load_ethos,
    load_hascosva,
    load_haternet,
    load_hateval,
    load_hatexplain,
    load_measuring_hate_speech,
    load_offendes,
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


def test_load_hateval_maps_target_to_category_and_keeps_language(tmp_path):
    train_path = tmp_path / "train.parquet"
    dev_path = tmp_path / "dev.parquet"
    test_path = tmp_path / "test.parquet"

    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "text": ["go back to your country", "que vuelvan a su pais", "have a nice day"],
            "target": ["mig", "mig", "mis"],
            "language": ["en", "es", "en"],
            "HS": [1, 1, 0],
            "TR": [0, 0, 0],
            "AG": [0, 0, 0],
        }
    ).to_parquet(train_path)
    empty_columns = ["id", "text", "target", "language", "HS", "TR", "AG"]
    pd.DataFrame(columns=empty_columns).to_parquet(dev_path)
    pd.DataFrame(columns=empty_columns).to_parquet(test_path)

    df = load_hateval(str(train_path), str(dev_path), str(test_path))

    assert list(df.columns) == SCHEMA_COLUMNS
    assert set(df["language"]) == {"en", "es"}

    en_hate_row = df[df["comment"] == "go back to your country"].iloc[0]
    assert en_hate_row["label"] == 1
    assert en_hate_row["categories"] == "xenophobia"

    es_hate_row = df[df["comment"] == "que vuelvan a su pais"].iloc[0]
    assert es_hate_row["language"] == "es"
    assert es_hate_row["categories"] == "xenophobia"

    not_hate_row = df[df["comment"] == "have a nice day"].iloc[0]
    assert not_hate_row["label"] == 0
    assert not_hate_row["categories"] == "other"


def test_load_haternet_parses_pipe_delimited_lines(tmp_path):
    path = tmp_path / "haternet.txt"
    path.write_text(
        "id=1;||;eres un inutil;||;1\n"
        "id=2;||;que tengas buen dia;||;0\n",
        encoding="utf-8",
    )

    df = load_haternet(str(path))

    assert list(df.columns) == SCHEMA_COLUMNS
    assert set(df["language"]) == {"es"}
    hate_row = df[df["comment"] == "eres un inutil"].iloc[0]
    assert hate_row["label"] == 1
    assert hate_row["categories"] == "other"


def test_load_offendes_maps_person_and_group_offense_to_hate(tmp_path):
    path = tmp_path / "offendes.tsv"
    path.write_text(
        'comment_id\tcomment\tinfluencer\tinfluencer_gender\tmedia\tlabel\n'
        '"1"\t"eres un inutil"\t"x"\t"man"\t"twitter"\t"OFP"\n'
        '"2"\t"odio a esa gente"\t"x"\t"man"\t"twitter"\t"OFG"\n'
        '"3"\t"buen dia a todos"\t"x"\t"man"\t"twitter"\t"NO"\n',
        encoding="utf-8",
    )

    df = load_offendes(str(path))

    assert list(df.columns) == SCHEMA_COLUMNS
    assert set(df["language"]) == {"es"}
    assert df[df["comment"] == "eres un inutil"].iloc[0]["label"] == 1
    assert df[df["comment"] == "odio a esa gente"].iloc[0]["label"] == 1
    assert df[df["comment"] == "buen dia a todos"].iloc[0]["label"] == 0


def test_load_hascosva_tags_hate_as_xenophobia(tmp_path):
    path = tmp_path / "hascosva.tsv"
    pd.DataFrame(
        {
            "text": ["vuelvan a su pais venecos", "bienvenidos a todos"],
            "label": [1, 0],
            "variation": ["latam", "europe"],
        }
    ).to_csv(path, sep="\t", index=False)

    df = load_hascosva(str(path))

    assert list(df.columns) == SCHEMA_COLUMNS
    assert set(df["language"]) == {"es"}
    hate_row = df[df["comment"] == "vuelvan a su pais venecos"].iloc[0]
    assert hate_row["label"] == 1
    assert hate_row["categories"] == "xenophobia"
    normal_row = df[df["comment"] == "bienvenidos a todos"].iloc[0]
    assert normal_row["label"] == 0
    assert normal_row["categories"] == "other"


def test_load_detests_tags_stereotype_as_xenophobia(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame(
        {"text": ["los inmigrantes roban trabajo"], "stereotype": [1]}
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        {"text": ["gracias por la ayuda"], "stereotype": [0]}
    ).to_csv(test_path, index=False)

    df = load_detests(str(train_path), str(test_path))

    assert list(df.columns) == SCHEMA_COLUMNS
    assert len(df) == 2
    hate_row = df[df["comment"] == "los inmigrantes roban trabajo"].iloc[0]
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
