import json

import pandas as pd

from src.data.sarcasm import SARCASM_SCHEMA_COLUMNS, build_spanish_sarcasm, load_sarcasm_en


def test_load_sarcasm_en_parses_json_lines(tmp_path):
    path = tmp_path / "sarcasm.json"
    lines = [
        {"is_sarcastic": 1, "headline": "local man discovers water is wet", "article_link": "x"},
        {"is_sarcastic": 0, "headline": "have a nice day", "article_link": "y"},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

    df = load_sarcasm_en(str(path))

    assert list(df.columns) == SARCASM_SCHEMA_COLUMNS
    assert len(df) == 2
    sarcastic_row = df[df["comment"] == "local man discovers water is wet"].iloc[0]
    assert sarcastic_row["is_sarcastic"] == 1
    assert sarcastic_row["language"] == "en"


def test_build_spanish_sarcasm_translates_and_relabels_language():
    english_df = pd.DataFrame(
        [
            {
                "comment": "hello world",
                "language": "en",
                "is_sarcastic": 1,
                "source": "sarcasm_headlines",
            },
            {
                "comment": "nice day",
                "language": "en",
                "is_sarcastic": 0,
                "source": "sarcasm_headlines",
            },
        ]
    )

    def fake_translate(texts: list[str]) -> list[str]:
        return [f"[es]{t}" for t in texts]

    spanish_df = build_spanish_sarcasm(english_df, translate_fn=fake_translate)

    assert list(spanish_df.columns) == SARCASM_SCHEMA_COLUMNS
    assert set(spanish_df["language"]) == {"es"}
    assert spanish_df["comment"].tolist() == ["[es]hello world", "[es]nice day"]
    assert spanish_df["is_sarcastic"].tolist() == [1, 0]
    assert set(spanish_df["source"]) == {"sarcasm_headlines_es_mt"}
