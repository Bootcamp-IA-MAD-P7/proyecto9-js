import pandas as pd
import pytest

from src.data.loader import dataset_summary, load_comments


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "Comment": ["I hate you", "Have a nice day", "You are trash", "Great video!"],
            "Label": [1, 0, 1, 0],
        }
    ).to_csv(path, index=False)
    return path


def test_load_comments_returns_dataframe_with_consistent_columns(sample_csv):
    df = load_comments(str(sample_csv))
    assert list(df.columns) == ["comment", "label"]
    assert len(df) == 4


def test_load_comments_missing_required_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["hi"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_comments(str(path))


def test_dataset_summary_reports_size_duplicates_and_class_balance(sample_csv):
    df = load_comments(str(sample_csv))
    summary = dataset_summary(df)
    assert summary["n_rows"] == 4
    assert summary["n_duplicates"] == 0
    assert summary["n_nulls"] == 0
    assert summary["label_counts"] == {1: 2, 0: 2}
