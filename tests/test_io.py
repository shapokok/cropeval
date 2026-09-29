"""Tests for loading and validating the input CSV."""

import pandas as pd
import pytest

from cropeval.io import DataValidationError, load_predictions, validate_predictions


def write_csv(tmp_path, text):
    """Write ``text`` to a temporary CSV file and return its path."""
    path = tmp_path / "preds.csv"
    path.write_text(text)
    return path


def test_load_valid_file(tmp_path):
    path = write_csv(tmp_path, "y_true,y_pred,domain\nrust,rust,lab\nrust,scab,field\n")
    df = load_predictions(path)
    assert list(df.columns) == ["y_true", "y_pred", "domain"]
    assert df["domain"].tolist() == ["lab", "field"]


def test_domain_is_normalised_and_extra_columns_dropped():
    df = pd.DataFrame(
        {"y_true": [0, 1], "y_pred": [0, 0], "domain": [" Lab", "FIELD "], "x": [1, 2]}
    )
    clean = validate_predictions(df)
    assert clean["domain"].tolist() == ["lab", "field"]
    assert "x" not in clean.columns
    # The caller's DataFrame must not be changed.
    assert df["domain"].tolist() == [" Lab", "FIELD "]


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_predictions(tmp_path / "nope.csv")


def test_empty_file(tmp_path):
    with pytest.raises(DataValidationError, match="empty"):
        load_predictions(write_csv(tmp_path, ""))


def test_header_only(tmp_path):
    with pytest.raises(DataValidationError, match="no rows"):
        load_predictions(write_csv(tmp_path, "y_true,y_pred,domain\n"))


def test_missing_column():
    df = pd.DataFrame({"y_true": [0], "domain": ["lab"]})
    with pytest.raises(DataValidationError, match="Missing required column.*y_pred"):
        validate_predictions(df)


def test_missing_value_reports_csv_line(tmp_path):
    path = write_csv(tmp_path, "y_true,y_pred,domain\n0,0,lab\n1,,field\n")
    # Data row 2 is line 3 of the file (line 1 is the header).
    with pytest.raises(DataValidationError, match=r"'y_pred'.*line\(s\) 3"):
        load_predictions(path)


def test_unknown_domain():
    df = pd.DataFrame(
        {"y_true": [0, 1], "y_pred": [0, 1], "domain": ["lab", "greenhouse"]}
    )
    with pytest.raises(DataValidationError, match="'greenhouse'.*line\\(s\\) 3"):
        validate_predictions(df)


def test_many_bad_rows_are_truncated():
    df = pd.DataFrame({"y_true": [0] * 8, "y_pred": [0] * 8, "domain": ["x"] * 8})
    with pytest.raises(DataValidationError, match=r"and 3 more"):
        validate_predictions(df)
