"""Tests for the synthetic demo data generator."""

import pandas as pd
import pytest

from cropeval.demo import DEFAULT_CLASSES, make_demo_data
from cropeval.io import validate_predictions
from cropeval.metrics import accuracy


def test_shape_and_columns():
    df = make_demo_data(n_lab=30, n_field=20, seed=0)
    assert list(df.columns) == ["y_true", "y_pred", "domain"]
    assert (df["domain"] == "lab").sum() == 30
    assert (df["domain"] == "field").sum() == 20
    assert set(df["y_true"]) <= set(DEFAULT_CLASSES)
    # The generated table must pass our own input validation.
    validate_predictions(df)


def test_same_seed_same_data():
    pd.testing.assert_frame_equal(make_demo_data(seed=7), make_demo_data(seed=7))


def test_different_seed_different_data():
    assert not make_demo_data(seed=1).equals(make_demo_data(seed=2))


def test_lab_is_more_accurate_than_field():
    df = make_demo_data(n_lab=2000, n_field=2000, seed=0)
    lab = df[df["domain"] == "lab"]
    field = df[df["domain"] == "field"]
    # Expected 0.92 and 0.71; with 2000 rows each the values are close.
    assert accuracy(lab["y_true"], lab["y_pred"]) == pytest.approx(0.92, abs=0.03)
    assert accuracy(field["y_true"], field["y_pred"]) == pytest.approx(0.71, abs=0.03)


def test_wrong_predictions_are_a_different_class():
    # With accuracy 0 every prediction must differ from the true label.
    df = make_demo_data(n_lab=200, n_field=200, lab_accuracy=0, field_accuracy=0)
    assert (df["y_true"] != df["y_pred"]).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"classes": ("only_one",)}, "two classes"),
        ({"n_lab": 0}, "n_lab"),
        ({"field_accuracy": 1.5}, "field_accuracy"),
    ],
)
def test_bad_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_demo_data(**kwargs)
