"""Tests for cropeval.gap: point estimate and bootstrap CI."""

import numpy as np
import pandas as pd
import pytest

from cropeval.gap import accuracy_gap, bootstrap_gap


def make_df(lab_correct: int, lab_total: int, field_correct: int, field_total: int):
    """Build a table with a given number of correct predictions per domain."""
    rows = []
    for domain, correct, total in (
        ("lab", lab_correct, lab_total),
        ("field", field_correct, field_total),
    ):
        for i in range(total):
            rows.append(
                {"y_true": 1, "y_pred": 1 if i < correct else 0, "domain": domain}
            )
    return pd.DataFrame(rows)


def test_gap_by_hand():
    # lab 9/10 = 0.9, field 6/10 = 0.6 -> gap 0.3
    df = make_df(9, 10, 6, 10)
    assert accuracy_gap(df) == pytest.approx(0.3)
    result = bootstrap_gap(df, n_boot=200, seed=0)
    assert result.lab_accuracy == pytest.approx(0.9)
    assert result.field_accuracy == pytest.approx(0.6)
    assert result.gap == pytest.approx(0.3)


def test_ci_contains_gap_and_is_ordered():
    df = make_df(80, 100, 55, 100)
    r = bootstrap_gap(df, n_boot=2000, seed=1)
    assert r.ci_low < r.gap < r.ci_high


def test_same_seed_same_result():
    df = make_df(80, 100, 55, 100)
    assert bootstrap_gap(df, n_boot=500, seed=42) == bootstrap_gap(
        df, n_boot=500, seed=42
    )


def test_different_seed_different_interval():
    df = make_df(80, 100, 55, 100)
    a = bootstrap_gap(df, n_boot=500, seed=1)
    b = bootstrap_gap(df, n_boot=500, seed=2)
    assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)
    assert a.gap == b.gap  # the point estimate does not depend on the seed


def test_no_variation_gives_zero_width_interval():
    # All lab right, all field wrong: every resample gives gap 1.0
    r = bootstrap_gap(make_df(5, 5, 0, 5), n_boot=100, seed=0)
    assert r.ci_low == r.ci_high == pytest.approx(1.0)


def test_ci_width_close_to_normal_approximation():
    # With large samples the bootstrap CI should be close to the textbook
    # formula gap +- 1.96 * sqrt(p1(1-p1)/n1 + p2(1-p2)/n2).
    df = make_df(800, 1000, 600, 1000)
    r = bootstrap_gap(df, n_boot=4000, seed=0)
    se = np.sqrt(0.8 * 0.2 / 1000 + 0.6 * 0.4 / 1000)
    assert r.ci_high - r.ci_low == pytest.approx(2 * 1.96 * se, rel=0.1)


def test_missing_domain_raises():
    df = make_df(5, 5, 0, 0)  # no field rows
    with pytest.raises(ValueError, match="domain 'field'"):
        bootstrap_gap(df)
    with pytest.raises(ValueError, match="domain 'field'"):
        accuracy_gap(df)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_boot": 0}, "n_boot"),
        ({"n_boot": 10.5}, "n_boot"),
        ({"confidence": 1.0}, "confidence"),
        ({"confidence": 0}, "confidence"),
    ],
)
def test_bad_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bootstrap_gap(make_df(5, 10, 5, 10), **kwargs)
