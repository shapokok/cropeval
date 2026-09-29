"""Tests for cropeval.metrics: hand-computed examples and scikit-learn."""

import numpy as np
import pytest
from sklearn import metrics as skm

from cropeval.metrics import accuracy, confusion_matrix, macro_f1, per_class_metrics

# A tiny example we can check by hand.
#
#   y_true: a a a b b c
#   y_pred: a a b b c c
#
# Confusion matrix (rows = true, columns = predicted, order a, b, c):
#   a: [2, 1, 0]
#   b: [0, 1, 1]
#   c: [0, 0, 1]
Y_TRUE = ["a", "a", "a", "b", "b", "c"]
Y_PRED = ["a", "a", "b", "b", "c", "c"]


def test_accuracy_by_hand():
    # 4 correct out of 6
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(4 / 6)


def test_confusion_matrix_by_hand():
    expected = np.array([[2, 1, 0], [0, 1, 1], [0, 0, 1]])
    np.testing.assert_array_equal(confusion_matrix(Y_TRUE, Y_PRED), expected)


def test_per_class_by_hand():
    m = per_class_metrics(Y_TRUE, Y_PRED)
    assert m["labels"].tolist() == ["a", "b", "c"]
    # precision = TP / predicted: a 2/2, b 1/2, c 1/2
    np.testing.assert_allclose(m["precision"], [1.0, 0.5, 0.5])
    # recall = TP / support: a 2/3, b 1/2, c 1/1
    np.testing.assert_allclose(m["recall"], [2 / 3, 0.5, 1.0])
    # f1 = 2PR / (P + R): a 0.8, b 0.5, c 2/3
    np.testing.assert_allclose(m["f1"], [0.8, 0.5, 2 / 3])
    np.testing.assert_array_equal(m["support"], [3, 2, 1])


def test_macro_f1_by_hand():
    assert macro_f1(Y_TRUE, Y_PRED) == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)


def test_class_never_predicted_gives_zero_not_nan():
    # Class 1 is never predicted -> precision 0/0 -> 0.0
    m = per_class_metrics([0, 1], [0, 0])
    np.testing.assert_allclose(m["precision"], [0.5, 0.0])
    np.testing.assert_allclose(m["f1"], [2 / 3, 0.0])


def test_explicit_labels_order():
    cm = confusion_matrix(Y_TRUE, Y_PRED, labels=["c", "b", "a"])
    np.testing.assert_array_equal(cm, [[1, 0, 0], [1, 1, 0], [0, 1, 2]])


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        ([0, 1], [0], "same length"),
        ([], [], "empty"),
        ([[0, 1]], [[0, 1]], "1-D"),
    ],
)
def test_bad_input(y_true, y_pred, message):
    with pytest.raises(ValueError, match=message):
        accuracy(y_true, y_pred)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_sklearn_on_random_data(seed):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 4, size=200)
    # Make predictions mostly right, with some random errors.
    y_pred = np.where(rng.random(200) < 0.7, y_true, rng.integers(0, 4, size=200))

    assert accuracy(y_true, y_pred) == pytest.approx(skm.accuracy_score(y_true, y_pred))
    np.testing.assert_array_equal(
        confusion_matrix(y_true, y_pred), skm.confusion_matrix(y_true, y_pred)
    )
    m = per_class_metrics(y_true, y_pred)
    p, r, f, s = skm.precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    np.testing.assert_allclose(m["precision"], p)
    np.testing.assert_allclose(m["recall"], r)
    np.testing.assert_allclose(m["f1"], f)
    np.testing.assert_array_equal(m["support"], s)
    assert macro_f1(y_true, y_pred) == pytest.approx(
        skm.f1_score(y_true, y_pred, average="macro", zero_division=0)
    )


def test_matches_sklearn_with_string_labels():
    assert macro_f1(Y_TRUE, Y_PRED) == pytest.approx(
        skm.f1_score(Y_TRUE, Y_PRED, average="macro", zero_division=0)
    )
