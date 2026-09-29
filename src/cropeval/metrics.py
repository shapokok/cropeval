"""Classification metrics implemented with NumPy.

All functions take two 1-D sequences of the same length: ``y_true`` (true
labels) and ``y_pred`` (predicted labels). Labels may be integers or
strings. The results match scikit-learn (this is checked in the tests).

Division by zero (e.g. precision of a class that is never predicted) gives
0.0, the same as scikit-learn with ``zero_division=0``.
"""

import numpy as np


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to 1-D NumPy arrays and check that they match."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {len(y_true)} and {len(y_pred)}."
        )
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    return y_true, y_pred


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise ``numerator / denominator`` with 0.0 where denominator is 0."""
    numerator = numerator.astype(float)
    result = np.zeros_like(numerator)
    nonzero = denominator != 0
    result[nonzero] = numerator[nonzero] / denominator[nonzero]
    return result


def accuracy(y_true, y_pred) -> float:
    """Fraction of predictions that are exactly right."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    """Confusion matrix: ``cm[i, j]`` = number of samples of true class
    ``labels[i]`` that were predicted as ``labels[j]``.

    Args:
        labels: order of the classes. By default, all labels that appear in
            ``y_true`` or ``y_pred``, sorted.
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)

    # Map every label to its row/column number.
    position = {label: i for i, label in enumerate(labels.tolist())}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true.tolist(), y_pred.tolist(), strict=True):
        # Samples with a label outside ``labels`` are skipped, like sklearn.
        if t in position and p in position:
            cm[position[t], position[p]] += 1
    return cm


def per_class_metrics(y_true, y_pred, labels=None) -> dict[str, np.ndarray]:
    """Precision, recall, F1 and support for every class.

    Returns:
        A dict with keys ``labels``, ``precision``, ``recall``, ``f1`` and
        ``support`` (number of true samples of the class). Each value is an
        array with one entry per class, in the order of ``labels``.
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    true_positive = np.diag(cm)
    predicted = cm.sum(axis=0)  # column sums: how often each class was predicted
    support = cm.sum(axis=1)  # row sums: how often each class really occurs

    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "labels": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def macro_f1(y_true, y_pred, labels=None) -> float:
    """Unweighted mean of per-class F1 scores (every class counts equally)."""
    return float(np.mean(per_class_metrics(y_true, y_pred, labels)["f1"]))
