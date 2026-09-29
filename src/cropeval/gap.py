"""Lab-to-field domain gap.

The gap is ``accuracy(lab) - accuracy(field)``. A positive gap means the
model is worse on field images than on lab images.

To tell how reliable the gap is, we compute a bootstrap confidence
interval (percentile method):

1. Resample the lab rows with replacement (same number of rows), and
   separately resample the field rows. Resampling each domain on its own
   keeps the lab/field sizes fixed, as in the real data.
2. Compute the gap on the resampled data.
3. Repeat ``n_boot`` times; the 2.5% and 97.5% percentiles of these gaps
   form the 95% confidence interval.

All randomness comes from ``numpy.random.default_rng(seed)``, so the same
seed always gives the same interval.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cropeval.metrics import accuracy


@dataclass(frozen=True)
class GapResult:
    """Accuracy gap between lab and field with its confidence interval."""

    lab_accuracy: float
    field_accuracy: float
    gap: float  # lab_accuracy - field_accuracy
    ci_low: float
    ci_high: float
    confidence: float
    n_boot: int
    seed: int


def _correct_by_domain(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return 0/1 arrays "prediction is correct" for lab and field rows."""
    correct = (df["y_true"] == df["y_pred"]).to_numpy(dtype=float)
    domain = df["domain"].to_numpy()
    lab = correct[domain == "lab"]
    field = correct[domain == "field"]
    for name, values in (("lab", lab), ("field", field)):
        if len(values) == 0:
            raise ValueError(f"No rows with domain '{name}'; cannot compute the gap.")
    return lab, field


def accuracy_gap(df: pd.DataFrame) -> float:
    """Accuracy on lab rows minus accuracy on field rows.

    ``df`` must have columns ``y_true``, ``y_pred`` and ``domain``
    (see ``cropeval.io.validate_predictions``).
    """
    _correct_by_domain(df)  # only to check that both domains are present
    lab = df[df["domain"] == "lab"]
    field = df[df["domain"] == "field"]
    return accuracy(lab["y_true"], lab["y_pred"]) - accuracy(
        field["y_true"], field["y_pred"]
    )


def _bootstrap_means(values: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """Means of ``n_boot`` bootstrap resamples of ``values``.

    Row ``b`` of ``idx`` holds the indices drawn for resample ``b``; taking
    the mean along each row gives the accuracy of every resample at once.
    """
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return values[idx].mean(axis=1)


def bootstrap_gap(
    df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> GapResult:
    """Accuracy gap (lab - field) with a bootstrap confidence interval.

    Args:
        df: validated predictions table with both ``lab`` and ``field`` rows.
        n_boot: number of bootstrap resamples.
        seed: seed for ``numpy.random.default_rng``; same seed, same result.
        confidence: confidence level of the interval, e.g. 0.95.
    """
    if not isinstance(n_boot, int | np.integer) or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}.")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence!r}.")

    lab, field = _correct_by_domain(df)
    rng = np.random.default_rng(seed)

    gaps = _bootstrap_means(lab, n_boot, rng) - _bootstrap_means(field, n_boot, rng)

    alpha = 1 - confidence
    ci_low, ci_high = np.percentile(gaps, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    lab_acc = float(lab.mean())
    field_acc = float(field.mean())
    return GapResult(
        lab_accuracy=lab_acc,
        field_accuracy=field_acc,
        gap=lab_acc - field_acc,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence=confidence,
        n_boot=n_boot,
        seed=seed,
    )
