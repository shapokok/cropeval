"""Synthetic demo data.

Generates a fake predictions table that looks like the output of a plant
disease classifier evaluated on lab and field images. The model is made
more accurate on lab images than on field images, so the domain gap is
clearly visible. The data is fully synthetic and reproducible: the same
``seed`` always gives the same table.
"""

import numpy as np
import pandas as pd

DEFAULT_CLASSES = (
    "healthy",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot",
)


def _simulate_domain(
    rng: np.random.Generator,
    classes: np.ndarray,
    n: int,
    accuracy: float,
    domain: str,
) -> pd.DataFrame:
    """Simulate ``n`` predictions for one domain with the given accuracy.

    Each true label is drawn uniformly from ``classes``. With probability
    ``accuracy`` the prediction equals the true label; otherwise it is a
    random *different* class.
    """
    k = len(classes)
    true_idx = rng.integers(0, k, size=n)
    is_correct = rng.random(n) < accuracy

    # For a wrong prediction, shift the true index by 1..k-1 (mod k): this
    # always lands on a different class, chosen uniformly among the others.
    shift = rng.integers(1, k, size=n)
    pred_idx = np.where(is_correct, true_idx, (true_idx + shift) % k)

    return pd.DataFrame(
        {
            "y_true": classes[true_idx],
            "y_pred": classes[pred_idx],
            "domain": domain,
        }
    )


def make_demo_data(
    n_lab: int = 400,
    n_field: int = 400,
    lab_accuracy: float = 0.92,
    field_accuracy: float = 0.71,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    seed: int = 0,
) -> pd.DataFrame:
    """Create a synthetic predictions table with columns
    ``y_true``, ``y_pred`` and ``domain``.

    ``lab_accuracy`` and ``field_accuracy`` are the *expected* accuracies;
    the actual values vary a little because the data is random.
    """
    if len(classes) < 2:
        raise ValueError("At least two classes are needed.")
    for name, n in (("n_lab", n_lab), ("n_field", n_field)):
        if n < 1:
            raise ValueError(f"{name} must be at least 1, got {n}.")
    for name, acc in (
        ("lab_accuracy", lab_accuracy),
        ("field_accuracy", field_accuracy),
    ):
        if not 0 <= acc <= 1:
            raise ValueError(f"{name} must be between 0 and 1, got {acc}.")

    rng = np.random.default_rng(seed)
    classes_arr = np.asarray(classes)
    lab = _simulate_domain(rng, classes_arr, n_lab, lab_accuracy, "lab")
    field = _simulate_domain(rng, classes_arr, n_field, field_accuracy, "field")
    return pd.concat([lab, field], ignore_index=True)
