"""Plots for the evaluation report.

- ``plot_confusion_matrices``: lab and field confusion matrices side by side.
- ``plot_f1_by_class``: grouped bar chart of per-class F1 for both domains.

Both functions save a PNG file and return its path. Matplotlib uses the
non-interactive Agg backend, so plots work on servers and in CI without a
display.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must be called before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cropeval.metrics import confusion_matrix, per_class_metrics  # noqa: E402

# One fixed colour per domain, used in every plot (blue = lab, orange = field).
# The pair stays distinguishable for colour-blind readers.
DOMAIN_COLORS = {"lab": "#2a78d6", "field": "#eb6834"}
DOMAINS = ("lab", "field")
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"


def _all_labels(df: pd.DataFrame) -> np.ndarray:
    """Sorted class labels found anywhere in the table.

    Using the same label list for both domains keeps rows/columns and bars
    aligned between lab and field.
    """
    return np.unique(np.concatenate([df["y_true"].to_numpy(), df["y_pred"].to_numpy()]))


def _style_axes(ax) -> None:
    """Quiet axes: no top/right spines, muted tick labels."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelcolor=TEXT_COLOR)


def plot_confusion_matrices(df: pd.DataFrame, path: str | Path) -> Path:
    """Save lab and field confusion matrices side by side.

    Cells show row-normalised values (share of each true class), so domains
    with different numbers of images can be compared. The raw count is
    printed in each cell as well.
    """
    labels = _all_labels(df)
    fig, axes = plt.subplots(
        1, 2, figsize=(4.2 + 1.3 * len(labels), 0.9 + 0.8 * len(labels)), sharey=True
    )

    for ax, domain in zip(axes, DOMAINS, strict=True):
        part = df[df["domain"] == domain]
        cm = confusion_matrix(part["y_true"], part["y_pred"], labels=labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        # Share of each true class; rows with no samples stay 0.
        share = np.divide(cm, row_sums, out=np.zeros(cm.shape), where=row_sums > 0)

        image = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                # Dark text on light cells, white text on dark cells.
                color = "white" if share[i, j] > 0.6 else TEXT_COLOR
                ax.text(
                    j, i, cm[i, j], ha="center", va="center", color=color, fontsize=9
                )

        accuracy = np.trace(cm) / cm.sum() if cm.sum() else 0.0
        ax.set_title(
            f"{domain}  (n={cm.sum()}, accuracy={accuracy:.2f})",
            color=TEXT_COLOR,
            fontsize=11,
            loc="left",
        )
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted class", color=MUTED_COLOR)
        _style_axes(ax)
    axes[0].set_ylabel("True class", color=MUTED_COLOR)

    colorbar = fig.colorbar(image, ax=axes, shrink=0.8)
    colorbar.set_label("Share of true class", color=MUTED_COLOR)
    colorbar.outline.set_visible(False)

    return _save(fig, path)


def plot_f1_by_class(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a grouped bar chart: per-class F1 for lab and field."""
    labels = _all_labels(df)
    x = np.arange(len(labels))
    width = 0.38
    gap = 0.02  # small empty space between the two bars of a class

    fig, ax = plt.subplots(figsize=(2.0 + 1.3 * len(labels), 4.2))
    for k, domain in enumerate(DOMAINS):
        part = df[df["domain"] == domain]
        f1 = per_class_metrics(part["y_true"], part["y_pred"], labels=labels)["f1"]
        offset = (k - 0.5) * (width + gap)
        ax.bar(
            x + offset,
            f1,
            width=width,
            color=DOMAIN_COLORS[domain],
            label=f"{domain} (macro F1 = {f1.mean():.2f})",
            zorder=2,
        )

    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score", color=MUTED_COLOR)
    ax.set_title("Per-class F1: lab vs field", color=TEXT_COLOR, loc="left")
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    _style_axes(ax)

    return _save(fig, path)


def _save(fig, path: str | Path) -> Path:
    """Save a figure as PNG, create the folder if needed, and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)  # free memory; important when many plots are made
    return path
