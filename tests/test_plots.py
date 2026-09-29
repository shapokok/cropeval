"""Tests for the plotting functions (they must produce valid PNG files)."""

import matplotlib
import matplotlib.pyplot as plt

from cropeval.demo import make_demo_data
from cropeval.plots import plot_confusion_matrices, plot_f1_by_class

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_backend_is_agg():
    assert matplotlib.get_backend().lower() == "agg"


def test_plots_write_png_files(tmp_path):
    df = make_demo_data(n_lab=50, n_field=50, seed=0)
    for plot, name in (
        (plot_confusion_matrices, "cm.png"),
        (plot_f1_by_class, "f1.png"),
    ):
        # A nested folder that does not exist yet must be created.
        path = plot(df, tmp_path / "plots" / name)
        assert path.read_bytes().startswith(PNG_SIGNATURE)
    # Figures are closed after saving, so memory does not grow.
    assert plt.get_fignums() == []


def test_plots_work_with_integer_labels(tmp_path):
    df = make_demo_data(n_lab=20, n_field=20, classes=(0, 1, 2), seed=0)
    assert plot_confusion_matrices(df, tmp_path / "cm.png").exists()
    assert plot_f1_by_class(df, tmp_path / "f1.png").exists()
