"""End-to-end tests of the command-line interface (files go to tmp_path)."""

import json

import pytest

from cropeval.cli import main, render_markdown


@pytest.fixture
def demo_csv(tmp_path):
    """A small synthetic CSV written by ``cropeval demo``."""
    path = tmp_path / "data.csv"
    assert main(["demo", "-o", str(path), "--n-lab", "80", "--n-field", "80"]) == 0
    return path


def test_demo_writes_csv(demo_csv, capsys):
    lines = demo_csv.read_text().splitlines()
    assert lines[0] == "y_true,y_pred,domain"
    assert len(lines) == 1 + 160


def test_report_writes_all_files(demo_csv, tmp_path, capsys):
    out = tmp_path / "out"
    code = main(["report", str(demo_csv), "-o", str(out), "--n-boot", "200"])
    assert code == 0
    assert "Accuracy gap" in capsys.readouterr().out
    for name in (
        "metrics.json",
        "report.md",
        "confusion_matrices.png",
        "f1_by_class.png",
    ):
        assert (out / name).is_file(), name


def test_metrics_json_content(demo_csv, tmp_path):
    out = tmp_path / "out"
    main(["report", str(demo_csv), "-o", str(out), "--n-boot", "200", "--seed", "3"])
    metrics = json.loads((out / "metrics.json").read_text())

    assert metrics["n_rows"] == 160
    lab, field = metrics["domains"]["lab"], metrics["domains"]["field"]
    gap = metrics["accuracy_gap"]
    assert gap["gap"] == pytest.approx(lab["accuracy"] - field["accuracy"])
    assert gap["ci_low"] <= gap["gap"] <= gap["ci_high"]
    assert (gap["n_boot"], gap["seed"]) == (200, 3)
    # Confusion matrix is labels x labels and sums to the number of rows.
    k = len(metrics["labels"])
    cm = lab["confusion_matrix"]
    assert len(cm) == k and all(len(row) == k for row in cm)
    assert sum(map(sum, cm)) == lab["n"] == 80


def test_report_is_reproducible(demo_csv, tmp_path):
    for name in ("a", "b"):
        main(["report", str(demo_csv), "-o", str(tmp_path / name), "--n-boot", "100"])
    assert (tmp_path / "a" / "metrics.json").read_text() == (
        tmp_path / "b" / "metrics.json"
    ).read_text()


def test_report_markdown(demo_csv, tmp_path):
    out = tmp_path / "out"
    main(["report", str(demo_csv), "-o", str(out), "--n-boot", "100"])
    text = (out / "report.md").read_text()
    assert "Accuracy gap (lab − field)" in text
    assert "healthy" in text
    assert "![Confusion matrices](confusion_matrices.png)" in text


def test_missing_input_file_is_a_clean_error(tmp_path, capsys):
    code = main(["report", str(tmp_path / "nope.csv"), "-o", str(tmp_path / "out")])
    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_invalid_csv_is_a_clean_error(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("y_true,y_pred\n1,1\n")
    assert main(["report", str(bad), "-o", str(tmp_path / "out")]) == 1
    assert "Missing required column" in capsys.readouterr().err


def test_single_domain_is_a_clean_error(tmp_path, capsys):
    only_lab = tmp_path / "lab.csv"
    only_lab.write_text("y_true,y_pred,domain\n1,1,lab\n0,1,lab\n")
    assert main(["report", str(only_lab), "-o", str(tmp_path / "out")]) == 1
    assert "field" in capsys.readouterr().err


def test_no_command_shows_usage(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert "usage" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("ci", "expected"),
    [
        ((0.05, 0.20), "less accurate on field"),
        ((-0.20, -0.05), "more accurate on field"),
        ((-0.05, 0.10), "no clear difference"),
    ],
)
def test_markdown_conclusion_follows_the_interval(ci, expected):
    domain = {"n": 10, "accuracy": 0.8, "macro_f1": 0.8, "per_class": {}}
    metrics = {
        "n_rows": 20,
        "labels": [],
        "domains": {"lab": domain, "field": domain},
        "accuracy_gap": {
            "gap": 0.0,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "confidence": 0.95,
            "n_boot": 100,
            "seed": 0,
        },
    }
    assert expected in render_markdown(metrics, "x.csv")
