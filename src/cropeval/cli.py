"""Command-line interface (``cropeval`` command, built with argparse).

Commands:

- ``cropeval demo -o data.csv``: write a synthetic predictions CSV.
- ``cropeval report data.csv -o out/``: evaluate a predictions CSV and write
  ``metrics.json``, ``report.md`` and two PNG plots into ``out/``.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cropeval import __version__
from cropeval.demo import make_demo_data
from cropeval.gap import GapResult, bootstrap_gap
from cropeval.io import DOMAINS, DataValidationError, load_predictions
from cropeval.metrics import accuracy, confusion_matrix, per_class_metrics
from cropeval.plots import plot_confusion_matrices, plot_f1_by_class

CONFUSION_PNG = "confusion_matrices.png"
F1_PNG = "f1_by_class.png"


# --------------------------------------------------------------------------
# Building the report
# --------------------------------------------------------------------------


def compute_metrics(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """Compute all metrics for a validated predictions table.

    Returns a plain dict (only lists, numbers and strings) so it can be
    written to JSON directly.
    """
    # Same label order for both domains; str() makes numeric labels JSON keys.
    labels = np.unique(
        np.concatenate([df["y_true"].to_numpy(), df["y_pred"].to_numpy()])
    )
    label_names = [str(label) for label in labels]

    domains = {}
    for domain in DOMAINS:
        part = df[df["domain"] == domain]
        if len(part) == 0:
            raise DataValidationError(f"No rows with domain '{domain}'.")
        m = per_class_metrics(part["y_true"], part["y_pred"], labels=labels)
        domains[domain] = {
            "n": int(len(part)),
            "accuracy": accuracy(part["y_true"], part["y_pred"]),
            "macro_f1": float(np.mean(m["f1"])),
            "per_class": {
                name: {
                    "precision": float(m["precision"][i]),
                    "recall": float(m["recall"][i]),
                    "f1": float(m["f1"][i]),
                    "support": int(m["support"][i]),
                }
                for i, name in enumerate(label_names)
            },
            "confusion_matrix": confusion_matrix(
                part["y_true"], part["y_pred"], labels=labels
            ).tolist(),
        }

    gap: GapResult = bootstrap_gap(df, n_boot=n_boot, seed=seed)
    return {
        "cropeval_version": __version__,
        "n_rows": int(len(df)),
        "labels": label_names,
        "domains": domains,
        "accuracy_gap": {
            "gap": gap.gap,
            "ci_low": gap.ci_low,
            "ci_high": gap.ci_high,
            "confidence": gap.confidence,
            "n_boot": gap.n_boot,
            "seed": gap.seed,
        },
    }


def render_markdown(metrics: dict, source: str) -> str:
    """Turn the metrics dict into a short human-readable Markdown report."""
    lab = metrics["domains"]["lab"]
    field = metrics["domains"]["field"]
    gap = metrics["accuracy_gap"]
    level = round(gap["confidence"] * 100)

    lines = [
        "# cropeval report",
        "",
        f"Input: `{source}` ({metrics['n_rows']} rows)",
        "",
        "## Summary",
        "",
        "| domain | n | accuracy | macro F1 |",
        "|---|---:|---:|---:|",
        f"| lab | {lab['n']} | {lab['accuracy']:.3f} | {lab['macro_f1']:.3f} |",
        f"| field | {field['n']} | {field['accuracy']:.3f} | {field['macro_f1']:.3f} |",
        "",
        f"**Accuracy gap (lab − field): {gap['gap']:+.3f}**, "
        f"{level}% bootstrap CI [{gap['ci_low']:+.3f}, {gap['ci_high']:+.3f}] "
        f"(n_boot={gap['n_boot']}, seed={gap['seed']}).",
        "",
    ]
    # One plain-language sentence, so the reader does not have to interpret
    # the interval themselves.
    if gap["ci_low"] > 0:
        lines.append("The model is significantly less accurate on field images.")
    elif gap["ci_high"] < 0:
        lines.append("The model is significantly more accurate on field images.")
    else:
        lines.append("The interval includes 0: no clear difference between domains.")

    lines += [
        "",
        "## Per-class F1",
        "",
        "| class | lab F1 | field F1 | difference |",
        "|---|---:|---:|---:|",
    ]
    for name in metrics["labels"]:
        f_lab = lab["per_class"][name]["f1"]
        f_field = field["per_class"][name]["f1"]
        lines.append(
            f"| {name} | {f_lab:.3f} | {f_field:.3f} | {f_lab - f_field:+.3f} |"
        )

    lines += [
        "",
        "## Plots",
        "",
        f"![Confusion matrices]({CONFUSION_PNG})",
        "",
        f"![Per-class F1]({F1_PNG})",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_demo(args: argparse.Namespace) -> int:
    """``cropeval demo``: write a synthetic CSV."""
    df = make_demo_data(n_lab=args.n_lab, n_field=args.n_field, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} synthetic predictions to {out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """``cropeval report``: evaluate a CSV and write the report files."""
    df = load_predictions(args.input)
    metrics = compute_metrics(df, n_boot=args.n_boot, seed=args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "report.md").write_text(render_markdown(metrics, Path(args.input).name))
    plot_confusion_matrices(df, out_dir / CONFUSION_PNG)
    plot_f1_by_class(df, out_dir / F1_PNG)

    gap = metrics["accuracy_gap"]
    print(
        f"Accuracy gap (lab - field): {gap['gap']:+.3f} "
        f"[{gap['ci_low']:+.3f}, {gap['ci_high']:+.3f}]"
    )
    print(f"Report written to {out_dir}/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="cropeval",
        description="Evaluate the lab-to-field domain gap of image classifiers.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="write a synthetic predictions CSV")
    demo.add_argument("-o", "--output", required=True, help="path of the CSV to write")
    demo.add_argument("--n-lab", type=int, default=400, help="lab rows (default 400)")
    demo.add_argument(
        "--n-field", type=int, default=400, help="field rows (default 400)"
    )
    demo.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    demo.set_defaults(func=cmd_demo)

    report = sub.add_parser("report", help="evaluate a predictions CSV")
    report.add_argument("input", help="CSV with columns y_true, y_pred, domain")
    report.add_argument("-o", "--output", required=True, help="output folder")
    report.add_argument(
        "--n-boot", type=int, default=1000, help="bootstrap resamples (default 1000)"
    )
    report.add_argument(
        "--seed", type=int, default=0, help="bootstrap seed (default 0)"
    )
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``cropeval`` console script.

    Returns the exit code: 0 on success, 1 on a data/input error. Expected
    errors are printed as one clear line instead of a Python traceback.
    """
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, DataValidationError, ValueError) as err:
        print(f"cropeval: error: {err}", file=sys.stderr)
        return 1
