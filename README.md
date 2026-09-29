# cropeval

[![CI](https://github.com/shapokok/cropeval/actions/workflows/ci.yml/badge.svg)](https://github.com/shapokok/cropeval/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)

**cropeval** is a small Python package for evaluating the **lab-to-field
domain gap** of image classification models for plant disease detection.

Models trained on clean laboratory photos often perform worse on images
taken in the field. cropeval takes a model's predictions from both settings
and answers: *how big is the drop, and how sure are we about it?*

## Input

A CSV file with one row per image:

| column   | meaning                                     |
|----------|---------------------------------------------|
| `y_true` | true class label                            |
| `y_pred` | predicted class label                       |
| `domain` | where the image came from: `lab` or `field` |

Labels can be strings or integers. The file is validated on load; errors
name the problem and the CSV line numbers.

## Installation

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/shapokok/cropeval.git
cd cropeval
uv sync            # creates .venv with runtime and dev dependencies
uv run pytest      # run the test suite
```

## Usage

```bash
# 1. Create a synthetic predictions CSV (reproducible with --seed)
uv run cropeval demo -o data.csv

# 2. Evaluate it and write the report into out/
uv run cropeval report data.csv -o out/
```

`cropeval report` writes:

| file                     | content                                                     |
|--------------------------|-------------------------------------------------------------|
| `metrics.json`           | per-domain accuracy, macro F1, per-class metrics, confusion matrices, accuracy gap with CI |
| `report.md`              | the same results as a short readable report                 |
| `confusion_matrices.png` | lab and field confusion matrices side by side               |
| `f1_by_class.png`        | per-class F1 for both domains                               |

Useful options: `demo --n-lab N --n-field N --seed S`,
`report --n-boot N --seed S`. Run `uv run cropeval --help` for all of them.

## Example output

The [`examples/`](examples/) folder contains a synthetic CSV and the report
generated from it:

```bash
uv run cropeval demo -o examples/demo_predictions.csv
uv run cropeval report examples/demo_predictions.csv -o examples/report
```

```text
Wrote 800 synthetic predictions to examples/demo_predictions.csv
Accuracy gap (lab - field): +0.200 [+0.147, +0.250]
Report written to examples/report/
```

| domain | n   | accuracy | macro F1 |
|--------|----:|---------:|---------:|
| lab    | 400 | 0.927    | 0.926    |
| field  | 400 | 0.728    | 0.727    |

**Accuracy gap (lab − field): +0.200**, 95% bootstrap CI [+0.147, +0.250].
The interval does not include 0, so the drop on field images is not just noise.

![Confusion matrices for lab and field](examples/report/confusion_matrices.png)

![Per-class F1 for lab and field](examples/report/f1_by_class.png)

Full report: [`examples/report/report.md`](examples/report/report.md).

## How the confidence interval is computed

The gap is `accuracy(lab) − accuracy(field)`. For the interval, lab rows and
field rows are each resampled with replacement (keeping their sizes), the gap
is recomputed, and this is repeated `n_boot` times. The 2.5% and 97.5%
percentiles of these gaps give the 95% interval. The random generator is
`numpy.random.default_rng(seed)`, so results are reproducible.

## Features

- [x] Per-domain metrics (accuracy, precision, recall, macro-F1, confusion
      matrix) implemented with NumPy and cross-checked against scikit-learn
- [x] Domain gap (`lab − field`) with a seeded bootstrap confidence interval
- [x] `cropeval` command-line tool
- [x] Synthetic demo data generator (no real research data in the repo)
- [x] Plots and a short evaluation report
- [x] CI (lint + tests) on GitHub Actions
- [ ] Release pipeline on GitHub Actions

## License

[MIT](LICENSE) © 2026 Nurshapagat Shapay
