# cropeval

[![CI](https://github.com/shapokok/cropeval/actions/workflows/ci.yml/badge.svg)](https://github.com/shapokok/cropeval/actions/workflows/ci.yml)

> Status: early scaffold — the API is not implemented yet.

**cropeval** is a small Python package for evaluating the **lab-to-field
domain gap** of image classification models for plant disease detection.

Models trained on clean laboratory photos often perform worse on images
taken in the field. cropeval takes a model's predictions from both settings
and answers: *how big is the drop, and how sure are we about it?*

## Input

A CSV file with one row per image:

| column   | meaning                              |
|----------|--------------------------------------|
| `y_true` | true class label                     |
| `y_pred` | predicted class label                |
| `domain` | where the image came from: `lab` or `field` |

## Installation

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/shapokok/cropeval.git
cd cropeval
uv sync            # creates .venv with runtime and dev dependencies
uv run pytest      # run the test suite
```

## Planned features

- [x] Per-domain metrics (accuracy, precision, recall, macro-F1, confusion
      matrix) implemented with NumPy and cross-checked against scikit-learn
- [x] Domain gap (`lab − field`) with a seeded bootstrap confidence interval
- [ ] `cropeval` command-line tool
- [ ] Synthetic demo data generator (no real research data in the repo)
- [ ] Plots and a short evaluation report
- [x] CI (lint + tests) on GitHub Actions
- [ ] Release pipeline on GitHub Actions

## License

[MIT](LICENSE) © 2026 Nurshapagat Shapay
