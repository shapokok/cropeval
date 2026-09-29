# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-29

First release.

### Added

- Loading and validation of the predictions CSV (`y_true`, `y_pred`,
  `domain`) with clear error messages that name the CSV line numbers (#5).
- NumPy metrics: accuracy, per-class precision / recall / F1, macro F1 and
  confusion matrix, cross-checked against scikit-learn in the tests (#5).
- Lab − field accuracy gap with a seeded, stratified percentile bootstrap
  confidence interval (`n_boot`, `seed`, `confidence`) (#5).
- Synthetic demo data generator with plant disease classes (#7).
- Plots: lab and field confusion matrices side by side, per-class F1 bar
  chart (#7).
- `cropeval` CLI: `cropeval demo -o data.csv` and
  `cropeval report data.csv -o out/` writing `metrics.json`, `report.md`
  and PNG plots (#7).
- Synthetic example data and report in `examples/` (#7).
- CI on GitHub Actions: ruff lint/format and tests on Python 3.11 and 3.12
  with coverage ≥ 80% (#6).
- Release workflow: on a `v*` tag, run the checks, build sdist and wheel,
  and publish a GitHub Release (#4).
- README sections on usage, development and technology choices;
  CONTRIBUTING.md (#8).

[Unreleased]: https://github.com/shapokok/cropeval/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shapokok/cropeval/releases/tag/v0.1.0
