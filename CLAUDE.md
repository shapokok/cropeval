# cropeval

## Purpose
Small Python package for evaluating the lab-to-field domain gap of image
classification models (plant disease detection). Input: CSV with columns
`y_true`, `y_pred`, `domain` (values: `lab`, `field`). Output: per-domain
metrics, domain gap with bootstrap confidence interval, plots, report.
Part of a master's course assignment; the author must be able to explain
every file, so keep code simple, readable, and well commented.

## Stack
- Python >= 3.11, managed with uv; build backend: hatchling; src layout
- Runtime deps: numpy, pandas, matplotlib (no PyTorch, no GPU)
- Dev deps: pytest, pytest-cov, ruff, scikit-learn (only for cross-check tests)
- CLI entry point: `cropeval = "cropeval.cli:main"` (argparse)
- CI/CD: GitHub Actions with astral-sh/setup-uv

## Layout
src/cropeval/{__init__,metrics,gap,io,plots,demo,cli}.py
tests/test_*.py
examples/ (only synthetic data)
.github/workflows/{ci,release}.yml

## Rules
- Metrics implemented with NumPy; tests compare against hand-computed
  small examples AND scikit-learn.
- All randomness uses an explicit seed (numpy.random.default_rng(seed)).
- Matplotlib must use the Agg backend in tests and CLI.
- Never commit real research data; only synthetic data from cropeval.demo.
- Before every commit: `uv run ruff check . && uv run ruff format --check . && uv run pytest`.
- Conventional Commits (feat:, fix:, test:, docs:, ci:, chore:, refactor:).
- Small, meaningful commits. After the initial scaffold, never commit to
  main directly: create a branch, open a PR with `gh pr create`, reference
  the issue ("Closes #N"), then merge with `gh pr merge --squash --delete-branch`
  only after CI is green.
- Do not rewrite git history or change commit dates.
- README and code comments in English.
- After each task, give a short summary of what each changed file does.