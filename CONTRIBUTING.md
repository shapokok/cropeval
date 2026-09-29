# Contributing to cropeval

Thanks for your interest! cropeval is a small teaching project, so the main
goal is code that is **simple, readable and well commented**. Please keep
changes small and easy to review.

## Setup

You need Python >= 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/shapokok/cropeval.git
cd cropeval
uv sync
```

## Workflow

1. **Open or pick an issue** describing the change.
2. **Create a branch** from `main` (never commit to `main` directly):

   ```bash
   git switch main && git pull
   git switch -c feature/short-name   # or fix/..., docs/..., ci/...
   ```

3. **Make small, meaningful commits** using
   [Conventional Commits](https://www.conventionalcommits.org/):

   | prefix      | use for                           |
   |-------------|-----------------------------------|
   | `feat:`     | new functionality                 |
   | `fix:`      | bug fixes                         |
   | `test:`     | adding or changing tests only     |
   | `docs:`     | README, docstrings, comments      |
   | `ci:`       | GitHub Actions workflows          |
   | `refactor:` | code changes without new behavior |
   | `chore:`    | tooling, dependencies, config     |

   A scope is optional, e.g. `feat(metrics): add balanced accuracy`.

4. **Run the checks** before every commit (the same ones CI runs):

   ```bash
   uv run ruff check . && uv run ruff format --check . && uv run pytest
   ```

5. **Open a pull request** and reference the issue in the description
   (`Closes #N`):

   ```bash
   git push -u origin feature/short-name
   gh pr create
   ```

6. **Merge only when CI is green**, with a squash merge:

   ```bash
   gh pr checks --watch
   gh pr merge --squash --delete-branch
   ```

Please do not rewrite shared history (no force-push to `main`).

## Code rules

- **Metrics use NumPy.** Every metric needs a test with a small example
  computed by hand *and* a comparison against scikit-learn (scikit-learn is
  a dev dependency only, never imported by the package itself).
- **Randomness is always seeded** with `numpy.random.default_rng(seed)`, and
  the seed is a parameter of the function.
- **Plots use the Agg backend**, so they work in CI and on servers.
- **No real research data** in the repository. Examples and tests use only
  synthetic data from `cropeval.demo`.
- Code, comments, docstrings and documentation are in **English**.
- Test coverage must stay at or above **80%** (CI fails otherwise).

## Adding a dependency

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # development-only dependency
```

Commit both `pyproject.toml` and `uv.lock`, and explain in the PR why the
dependency is needed. The project deliberately stays light: no deep
learning frameworks, no GPU.

## Reporting bugs

Open an issue with the command you ran, the full error message, and, if
possible, a small synthetic CSV that reproduces the problem.
