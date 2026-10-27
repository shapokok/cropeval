# syntax=docker/dockerfile:1
#
# Multi-stage image for the cropeval CLI.
#
#   docker build --target test -t shapok/cropeval:test .   # lint + tests
#   docker build -t shapok/cropeval:dev .                  # runtime image
#
# Stages: base (dependencies from uv.lock) -> test (checks) and
#         base -> runtime (only the virtual environment, non-root user).

# --------------------------------------------------------------------------
# base: Python + uv, runtime dependencies and the cropeval package
# --------------------------------------------------------------------------
FROM python:3.12-slim AS base

# Copy the uv binaries from the official image (pinned version).
COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    MPLBACKEND=Agg

WORKDIR /app

# 1) Dependencies only. This layer is rebuilt only when pyproject.toml or
#    uv.lock change, not on every code change. --locked fails if uv.lock
#    is out of date; --no-dev skips pytest, ruff and scikit-learn.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# 2) The package itself. README.md and LICENSE are needed by the build
#    (pyproject.toml references them). --no-editable copies the package
#    into the venv, so the venv works without /app/src.
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# --------------------------------------------------------------------------
# test: add dev dependencies and run the same checks as CI
# --------------------------------------------------------------------------
FROM base AS test

RUN uv sync --locked --no-editable

COPY tests ./tests
RUN uv run --no-sync ruff check . \
 && uv run --no-sync ruff format --check . \
 && uv run --no-sync pytest

# --------------------------------------------------------------------------
# runtime: clean Python image + the venv from base, non-root user
# --------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Unprivileged user with a fixed UID (handy for file permissions on mounts).
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin cropeval

# Only the virtual environment: no uv, no sources, no tests, no dev tools.
COPY --from=base /opt/venv /opt/venv

# MPLCONFIGDIR: matplotlib needs a writable cache folder, also when the
# container is started with another UID (docker run --user ...).
ENV PATH="/opt/venv/bin:$PATH" \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER cropeval
WORKDIR /work

ENTRYPOINT ["cropeval"]
CMD ["--help"]
