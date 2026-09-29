"""Smoke test: the package and all its modules can be imported."""

import importlib

import pytest

MODULES = ["metrics", "gap", "io", "plots", "demo", "cli"]


def test_package_has_version():
    import cropeval

    assert isinstance(cropeval.__version__, str)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"cropeval.{name}")


def test_version_matches_pyproject():
    import tomllib
    from pathlib import Path

    import cropeval

    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    with pyproject.open("rb") as f:
        assert cropeval.__version__ == tomllib.load(f)["project"]["version"]
