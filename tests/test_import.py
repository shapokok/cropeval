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
