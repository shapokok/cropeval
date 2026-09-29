"""cropeval: evaluate the lab-to-field domain gap of image classifiers.

The package reads predictions of a plant disease classifier (a CSV with
columns ``y_true``, ``y_pred`` and ``domain``), computes metrics separately
for the ``lab`` and ``field`` domains, and quantifies the gap between them
with a bootstrap confidence interval.
"""

from importlib.metadata import version

# The version is defined once, in pyproject.toml, and read from the
# installed package metadata here, so the two can never disagree.
__version__ = version("cropeval")
