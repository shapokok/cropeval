"""cropeval: evaluate the lab-to-field domain gap of image classifiers.

The package reads predictions of a plant disease classifier (a CSV with
columns ``y_true``, ``y_pred`` and ``domain``), computes metrics separately
for the ``lab`` and ``field`` domains, and quantifies the gap between them
with a bootstrap confidence interval.
"""

__version__ = "0.1.0"
