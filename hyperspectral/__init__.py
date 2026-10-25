"""Hyperspectral image processing toolkit.

Algorithms for hyperspectral remote-sensing imagery in pure Python:

* loading of MATLAB/NumPy data cubes (:mod:`hyperspectral.io`)
* random training/test sample selection, by number or percentage
  (:mod:`hyperspectral.split`)
* spectral distances and angles (:mod:`hyperspectral.distances`,
  :mod:`hyperspectral.angles`)
* band selection with Maximum Tangent Discrimination
  (:mod:`hyperspectral.features`)
* Maximum Likelihood Classification (:mod:`hyperspectral.mlc`)
* K-means clustering (:mod:`hyperspectral.cluster`)
* accuracy assessment: confusion matrix, OA, AA and kappa
  (:mod:`hyperspectral.metrics`)
* figures (:mod:`hyperspectral.plotting`)

Example
-------
>>> from hyperspectral import load_hypercube, split_samples, MaximumLikelihoodClassifier
>>> cube = load_hypercube("data/Indian_Pine.mat")            # doctest: +SKIP
>>> split = split_samples(cube.dataset(), cube.ground_truth,
...                       samples=100, seed=0)              # doctest: +SKIP
>>> clf = MaximumLikelihoodClassifier(cube.shape[:2]).fit(
...     split.train_dataset, split.train_counts, split.labels)  # doctest: +SKIP
>>> clf.score(split.test_map, cube.dataset())["overall_accuracy"]  # doctest: +SKIP
"""

from __future__ import annotations

from .angles import angles_between_vectors
from .cluster import KMeansResult, cluster_map, kmeans
from .datasets import class_dataset, class_means, cumulative_ranges
from .distances import pairwise_distances
from .features import maximum_tangent_discrimination, select_bands
from .io import Hypercube, load_hypercube, save_hypercube, to_cube, to_dataset
from .metrics import (
    average_accuracy,
    confusion_matrix,
    evaluate,
    kappa_coefficient,
    overall_accuracy,
    per_class_accuracy,
)
from .mlc import MaximumLikelihoodClassifier
from .split import SplitResult, select_samples, split_samples

__version__ = "1.0.0"
__author__ = "Mohsen Ghamary Asl"
__license__ = "MIT"

__all__ = [
    "Hypercube",
    "load_hypercube",
    "save_hypercube",
    "to_dataset",
    "to_cube",
    "split_samples",
    "select_samples",
    "SplitResult",
    "cumulative_ranges",
    "class_dataset",
    "class_means",
    "pairwise_distances",
    "angles_between_vectors",
    "maximum_tangent_discrimination",
    "select_bands",
    "MaximumLikelihoodClassifier",
    "kmeans",
    "cluster_map",
    "KMeansResult",
    "confusion_matrix",
    "overall_accuracy",
    "average_accuracy",
    "per_class_accuracy",
    "kappa_coefficient",
    "evaluate",
    "__version__",
    "__author__",
    "__license__",
]
