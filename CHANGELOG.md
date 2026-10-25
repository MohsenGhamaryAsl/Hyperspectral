# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-25

First release of the packaged version of the toolkit.

### Added

* `hyperspectral` package with a documented, type-annotated API:
  `io`, `datasets`, `split`, `distances`, `angles`, `features`, `mlc`,
  `cluster`, `metrics`, `plotting` and `cli`.
* `Hypercube` object that loads MATLAB `.mat`, NumPy `.npz`/`.npy` cubes with
  their ground truth and class names.
* Random training/test sample selection by number or by percentage of each
  class, with reproducible seeds and optional skipping of small classes.
* Maximum Tangent Discrimination band selection in a numerically stable
  log-domain formulation.
* `MaximumLikelihoodClassifier` with regularised covariances, optional priors,
  log-likelihood output and one-call `score()`.
* K-means clustering with k-means++ initialisation, empty-cluster handling and
  a convergence history.
* Accuracy metrics: confusion matrix, OA, AA, Cohen's kappa and per-class
  accuracy.
* Command line interface (`python -m hyperspectral`) that writes figures and a
  JSON report.
* Five deterministic examples, a 36-test pytest suite and a CI workflow for
  Linux, macOS and Windows.
* `pyproject.toml`, `CITATION.cff`, `CONTRIBUTING.md` and this changelog.

### Changed

* The distance and angle computations are vectorised with NumPy; they are no
  longer Python loops.
* `MaximumLikelihoodClassifier` uses `slogdet`/`pinv` and a regularisation
  term instead of the ad-hoc covariance scaling of the old scripts.
* Accuracy is measured only on labelled test pixels, so background pixels no
  longer influence the result.
* The data files moved to `data/`.

### Removed

* The top-level scripts `MLC.py`, `MakeTT.py`, `makeDataset.py`,
  `classDataset.py`, `StartEnd.py`, `K_Means.py`, `anglesBtwVectors.py`,
  `distancesBtwVectors.py`, `MaximumTangentDiscrimination.py` and
  `nargoutController.py`. They are superseded by the package (see the
  migration table in the README) and remain available in the git history.

## [0.1.0] - 2019-08-25

* Initial scripts: sample selection, MLC, K-means, MTD band selection and the
  vector geometry helpers for the Indian Pines scene.
