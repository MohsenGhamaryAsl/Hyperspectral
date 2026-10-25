"""Maximum Likelihood Classification (Gaussian, per class)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .datasets import cumulative_ranges
from .metrics import evaluate

__all__ = ["MaximumLikelihoodClassifier"]


@dataclass
class _ClassStatistics:
    """Gaussian parameters of one class."""

    label: int
    mean: np.ndarray
    precision: np.ndarray           # inverse of the covariance matrix
    log_determinant: float          # ln |cov|
    log_prior: float
    n_samples: int


@dataclass
class MaximumLikelihoodClassifier:
    """Gaussian maximum-likelihood classifier for hyperspectral pixels.

    Each class is modelled by a multivariate normal distribution estimated
    from its training samples; a pixel is assigned to the class with the
    highest *a posteriori* probability (equal priors are used when
    ``priors=None`` and ``use_sample_priors=False``).

    Parameters
    ----------
    shape : (rows, columns)
        Spatial shape of the image, used by :meth:`predict_map`.
    regularization : float
        Value added to the diagonal of every covariance matrix before
        inversion. Keeps the estimate stable when a class has fewer samples
        than bands (``1e-6`` by default).
    use_sample_priors : bool
        Use the training class frequencies as prior probabilities. Otherwise
        the priors are uniform. Sample priors ``p_i`` can also be given
        explicitly with :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> from hyperspectral.mlc import MaximumLikelihoodClassifier
    >>> rng = np.random.default_rng(0)
    >>> train = np.hstack([rng.normal(0, .1, (3, 40)), rng.normal(5, .1, (3, 40))])
    >>> labels = [1, 2]
    >>> clf = MaximumLikelihoodClassifier(shape=(1, 6)).fit(train, [40, 40], labels)
    >>> clf.predict(np.array([[0.0], [0.0], [0.0]]))[0]
    array([1, 1, 1])
    """

    shape: Tuple[int, int]
    regularization: float = 1e-6
    use_sample_priors: bool = False
    labels: List[int] = field(default_factory=list, init=False)
    statistics: List[_ClassStatistics] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.shape = (int(self.shape[0]), int(self.shape[1]))
        if self.regularization < 0:
            raise ValueError("regularization must be non-negative")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        train_dataset: np.ndarray,
        train_counts: Sequence[int],
        labels: Optional[Sequence[int]] = None,
        *,
        priors: Optional[Sequence[float]] = None,
    ) -> "MaximumLikelihoodClassifier":
        """Estimate one Gaussian per class.

        Parameters
        ----------
        train_dataset : numpy.ndarray
            ``(bands, n_samples)`` matrix with the classes stored as
            consecutive blocks of columns.
        train_counts : sequence of int
            Number of columns of each block (one per class).
        labels : sequence, optional
            Class label of every block (default ``1, 2, ...``).
        priors : sequence of float, optional
            Explicit prior probability of every class. They are normalised
            internally. When omitted, ``use_sample_priors`` decides between
            sample frequencies and uniform priors.

        Returns
        -------
        MaximumLikelihoodClassifier
            ``self``, so calls can be chained.
        """
        matrix = np.asarray(train_dataset, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("train_dataset must be 2-D (bands, samples)")

        counts = [int(c) for c in np.atleast_1d(train_counts)]
        starts, ends = cumulative_ranges(counts)

        if labels is None:
            labels = list(range(1, len(counts) + 1))
        labels = [int(label) for label in labels]
        if len(labels) != len(counts):
            raise ValueError(
                f"got {len(labels)} labels for {len(counts)} class blocks"
            )
        if ends[-1] != matrix.shape[1]:
            raise ValueError(
                f"the train dataset has {matrix.shape[1]} columns but the "
                f"class counts sum to {int(ends[-1])}"
            )

        if priors is not None:
            prior_values = np.asarray(priors, dtype=float)
            if prior_values.size != len(counts):
                raise ValueError("one prior per class is required")
        elif self.use_sample_priors:
            prior_values = np.asarray(counts, dtype=float)
        else:
            prior_values = np.ones(len(counts), dtype=float)
        prior_values = prior_values / prior_values.sum()

        stats: List[_ClassStatistics] = []
        for index, (label, start, end) in enumerate(zip(labels, starts, ends)):
            block = matrix[:, start:end]
            n_samples, n_bands = block.shape[1], block.shape[0]
            if n_samples < 2:
                raise ValueError(
                    f"class {label} has {n_samples} training sample(s); "
                    "at least 2 are required to estimate a covariance"
                )
            mean = block.mean(axis=1)
            covariance = np.atleast_2d(np.cov(block, rowvar=True, bias=False))
            covariance = np.asarray(covariance, dtype=float)
            covariance.flat[:: n_bands + 1] += self.regularization

            sign, log_determinant = np.linalg.slogdet(covariance)
            if not np.isfinite(log_determinant):
                raise np.linalg.LinAlgError(
                    f"the covariance of class {label} is singular; "
                    "increase regularization"
                )
            precision = np.linalg.pinv(covariance, hermitian=True)

            stats.append(
                _ClassStatistics(
                    label=label,
                    mean=mean,
                    precision=precision,
                    log_determinant=float(log_determinant),
                    log_prior=float(np.log(max(prior_values[index], 1e-300))),
                    n_samples=n_samples,
                )
            )

        self.labels = labels
        self.statistics = stats
        return self

    def _check_fitted(self) -> None:
        if not self.statistics:
            raise RuntimeError("call fit() before predicting")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def log_likelihood(self, dataset: np.ndarray) -> np.ndarray:
        """``(n_classes, n_pixels)`` matrix of log posterior probabilities."""
        self._check_fitted()
        matrix = np.asarray(dataset, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("dataset must be 2-D (bands, samples)")
        if matrix.shape[0] != self.statistics[0].mean.size:
            raise ValueError(
                f"expected {self.statistics[0].mean.size} bands, "
                f"got {matrix.shape[0]}"
            )

        scores = np.empty((len(self.statistics), matrix.shape[1]), dtype=float)
        n_bands = matrix.shape[0]
        log_2pi = np.log(2.0 * np.pi)
        for index, stat in enumerate(self.statistics):
            centred = matrix - stat.mean[:, None]
            # Mahalanobis distance: sum_i c_i * (P c)_i
            quadratic = np.sum(centred * (stat.precision @ centred), axis=0)
            scores[index] = (
                -0.5 * quadratic
                - 0.5 * stat.log_determinant
                + stat.log_prior
                - 0.5 * n_bands * log_2pi
            )
        return scores

    def predict(self, dataset: np.ndarray) -> np.ndarray:
        """Class label of every column of *dataset*."""
        scores = self.log_likelihood(dataset)
        winners = np.argmax(scores, axis=0)
        return np.asarray(self.labels, dtype=np.int32)[winners]

    def predict_map(self, dataset: np.ndarray) -> np.ndarray:
        """Classification map ``(rows, columns)`` of a ``(bands, N)`` dataset."""
        if np.shape(dataset)[1] != self.shape[0] * self.shape[1]:
            raise ValueError(
                f"dataset has {np.shape(dataset)[1]} pixels but the image "
                f"shape {self.shape} needs {self.shape[0] * self.shape[1]}"
            )
        return self.predict(dataset).reshape(self.shape)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def score(
        self,
        reference: np.ndarray,
        dataset: np.ndarray,
        *,
        ignore: Sequence[int] = (0,),
        class_names: Optional[Union[Mapping[int, str], Sequence[str]]] = None,
    ) -> Dict[str, object]:
        """Accuracy figures of the prediction on labelled pixels only.

        ``reference`` is typically the test-sample map of
        :class:`hyperspectral.split.SplitResult`, so that the reported
        accuracy is measured on pixels the classifier has never seen.
        """
        predicted = self.predict(dataset).reshape(np.asarray(reference).shape)
        return evaluate(
            reference,
            predicted,
            ignore=ignore,
            labels=self.labels,
            class_names=class_names,
        )

    def describe(self) -> List[Dict[str, object]]:
        """Model parameters as a list of dictionaries (one per class)."""
        self._check_fitted()
        return [
            {
                "label": stat.label,
                "n_samples": stat.n_samples,
                "log_prior": stat.log_prior,
                "log_determinant": stat.log_determinant,
                "mean": stat.mean.copy(),
            }
            for stat in self.statistics
        ]
