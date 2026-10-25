"""Distances between sets of vectors.

Every function in this module takes two 2-D arrays whose **rows are the
vectors** (``(n_vectors, n_features)``) and returns the full ``(n_X, n_Y)``
matrix of pairwise distances/angles, exactly like the original
``distancesBtwVectors`` / ``anglesBtwVectors`` helpers - but vectorised with
NumPy instead of Python loops.

Examples
--------
>>> import numpy as np
>>> from hyperspectral.distances import pairwise_distances
>>> X = np.array([[0.0, 0.0], [3.0, 4.0]])
>>> pairwise_distances(X)
array([[0., 5.],
       [5., 0.]])
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["pairwise_distances"]

SUPPORTED_METRICS = (
    "euclidean",
    "sqeuclidean",
    "cityblock",
    "chebyshev",
    "cosine",
)


def _as_vectors(array: np.ndarray, name: str) -> np.ndarray:
    """Return *array* as a 2-D float array of row vectors."""
    vectors = np.atleast_2d(np.asarray(array, dtype=float))
    if vectors.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {vectors.shape}")
    return vectors


def pairwise_distances(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    metric: str = "euclidean",
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Pairwise distances between the rows of ``X`` and the rows of ``Y``.

    Parameters
    ----------
    X, Y : array_like
        ``(n_X, n_features)`` and ``(n_Y, n_features)``. When ``Y`` is ``None``
        the distance of every row of ``X`` to every other row of ``X`` is
        computed.
    metric : {"euclidean", "sqeuclidean", "cityblock", "chebyshev", "cosine"}
        Distance measure to use.
    eps : float
        Guard used by the ``cosine`` metric to avoid a division by zero for
        all-zero vectors.

    Returns
    -------
    numpy.ndarray
        ``(n_X, n_Y)`` matrix of distances.

    Raises
    ------
    ValueError
        If the number of features differ or the metric is unknown.
    """
    x = _as_vectors(X, "X")
    y = x if Y is None else _as_vectors(Y, "Y")

    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"X and Y must have the same number of features, "
            f"got {x.shape[1]} and {y.shape[1]}"
        )
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"unknown metric {metric!r}, choose from {', '.join(SUPPORTED_METRICS)}"
        )

    if metric in ("euclidean", "sqeuclidean"):
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y  (fast matrix product form)
        x_sq = np.einsum("ij,ij->i", x, x)[:, None]
        y_sq = np.einsum("ij,ij->i", y, y)[None, :]
        distances = x_sq + y_sq - 2.0 * (x @ y.T)
        np.maximum(distances, 0.0, out=distances)  # numerical noise
        return distances if metric == "sqeuclidean" else np.sqrt(distances)

    if metric == "cityblock":
        return np.abs(x[:, None, :] - y[None, :, :]).sum(axis=2)

    if metric == "chebyshev":
        return np.abs(x[:, None, :] - y[None, :, :]).max(axis=2)

    # cosine distance
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    denominator = np.maximum(x_norm @ y_norm.T, eps)
    return 1.0 - (x @ y.T) / denominator
