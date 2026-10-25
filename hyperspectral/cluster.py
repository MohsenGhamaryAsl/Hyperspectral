"""K-means clustering for hyperspectral data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from .distances import pairwise_distances

__all__ = ["KMeansResult", "kmeans", "cluster_map"]


@dataclass
class KMeansResult:
    """Result of :func:`kmeans`.

    Attributes
    ----------
    labels : numpy.ndarray
        Cluster index (``1..k``) of every sample.
    centers : numpy.ndarray
        ``(k, n_features)`` final cluster centres.
    inertia : float
        Sum of the squared distance of every sample to its centre.
    n_iter : int
        Number of iterations actually performed.
    history : list of float
        Inertia after every iteration (useful to plot the convergence).
    """

    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: int
    history: list

    @property
    def n_clusters(self) -> int:
        return int(self.centers.shape[0])


def _kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Choose ``k`` well-spread initial centres (k-means++)."""
    n_samples = X.shape[0]
    centers = [X[rng.integers(n_samples)]]
    closest = pairwise_distances(X, centers[0][None, :], "sqeuclidean").ravel()

    for _ in range(1, k):
        total = closest.sum()
        if total <= 0:
            # All points coincide with the chosen centres: pick any other point.
            centers.append(X[rng.integers(n_samples)])
        else:
            probabilities = closest / total
            centers.append(X[rng.choice(n_samples, p=probabilities)])
        distance = pairwise_distances(X, centers[-1][None, :], "sqeuclidean").ravel()
        closest = np.minimum(closest, distance)
    return np.vstack(centers)


def kmeans(
    data: np.ndarray,
    n_clusters: int = 2,
    max_iter: int = 100,
    tol: float = 1e-6,
    *,
    seed: Optional[int] = None,
    init: Union[str, np.ndarray] = "k-means++",
    verbose: bool = False,
) -> KMeansResult:
    """K-means clustering of the rows of *data*.

    Parameters
    ----------
    data : numpy.ndarray
        ``(n_samples, n_features)`` matrix (one pixel per row).
    n_clusters : int
        Number of clusters ``k``.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence threshold on the relative change of the inertia.
    seed : int, optional
        Seed of the random generator (reproducible results).
    init : {"k-means++", "random"} or numpy.ndarray
        Initialisation strategy, or explicit ``(k, n_features)`` centres.
    verbose : bool
        Print the inertia of every iteration.

    Returns
    -------
    KMeansResult

    Examples
    --------
    >>> import numpy as np
    >>> from hyperspectral.cluster import kmeans
    >>> rng = np.random.default_rng(0)
    >>> data = np.vstack([rng.normal(0, .05, (30, 2)), rng.normal(5, .05, (30, 2))])
    >>> result = kmeans(data, 2, seed=0)
    >>> sorted(np.unique(result.labels).tolist())
    [1, 2]
    """
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"data must be 2-D (samples, features), got {matrix.shape}")
    n_samples = matrix.shape[0]
    k = int(n_clusters)
    if not 1 <= k <= n_samples:
        raise ValueError(f"n_clusters must be between 1 and {n_samples}, got {k}")

    rng = np.random.default_rng(seed)

    if isinstance(init, str):
        if init == "k-means++":
            centers = _kmeans_plus_plus(matrix, k, rng)
        elif init == "random":
            centers = matrix[rng.choice(n_samples, size=k, replace=False)].copy()
        else:
            raise ValueError("init must be 'k-means++', 'random' or an array")
    else:
        centers = np.asarray(init, dtype=float).copy()
        if centers.shape != (k, matrix.shape[1]):
            raise ValueError(
                f"init centres must have shape {(k, matrix.shape[1])}, "
                f"got {centers.shape}"
            )

    labels = np.zeros(n_samples, dtype=int)
    history = []
    inertia = np.inf
    n_iter = 0

    for iteration in range(1, int(max_iter) + 1):
        distances = pairwise_distances(matrix, centers, "sqeuclidean")
        new_labels = np.argmin(distances, axis=1)
        new_inertia = float(distances[np.arange(n_samples), new_labels].sum())

        # Empty clusters: move them to the points that are the worst fitted.
        counts = np.bincount(new_labels, minlength=k)
        if np.any(counts == 0):
            worst = np.argsort(-distances[np.arange(n_samples), new_labels])[: int((counts == 0).sum())]
            for position, sample in zip(np.flatnonzero(counts == 0), worst):
                new_labels[sample] = position
            distances = pairwise_distances(matrix, centers, "sqeuclidean")
            new_inertia = float(distances[np.arange(n_samples), new_labels].sum())

        for cluster in range(k):
            members = matrix[new_labels == cluster]
            if members.size:
                centers[cluster] = members.mean(axis=0)

        labels = new_labels
        history.append(new_inertia)
        n_iter = iteration
        if verbose:
            print(f"iteration {iteration:3d}  inertia {new_inertia:.6g}")

        converged = abs(inertia - new_inertia) <= tol * max(abs(inertia), 1.0)
        inertia = new_inertia
        if converged and iteration > 1:
            break

    return KMeansResult(
        labels=labels + 1,          # labels start at 1, as in the original code
        centers=centers,
        inertia=inertia,
        n_iter=n_iter,
        history=history,
    )


def cluster_map(
    data: np.ndarray,
    shape: Sequence[int],
    n_clusters: int = 2,
    **kwargs,
) -> np.ndarray:
    """Cluster a ``(rows * columns, features)`` matrix and return a 2-D map.

    Parameters
    ----------
    data : numpy.ndarray
        One pixel per row.
    shape : (rows, columns)
        Spatial shape of the image.
    n_clusters : int
        Number of clusters, forwarded to :func:`kmeans`.
    **kwargs
        Forwarded to :func:`kmeans` (``seed``, ``max_iter``, ``init``, ...).

    Returns
    -------
    numpy.ndarray
        ``(rows, columns)`` map of cluster indices (``1..k``).
    """
    rows, columns = int(shape[0]), int(shape[1])
    if np.shape(data)[0] != rows * columns:
        raise ValueError(
            f"data has {np.shape(data)[0]} rows but the image shape {shape} "
            f"needs {rows * columns}"
        )
    result = kmeans(data, n_clusters, **kwargs)
    return result.labels.reshape(rows, columns)
