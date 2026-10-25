"""Band selection with the Maximum Tangent Discrimination (MTD) criterion.

Reference
---------
M. Imani and H. Ghassemian, "Band Clustering-based Feature Extraction for
Hyperspectral Images Classification", IEEE IGARSS 2013.
https://ieeexplore.ieee.org/document/6587529

Two bands whose mean signatures point in almost the same direction are highly
redundant, so the tangent of the angle between them is small. MTD starts with
the band whose total tangent to all the other bands is the largest, then keeps
adding the band that maximises the product of the tangents with the bands
already selected, and stops as soon as the band most similar to the first one
is reached (i.e. when the selection becomes redundant).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .angles import angles_between_vectors

__all__ = ["maximum_tangent_discrimination", "select_bands"]

_MIN_LOG = np.log(1e-12)


def _tangent_matrix(signatures: np.ndarray, eps: float) -> np.ndarray:
    """Tangent of the angles between every pair of signatures (rows)."""
    angles = angles_between_vectors(signatures, signatures, unit="rad")
    tangents = np.tan(np.clip(angles, 0.0, np.pi / 2 - eps))
    np.fill_diagonal(tangents, 0.0)
    return tangents


def maximum_tangent_discrimination(
    class_dataset: np.ndarray,
    *,
    max_features: Optional[int] = None,
    eps: float = 1e-9,
    log_tangents: bool = True,
) -> List[int]:
    """Select bands with the maximum tangent discrimination criterion.

    Parameters
    ----------
    class_dataset : numpy.ndarray
        ``(bands, n_classes)`` matrix of class-mean spectra (see
        :func:`hyperspectral.datasets.class_dataset`). A ``(bands, samples)``
        dataset also works, but class means are recommended because they are
        far less noisy.
    max_features : int, optional
        Never return more than this number of bands (a sensible value is
        ``min(train_counts) - 1`` so that every class covariance stays
        invertible). ``None`` means "let MTD decide".
    eps : float
        Margin used to avoid ``tan(pi/2)`` overflows.
    log_tangents : bool
        Work with the logarithms of the tangents (default). This is
        mathematically identical to the product of the original formulation
        but cannot overflow, which makes the selection reproducible.

    Returns
    -------
    list of int
        Selected band indices, best first.

    Examples
    --------
    >>> import numpy as np
    >>> from hyperspectral.features import maximum_tangent_discrimination
    >>> rng = np.random.default_rng(0)
    >>> signatures = rng.normal(size=(12, 4))
    >>> features = maximum_tangent_discrimination(signatures)
    >>> 1 <= len(features) <= 11
    True
    """
    signatures = np.atleast_2d(np.asarray(class_dataset, dtype=float))
    n_bands = signatures.shape[0]
    if n_bands < 3:
        raise ValueError("at least three bands are required to select features")
    if max_features is not None and int(max_features) < 1:
        raise ValueError("max_features must be positive")

    tangents = _tangent_matrix(signatures, eps)

    # 1) The band with the largest total tangent to all the other bands.
    selected = [int(np.argmax(tangents.sum(axis=0)))]

    if log_tangents:
        scores = np.log(np.maximum(tangents, np.exp(_MIN_LOG)))
        np.fill_diagonal(scores, 0.0)
    else:  # pragma: no cover - kept for numerical comparison
        scores = tangents.copy()

    # 2) Stop point: the band most similar to the first selected one (the
    #    smallest non-zero tangent), as in the original formulation.
    similarity = tangents[selected[0]].copy()
    similarity[selected[0]] = np.inf
    stop_point = int(np.argmin(similarity))

    running = scores[selected[0]].copy()
    running[selected] = -np.inf

    guard = 0
    while selected[-1] != stop_point and len(selected) < n_bands - 1:
        candidate = int(np.argmax(running))
        if not np.isfinite(running[candidate]):
            break                       # nothing selectable left
        selected.append(candidate)
        if log_tangents:
            running = running + scores[candidate]
        else:  # pragma: no cover
            running = running * scores[candidate]
        running[selected] = -np.inf
        guard += 1
        if guard > n_bands + 5:         # pragma: no cover - safety net
            break

    # The stop point itself is not part of the result.
    if len(selected) > 1 and selected[-1] == stop_point:
        selected = selected[:-1]

    if max_features is not None:
        selected = selected[: int(max_features)]
    return selected


def select_bands(dataset: np.ndarray, features) -> np.ndarray:
    """Return the rows of *dataset* listed in *features* (band selection)."""
    indices = np.atleast_1d(np.asarray(features, dtype=int))
    matrix = np.asarray(dataset)
    if indices.size == 0:
        raise ValueError("the feature list is empty")
    if indices.max() >= matrix.shape[0] or indices.min() < 0:
        raise ValueError(
            f"feature indices out of range for a dataset with {matrix.shape[0]} bands"
        )
    return matrix[indices, :]
