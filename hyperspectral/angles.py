"""Angles between sets of vectors (spectral angle measure).

The functions take arrays whose **rows are the vectors** (one spectrum per row)
and return the full matrix of angles between every pair of vectors.

Typical use in hyperspectral analysis: measuring the angle between two band
images or two spectral signatures. Two vectors pointing in the same direction
have an angle of 0; orthogonal vectors have an angle of ``pi / 2``.

Examples
--------
>>> import numpy as np
>>> from hyperspectral.angles import angles_between_vectors
>>> X = np.array([[1.0, 0.0], [0.0, 1.0]])
>>> np.round(angles_between_vectors(X, unit="deg"), 6)
array([[ 0., 90.],
       [90.,  0.]])
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["angles_between_vectors", "UNIT_FACTORS"]

#: Multipliers that convert radians into the requested unit.
UNIT_FACTORS = {"rad": 1.0, "deg": 180.0 / np.pi, "grad": 200.0 / np.pi}


def angles_between_vectors(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    unit: str = "rad",
) -> np.ndarray:
    """Angles between the rows of ``X`` and the rows of ``Y``.

    Parameters
    ----------
    X, Y : array_like
        ``(n_X, n_features)`` and ``(n_Y, n_features)``. When ``Y`` is ``None``
        the angle of every row of ``X`` with every other row of ``X`` is used.
    unit : {"rad", "deg", "grad"}
        Unit of the returned angles.

    Returns
    -------
    numpy.ndarray
        ``(n_X, n_Y)`` matrix of angles. Pairs involving an all-zero vector are
        reported as 0 instead of ``nan``.
    """
    if unit not in UNIT_FACTORS:
        raise ValueError(
            f"unknown unit {unit!r}, choose from {', '.join(UNIT_FACTORS)}"
        )

    x = np.atleast_2d(np.asarray(X, dtype=float))
    y = x if Y is None else np.atleast_2d(np.asarray(Y, dtype=float))
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"X and Y must have the same number of features, "
            f"got {x.shape[1]} and {y.shape[1]}"
        )

    # Normalise each vector first: this keeps the dot product in [-1, 1] and
    # makes the computation scale independent (no manual "regularisation").
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)

    x_unit = np.divide(x, x_norm, out=np.zeros_like(x), where=x_norm > 0)
    y_unit = np.divide(y, y_norm, out=np.zeros_like(y), where=y_norm > 0)

    cosine = np.clip(x_unit @ y_unit.T, -1.0, 1.0)  # clip: arccos domain
    angles = np.arccos(cosine) * UNIT_FACTORS[unit]
    return np.nan_to_num(angles, nan=0.0, posinf=0.0, neginf=0.0)
