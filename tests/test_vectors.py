"""Tests for the vector geometry helpers."""

from __future__ import annotations

import numpy as np
import pytest

from hyperspectral.angles import angles_between_vectors
from hyperspectral.distances import pairwise_distances


def test_euclidean_distances():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    expected = np.array([[0.0, 5.0], [5.0, 0.0]])
    np.testing.assert_allclose(pairwise_distances(X), expected)


def test_distance_metrics():
    X = np.array([[0.0, 0.0]])
    Y = np.array([[3.0, 4.0]])
    assert pairwise_distances(X, Y)[0, 0] == pytest.approx(5.0)
    assert pairwise_distances(X, Y, "sqeuclidean")[0, 0] == pytest.approx(25.0)
    assert pairwise_distances(X, Y, "cityblock")[0, 0] == pytest.approx(7.0)
    assert pairwise_distances(X, Y, "chebyshev")[0, 0] == pytest.approx(4.0)


def test_metric_validation():
    with pytest.raises(ValueError):
        pairwise_distances(np.zeros((2, 3)), metric="manhattan-ish")
    with pytest.raises(ValueError):
        pairwise_distances(np.zeros((2, 3)), np.zeros((2, 4)))


def test_angles_units_and_self():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    radians = angles_between_vectors(X)
    np.testing.assert_allclose(radians, np.pi / 2 * np.array([[0, 1], [1, 0]]))
    degrees = angles_between_vectors(X, unit="deg")
    np.testing.assert_allclose(degrees, [[0, 90], [90, 0]], atol=1e-9)


def test_angles_are_finite_for_parallel_and_zero_vectors():
    X = np.array([[1.0, 2.0], [2.0, 4.0], [0.0, 0.0]])
    angles = angles_between_vectors(X, unit="deg")
    assert np.all(np.isfinite(angles))
    assert angles[0, 1] == pytest.approx(0.0, abs=1e-6)
