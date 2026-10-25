"""Shared pytest fixtures: small synthetic hyperspectral data."""

from __future__ import annotations

import numpy as np
import pytest

from hyperspectral.io import Hypercube


@pytest.fixture(scope="session")
def synthetic_cube() -> Hypercube:
    """A tiny, perfectly separable 3-class cube (12 x 12 pixels, 6 bands)."""
    rng = np.random.default_rng(12345)
    rows = columns = 12
    bands = 6

    means = {
        1: np.array([1.0, 0.0, 0.0, 0.0, 0.5, 0.5]),
        2: np.array([0.0, 2.0, 0.0, 0.5, 0.0, 0.5]),
        3: np.array([0.0, 0.0, 3.0, 0.5, 0.5, 0.0]),
    }

    ground_truth = np.zeros((rows, columns), dtype=np.int32)
    pixels = np.zeros((rows, columns, bands), dtype=float)

    block = rows // 3
    for index, label in enumerate(sorted(means), start=1):
        r0, r1 = (index - 1) * block, index * block
        ground_truth[r0:r1, :] = label
        noise = rng.normal(0.0, 0.02, size=(r1 - r0, columns, bands))
        pixels[r0:r1, :, :] = means[label] + noise

    return Hypercube(
        pixels=pixels,
        ground_truth=ground_truth,
        class_names=["Background", "Alpha", "Beta", "Gamma"],
    )


@pytest.fixture(scope="session")
def synthetic_dataset(synthetic_cube):
    return synthetic_cube.dataset()
