"""Tests for loading data cubes."""

from __future__ import annotations

import numpy as np
import pytest

from hyperspectral.io import Hypercube, load_hypercube, save_hypercube, to_cube, to_dataset


def test_dataset_roundtrip(synthetic_cube):
    matrix = to_dataset(synthetic_cube.pixels)
    assert matrix.shape == (synthetic_cube.bands, synthetic_cube.rows * synthetic_cube.columns)
    restored = to_cube(matrix, synthetic_cube.shape[:2])
    np.testing.assert_allclose(restored, synthetic_cube.pixels)


def test_label_counts_and_names(synthetic_cube):
    assert synthetic_cube.labels.tolist() == [1, 2, 3]
    assert synthetic_cube.label_counts() == {1: 48, 2: 48, 3: 48}
    assert synthetic_cube.name_map() == {1: "Alpha", 2: "Beta", 3: "Gamma"}
    assert synthetic_cube.name_of(2) == "Beta"


def test_load_and_save_npz(tmp_path, synthetic_cube):
    path = tmp_path / "cube.npz"
    save_hypercube(str(path), synthetic_cube)
    loaded = load_hypercube(str(path))
    np.testing.assert_array_equal(loaded.pixels, synthetic_cube.pixels)
    np.testing.assert_array_equal(loaded.ground_truth, synthetic_cube.ground_truth)


def test_load_npy(tmp_path, synthetic_cube):
    path = tmp_path / "cube.npy"
    np.save(path, synthetic_cube.pixels)
    loaded = load_hypercube(str(path))
    assert loaded.shape == synthetic_cube.shape
    assert np.count_nonzero(loaded.ground_truth) == 0


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_hypercube("does-not-exist.mat")


def test_invalid_cube_shape():
    with pytest.raises(ValueError):
        Hypercube(pixels=np.zeros((4, 4)), ground_truth=np.zeros((4, 4)))
    with pytest.raises(ValueError):
        Hypercube(pixels=np.zeros((4, 4, 3)), ground_truth=np.zeros((4, 5)))
