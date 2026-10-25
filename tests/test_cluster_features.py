"""Tests for K-means clustering, band selection and plotting helpers."""

from __future__ import annotations

import numpy as np
import pytest

from hyperspectral.cluster import cluster_map, kmeans
from hyperspectral.features import maximum_tangent_discrimination, select_bands
from hyperspectral.plotting import save_figure, plot_samples
from hyperspectral.split import split_samples


def _two_blobs(seed=0, n=60):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(0, 0.1, (n, 3)), rng.normal(6, 0.1, (n, 3))])


def test_kmeans_separates_blobs():
    data = _two_blobs()
    result = kmeans(data, 2, seed=0)
    assert result.n_clusters == 2
    assert result.labels.shape == (data.shape[0],)
    assert set(np.unique(result.labels).tolist()) == {1, 2}
    first = result.labels[:60]
    second = result.labels[60:]
    assert len(set(first.tolist())) == 1
    assert len(set(second.tolist())) == 1
    assert first[0] != second[0]


def test_kmeans_reproducible_and_monotone():
    data = _two_blobs()
    first = kmeans(data, 3, seed=4)
    second = kmeans(data, 3, seed=4)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert all(a >= b - 1e-9 for a, b in zip(first.history, first.history[1:]))


def test_kmeans_validation():
    with pytest.raises(ValueError):
        kmeans(np.zeros((5, 2)), 0)
    with pytest.raises(ValueError):
        kmeans(np.zeros((5, 2)), 3, init="not-a-strategy")


def test_cluster_map_shape(synthetic_cube, synthetic_dataset):
    result = cluster_map(synthetic_dataset.T, synthetic_cube.shape[:2], 3, seed=0)
    assert result.shape == synthetic_cube.shape[:2]
    assert set(np.unique(result).tolist()) <= {1, 2, 3}


def test_feature_selection(synthetic_cube, synthetic_dataset):
    split = split_samples(synthetic_dataset, synthetic_cube.ground_truth, 12, seed=0)
    from hyperspectral.datasets import class_means

    features = maximum_tangent_discrimination(class_means(split.train_dataset,
                                                          split.train_counts))
    assert features, "at least one band must be selected"
    assert len(features) <= synthetic_cube.bands - 1
    assert len(set(features)) == len(features)
    selected = select_bands(split.train_dataset, features)
    assert selected.shape[0] == len(features)


def test_select_bands_validation(synthetic_dataset):
    with pytest.raises(ValueError):
        select_bands(synthetic_dataset, [])
    with pytest.raises(ValueError):
        select_bands(synthetic_dataset, [10_000])


def test_plot_samples_is_saved(tmp_path, synthetic_cube, synthetic_dataset):
    split = split_samples(synthetic_dataset, synthetic_cube.ground_truth, 12, seed=0)
    path = save_figure(plot_samples(split), str(tmp_path / "figures" / "samples.png"))
    assert (tmp_path / "figures" / "samples.png").is_file()
    assert path.endswith("samples.png")
