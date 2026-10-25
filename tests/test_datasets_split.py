"""Tests for the sample-selection helpers."""

from __future__ import annotations

import numpy as np
import pytest

from hyperspectral.datasets import class_dataset, class_means, cumulative_ranges
from hyperspectral.split import select_samples, split_samples


def test_cumulative_ranges():
    starts, ends = cumulative_ranges([3, 2, 5])
    assert starts.tolist() == [0, 3, 5]
    assert ends.tolist() == [3, 5, 10]


def test_class_dataset_means():
    data = np.array([[1.0, 3.0, 10.0], [2.0, 4.0, 20.0]])
    means = class_dataset(data, cumulative_ranges([2, 1]))
    np.testing.assert_allclose(means, [[2.0, 10.0], [3.0, 20.0]])
    np.testing.assert_allclose(class_means(data, [2, 1]), means)


def test_split_by_number(synthetic_cube, synthetic_dataset):
    split = split_samples(synthetic_dataset, synthetic_cube.ground_truth, samples=10, seed=0)
    assert split.labels == [1, 2, 3]
    assert split.train_counts == [10, 10, 10]
    assert split.test_counts == [38, 38, 38]
    assert split.train_dataset.shape == (synthetic_cube.bands, 30)
    assert np.count_nonzero(split.train_map) == 30


def test_split_by_percentage(synthetic_cube, synthetic_dataset):
    split = split_samples(synthetic_dataset, synthetic_cube.ground_truth, samples=25,
                          mode="percent", seed=0)
    assert split.train_counts == [12, 12, 12]
    assert np.count_nonzero(split.test_map) == 108


def test_split_reproducible(synthetic_cube, synthetic_dataset):
    first = split_samples(synthetic_dataset, synthetic_cube.ground_truth, 10, seed=7)
    second = split_samples(synthetic_dataset, synthetic_cube.ground_truth, 10, seed=7)
    np.testing.assert_array_equal(first.train_map, second.train_map)


def test_split_train_and_test_disjoint(synthetic_cube, synthetic_dataset):
    split = split_samples(synthetic_dataset, synthetic_cube.ground_truth, 10, seed=1)
    assert np.count_nonzero((split.train_map > 0) & (split.test_map > 0)) == 0


def test_too_many_samples_raises_or_skips(synthetic_cube, synthetic_dataset):
    with pytest.raises(ValueError):
        split_samples(synthetic_dataset, synthetic_cube.ground_truth, 100, seed=0)

    split = split_samples(synthetic_dataset, synthetic_cube.ground_truth, 100,
                          seed=0, skip_insufficient=True)
    assert split.labels == []
    assert set(split.skipped_classes) == {1, 2, 3}


def test_background_labels_are_ignored(synthetic_cube, synthetic_dataset):
    selection = select_samples(synthetic_cube.ground_truth, 5, background=[0, 3], seed=0)
    assert selection["labels"].tolist() == [1, 2]


def test_invalid_mode(synthetic_cube):
    with pytest.raises(ValueError):
        select_samples(synthetic_cube.ground_truth, 5, mode="everything")
