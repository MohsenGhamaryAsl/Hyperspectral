"""Tests for the Maximum Likelihood Classifier and the accuracy metrics."""

from __future__ import annotations

import numpy as np
import pytest

from hyperspectral.metrics import (
    confusion_matrix,
    evaluate,
    kappa_coefficient,
    overall_accuracy,
)
from hyperspectral.mlc import MaximumLikelihoodClassifier
from hyperspectral.split import split_samples


def test_fit_predict_separable(synthetic_cube, synthetic_dataset):
    split = split_samples(synthetic_dataset, synthetic_cube.ground_truth, 12, seed=0)
    classifier = MaximumLikelihoodClassifier(synthetic_cube.shape[:2])
    classifier.fit(split.train_dataset, split.train_counts, split.labels)

    metrics = classifier.score(split.test_map, synthetic_dataset)
    assert metrics["overall_accuracy"] == pytest.approx(100.0)
    assert metrics["kappa"] == pytest.approx(1.0)


def test_predict_map_shape(synthetic_cube, synthetic_dataset):
    split = split_samples(synthetic_dataset, synthetic_cube.ground_truth, 12, seed=0)
    classifier = MaximumLikelihoodClassifier(synthetic_cube.shape[:2])
    classifier.fit(split.train_dataset, split.train_counts, split.labels)
    prediction = classifier.predict_map(synthetic_dataset)
    assert prediction.shape == synthetic_cube.shape[:2]
    assert set(np.unique(prediction).tolist()) <= {1, 2, 3}


def test_log_likelihood_shape(synthetic_cube, synthetic_dataset):
    split = split_samples(synthetic_dataset, synthetic_cube.ground_truth, 12, seed=0)
    classifier = MaximumLikelihoodClassifier(synthetic_cube.shape[:2])
    classifier.fit(split.train_dataset, split.train_counts, split.labels)
    scores = classifier.log_likelihood(synthetic_dataset)
    assert scores.shape == (3, synthetic_cube.rows * synthetic_cube.columns)
    assert np.all(np.isfinite(scores))


def test_predict_before_fit_raises():
    classifier = MaximumLikelihoodClassifier((2, 2))
    with pytest.raises(RuntimeError):
        classifier.predict(np.zeros((3, 4)))


def test_inconsistent_counts_raise(synthetic_cube, synthetic_dataset):
    split = split_samples(synthetic_dataset, synthetic_cube.ground_truth, 12, seed=0)
    classifier = MaximumLikelihoodClassifier(synthetic_cube.shape[:2])
    with pytest.raises(ValueError):
        classifier.fit(split.train_dataset[:, :-5], split.train_counts, split.labels)


def test_accuracy_helpers():
    reference = np.array([[1, 2], [3, 0]])
    predicted = np.array([[1, 2], [1, 0]])
    assert overall_accuracy(reference, predicted) == pytest.approx(200 / 3)
    matrix = confusion_matrix(reference, predicted)
    assert matrix.shape == (3, 3)
    assert np.trace(matrix) == 2
    report = evaluate(reference, predicted)
    assert report["n_samples"] == 3
    assert 0.0 <= report["kappa"] <= 1.0
    assert kappa_coefficient(reference, predicted) == pytest.approx(report["kappa"])
