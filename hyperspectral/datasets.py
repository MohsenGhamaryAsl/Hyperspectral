"""Dataset helpers: cumulative ranges and per-class mean signatures."""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["cumulative_ranges", "class_dataset", "class_means"]


def cumulative_ranges(numbers: Sequence[int]) -> tuple:
    """Start/end column indices of consecutive blocks of the given sizes.

    This is the building block behind "the first 100 columns belong to class 1,
    the next 100 to class 2, ..." datasets.

    Parameters
    ----------
    numbers : sequence of int
        Number of columns of each block.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Arrays of start (inclusive) and end (exclusive) indices.

    Examples
    --------
    >>> from hyperspectral.datasets import cumulative_ranges
    >>> start, end = cumulative_ranges([3, 2])
    >>> start.tolist(), end.tolist()
    ([0, 3], [3, 5])
    """
    sizes = np.atleast_1d(np.asarray(numbers, dtype=int))
    if sizes.ndim != 1:
        raise ValueError("numbers must be a 1-D sequence of block sizes")
    ends = np.cumsum(sizes)
    starts = np.concatenate(([0], ends[:-1])).astype(int)
    return starts, ends.astype(int)


def class_dataset(dataset: np.ndarray, ranges: Sequence[np.ndarray]) -> np.ndarray:
    """Mean signature of every block of columns of *dataset*.

    Parameters
    ----------
    dataset : numpy.ndarray
        ``(bands, samples)`` matrix whose columns are grouped by class.
    ranges : sequence
        ``(starts, ends)`` as returned by :func:`cumulative_ranges`.

    Returns
    -------
    numpy.ndarray
        ``(bands, n_classes)`` matrix of class-mean spectra.
    """
    starts, ends = ranges
    matrix = np.asarray(dataset, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"dataset must be 2-D, got shape {matrix.shape}")

    means = np.empty((matrix.shape[0], len(starts)), dtype=float)
    for index, (start, end) in enumerate(zip(starts, ends)):
        block = matrix[:, int(start):int(end)]
        if block.size == 0:
            raise ValueError(
                f"empty range for class at position {index} (start={start}, end={end})"
            )
        means[:, index] = block.mean(axis=1)
    return means


def class_means(dataset: np.ndarray, counts: Sequence[int]) -> np.ndarray:
    """Convenience wrapper: mean signature per class from class sizes."""
    return class_dataset(dataset, cumulative_ranges(counts))
