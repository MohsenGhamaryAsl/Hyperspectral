"""Random training / test sample selection (by number or by percentage)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

__all__ = ["SplitResult", "split_samples", "select_samples"]


@dataclass
class SplitResult:
    """Training and test samples selected from the labelled pixels.

    Attributes
    ----------
    train_dataset, test_dataset : numpy.ndarray
        ``(bands, n_samples)`` matrices, classes concatenated in the order of
        :attr:`labels`.
    train_counts, test_counts : list of int
        Number of samples of every class in each set.
    labels : list of int
        Class labels, in the order used by the datasets.
    class_totals : list of int
        Number of available labelled pixels of every class.
    train_map, test_map : numpy.ndarray
        ``(rows, columns)`` maps containing the class label at the selected
        pixel positions and ``0`` elsewhere (handy for figures).
    class_names : list of str
        Human readable class names, aligned with :attr:`labels`.
    """

    train_dataset: np.ndarray
    test_dataset: np.ndarray
    train_counts: List[int]
    test_counts: List[int]
    labels: List[int]
    class_totals: List[int]
    train_map: np.ndarray
    test_map: np.ndarray
    class_names: List[str]
    skipped_classes: Dict[int, int] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------
    @property
    def n_train(self) -> int:
        return int(sum(self.train_counts))

    @property
    def n_test(self) -> int:
        return int(sum(self.test_counts))

    @property
    def n_classes(self) -> int:
        return len(self.labels)

    def train_ranges(self) -> tuple:
        """``(starts, ends)`` column ranges of the classes in ``train_dataset``."""
        from .datasets import cumulative_ranges

        return cumulative_ranges(self.train_counts)

    def summary(self) -> Dict[str, Union[int, List[int]]]:
        """Compact dictionary describing the split (nice for logging)."""
        return {
            "classes": self.n_classes,
            "train_samples": self.n_train,
            "test_samples": self.n_test,
            "train_counts": list(map(int, self.train_counts)),
            "test_counts": list(map(int, self.test_counts)),
            "skipped_classes": dict(self.skipped_classes),
        }


def _as_per_class(items: Union[int, float, Sequence[float], np.ndarray], n: int, what: str):
    """Broadcast a scalar to one value per class."""
    array = np.atleast_1d(np.asarray(items, dtype=float))
    if array.size == 1:
        return np.repeat(array, n)
    if array.size != n:
        raise ValueError(
            f"{what} must be a scalar or have one value per class "
            f"({n} classes), got {array.size}"
        )
    return array


def select_samples(
    ground_truth: np.ndarray,
    samples: Union[int, float, Sequence[float]],
    mode: str = "number",
    *,
    background: Optional[Sequence[int]] = (0,),
    seed: Optional[int] = None,
    skip_insufficient: bool = False,
) -> Dict[str, np.ndarray]:
    """Pick random training positions and return them with the remaining ones.

    Parameters
    ----------
    ground_truth : numpy.ndarray
        ``(rows, columns)`` label map.
    samples : int, float or sequence
        Number of training samples per class (``mode="number"``) or percentage
        of every class (``mode="percent"``). A sequence gives one value per
        class.
    mode : {"number", "percent"}
        How to interpret *samples*.
    background : sequence of int, optional
        Labels that are ignored (default ``(0,)``, the unlabelled pixels).
    seed : int, optional
        Seed of the random generator, for reproducible splits.
    skip_insufficient : bool
        When ``True``, classes that do not contain enough pixels for the
        requested number of training samples are dropped instead of raising an
        error (their label is reported in ``skipped``).

    Returns
    -------
    dict
        ``labels``, ``counts`` (labelled pixels per class), ``train_counts``,
        ``test_counts``, ``train_positions`` / ``test_positions`` (flat pixel
        indices per class), ``train_map`` / ``test_map`` and ``skipped``
        (``{label: available_pixels}`` of the dropped classes).
    """
    if mode not in ("number", "percent"):
        raise ValueError("mode must be 'number' or 'percent'")

    gt = np.asarray(ground_truth)
    if gt.ndim != 2:
        raise ValueError(f"ground_truth must be 2-D, got shape {gt.shape}")

    ignored = set(int(v) for v in np.atleast_1d(background if background is not None else ()))
    labels = [int(v) for v in np.unique(gt) if int(v) not in ignored]
    if not labels:
        raise ValueError("the ground truth contains no labelled pixel")

    counts = np.array([int(np.count_nonzero(gt == label)) for label in labels])

    if mode == "number":
        train_counts = _as_per_class(samples, len(labels), "samples").astype(int)
    else:
        percentages = _as_per_class(samples, len(labels), "samples")
        train_counts = np.floor(percentages / 100.0 * counts).astype(int)

    if np.any(train_counts < 0):
        raise ValueError("the number of training samples cannot be negative")

    skipped: Dict[int, int] = {}
    insufficient = np.where(train_counts > counts)[0]
    if insufficient.size:
        if not skip_insufficient:
            raise ValueError(
                "not enough labelled pixels for the requested training samples: "
                + ", ".join(
                    f"class {labels[i]} has {int(counts[i])}" for i in insufficient
                )
                + " (use skip_insufficient=True or fewer samples)"
            )
        skipped = {int(labels[i]): int(counts[i]) for i in insufficient}
        keep = np.setdiff1d(np.arange(len(labels)), insufficient)
        labels = [labels[i] for i in keep]
        counts = counts[keep]
        train_counts = train_counts[keep]

    test_counts = counts - train_counts

    rng = np.random.default_rng(seed)
    train_positions, test_positions = [], []
    for label, n_train, n_total in zip(labels, train_counts, counts):
        flat = np.flatnonzero(gt.ravel() == label)
        order = rng.permutation(flat.size)
        train_positions.append(np.sort(flat[order[:n_train]]))
        test_positions.append(np.sort(flat[order[n_train:]]))

    def to_map(positives) -> np.ndarray:
        result = np.zeros(gt.size, dtype=np.int32)
        for label, positions in zip(labels, positives):
            result[positions] = label
        return result.reshape(gt.shape)

    return {
        "labels": np.asarray(labels),
        "counts": counts,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "train_positions": train_positions,
        "test_positions": test_positions,
        "train_map": to_map(train_positions),
        "test_map": to_map(test_positions),
        "skipped": skipped,
    }


def split_samples(
    dataset: np.ndarray,
    ground_truth: np.ndarray,
    samples: Union[int, float, Sequence[float]] = 100,
    mode: str = "number",
    *,
    background: Optional[Sequence[int]] = (0,),
    seed: Optional[int] = None,
    class_names: Optional[Union[Mapping[int, str], Sequence[str]]] = None,
    skip_insufficient: bool = False,
) -> SplitResult:
    """Select training/test samples and build both datasets.

    Parameters
    ----------
    dataset : numpy.ndarray
        ``(bands, rows * columns)`` matrix - see
        :func:`hyperspectral.io.to_dataset`.
    ground_truth : numpy.ndarray
        ``(rows, columns)`` label map.
    samples : int, float or sequence
        Samples per class (number or percentage).
    mode : {"number", "percent"}
        Interpretation of *samples*.
    background : sequence of int, optional
        Labels to ignore.
    seed : int, optional
        Seed for reproducibility.
    class_names : mapping or sequence of str, optional
        Class names. A ``{label: name}`` mapping (see
        :meth:`hyperspectral.io.Hypercube.name_map`) is used directly; a plain
        sequence is assumed to be indexed by ``label - 1``.
    skip_insufficient : bool
        Drop the classes that cannot provide the requested number of training
        samples instead of raising an error.

    Returns
    -------
    SplitResult
    """
    matrix = np.asarray(dataset)
    gt = np.asarray(ground_truth)
    if matrix.ndim != 2 or matrix.shape[1] != gt.size:
        raise ValueError(
            f"dataset must have shape (bands, rows*columns) = "
            f"(bands, {gt.size}), got {matrix.shape}"
        )

    selection = select_samples(
        gt, samples, mode, background=background, seed=seed,
        skip_insufficient=skip_insufficient,
    )
    labels = selection["labels"]

    def build(positions) -> np.ndarray:
        if not positions:
            return np.empty((matrix.shape[0], 0))
        indices = np.concatenate(positions)
        return matrix[:, indices]

    if isinstance(class_names, Mapping):
        names = [str(class_names.get(int(label), f"Class {label}")) for label in labels]
    elif class_names:
        sequence = list(class_names)
        names = [
            sequence[int(label) - 1]
            if 0 < int(label) <= len(sequence) else f"Class {label}"
            for label in labels
        ]
    else:
        names = []

    return SplitResult(
        train_dataset=build(selection["train_positions"]),
        test_dataset=build(selection["test_positions"]),
        train_counts=[int(v) for v in selection["train_counts"]],
        test_counts=[int(v) for v in selection["test_counts"]],
        labels=[int(v) for v in labels],
        class_totals=[int(v) for v in selection["counts"]],
        train_map=selection["train_map"],
        test_map=selection["test_map"],
        class_names=names,
        skipped_classes=selection.get("skipped", {}),
    )
