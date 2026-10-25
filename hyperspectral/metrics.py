"""Classification accuracy metrics: confusion matrix, OA, AA, kappa."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np

__all__ = [
    "confusion_matrix",
    "overall_accuracy",
    "average_accuracy",
    "kappa_coefficient",
    "per_class_accuracy",
    "evaluate",
]


def _pairs(reference: np.ndarray, predicted: np.ndarray, ignore: Sequence[int]):
    reference = np.asarray(reference).ravel()
    predicted = np.asarray(predicted).ravel()
    if reference.size != predicted.size:
        raise ValueError(
            f"reference and prediction must have the same size, "
            f"got {reference.size} and {predicted.size}"
        )
    ignored = set(int(v) for v in np.atleast_1d(ignore))
    mask = np.array([int(r) not in ignored for r in reference], dtype=bool)
    return reference[mask].astype(int), predicted[mask].astype(int)


def confusion_matrix(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    ignore: Sequence[int] = (0,),
    labels: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Confusion matrix with rows = reference class, columns = prediction.

    Parameters
    ----------
    reference, predicted : array_like
        Label maps or vectors of the same size.
    ignore : sequence of int
        Labels excluded from the statistics (background by default).
    labels : sequence of int, optional
        Classes to report, in order (default: every class found).

    Returns
    -------
    numpy.ndarray
        ``(n_classes, n_classes)`` integer matrix.
    """
    ref, pred = _pairs(reference, predicted, ignore)
    classes = (
        np.asarray(sorted(set(int(v) for v in labels)), dtype=int)
        if labels is not None
        else np.union1d(np.unique(ref), np.unique(pred))
    )
    index = {int(label): position for position, label in enumerate(classes)}
    matrix = np.zeros((classes.size, classes.size), dtype=int)
    for r, p in zip(ref, pred):
        if r in index and p in index:
            matrix[index[r], index[p]] += 1
    return matrix


def overall_accuracy(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    ignore: Sequence[int] = (0,),
) -> float:
    """Percentage of correctly classified labelled pixels (OA)."""
    ref, pred = _pairs(reference, predicted, ignore)
    if ref.size == 0:
        return float("nan")
    return float(100.0 * np.count_nonzero(ref == pred) / ref.size)


def per_class_accuracy(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    ignore: Sequence[int] = (0,),
    labels: Optional[Sequence[int]] = None,
) -> Dict[int, float]:
    """Producer's accuracy (recall) of every class in percent."""
    ref, pred = _pairs(reference, predicted, ignore)
    classes = (
        [int(v) for v in sorted(int(x) for x in labels)]
        if labels is not None
        else [int(v) for v in np.union1d(np.unique(ref), np.unique(pred))]
    )
    matrix = confusion_matrix(reference, predicted, ignore=ignore, labels=classes)
    totals = matrix.sum(axis=1)
    return {
        label: (float(100.0 * matrix[i, i] / totals[i]) if totals[i] else float("nan"))
        for i, label in enumerate(classes)
    }


def average_accuracy(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    ignore: Sequence[int] = (0,),
) -> float:
    """Mean of the per-class accuracies in percent (AA)."""
    values = [v for v in per_class_accuracy(reference, predicted, ignore=ignore).values()
              if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def kappa_coefficient(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    ignore: Sequence[int] = (0,),
) -> float:
    """Cohen's kappa of the classification."""
    matrix = confusion_matrix(reference, predicted, ignore=ignore).astype(float)
    total = matrix.sum()
    if total == 0:
        return float("nan")
    observed = np.trace(matrix) / total
    expected = float((matrix.sum(axis=0) * matrix.sum(axis=1)).sum()) / (total * total)
    if expected >= 1.0:
        return 0.0
    return float((observed - expected) / (1.0 - expected))


def evaluate(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    ignore: Sequence[int] = (0,),
    labels: Optional[Sequence[int]] = None,
    class_names: Optional[Union[Mapping[int, str], Sequence[str]]] = None,
) -> Dict[str, object]:
    """All the usual accuracy figures in one dictionary.

    Returns
    -------
    dict
        ``overall_accuracy``, ``average_accuracy``, ``kappa``,
        ``per_class`` (label -> percent), ``class_names``, ``n_samples`` and
        ``confusion_matrix``.
    """
    matrix = confusion_matrix(reference, predicted, ignore=ignore, labels=labels)
    per_class = per_class_accuracy(reference, predicted, ignore=ignore, labels=labels)
    resolved = {}
    for label, value in per_class.items():
        if isinstance(class_names, Mapping):
            name = str(class_names.get(int(label), f"Class {label}"))
        elif class_names and 0 < int(label) <= len(class_names):
            name = str(list(class_names)[int(label) - 1])
        else:
            name = f"Class {label}"
        resolved[name] = value

    return {
        "overall_accuracy": overall_accuracy(reference, predicted, ignore=ignore),
        "average_accuracy": average_accuracy(reference, predicted, ignore=ignore),
        "kappa": kappa_coefficient(reference, predicted, ignore=ignore),
        "per_class": resolved,
        "n_samples": int(matrix.sum()),
        "confusion_matrix": matrix,
    }
