"""Figures for hyperspectral maps (matplotlib)."""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # head-less friendly: saving figures never needs a screen
import matplotlib.pyplot as plt  # noqa: E402  (after the backend selection)
import numpy as np  # noqa: E402

from .io import to_cube  # noqa: E402

__all__ = [
    "plot_map",
    "plot_samples",
    "plot_classification",
    "plot_spectra",
    "plot_band",
    "plot_convergence",
    "save_figure",
]

DEFAULT_CMAP = "nipy_spectral"


def _style_map(image: plt.Axes, title: str, n_classes: int) -> None:
    image.axes.set_title(title, fontsize=10)
    image.axes.set_xticks([])
    image.axes.set_yticks([])


def plot_map(
    label_map: np.ndarray,
    title: str = "",
    *,
    cmap: str = DEFAULT_CMAP,
    ax: Optional[plt.Axes] = None,
):
    """Show a label/cluster map (``0`` is drawn in white)."""
    values = np.asarray(label_map, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    masked = np.ma.masked_where(values <= 0, values)
    displayed = ax.imshow(masked, cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return displayed


def plot_samples(split, title: str = "Training and test samples"):
    """Side-by-side maps of the training and test samples of a split."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_map(split.train_map, "Training samples", ax=axes[0])
    plot_map(split.test_map, "Test samples", ax=axes[1])
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_classification(
    classification: np.ndarray,
    title: str = "Classification map",
    *,
    ground_truth: Optional[np.ndarray] = None,
):
    """Classification map, optionally next to the ground truth."""
    if ground_truth is None:
        figure, ax = plt.subplots(figsize=(5.5, 5))
        plot_map(classification, title, ax=ax)
        figure.tight_layout()
        return figure

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_map(ground_truth, "Ground truth", ax=axes[0])
    plot_map(classification, title, ax=axes[1])
    figure.tight_layout()
    return figure


def plot_spectra(
    dataset: np.ndarray,
    title: str = "Mean spectra",
    *,
    labels: Optional[Sequence[str]] = None,
    max_lines: int = 40,
    ax: Optional[plt.Axes] = None,
):
    """Plot the mean signature of every class block of *dataset*."""
    matrix = np.asarray(dataset, dtype=float)
    if matrix.shape[1] > max_lines:
        matrix = matrix[:, :max_lines]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for column in range(matrix.shape[1]):
        name = labels[column] if labels and column < len(labels) else f"sample {column + 1}"
        ax.plot(matrix[:, column], linewidth=1, label=name)
    ax.set_xlabel("band")
    ax.set_ylabel("reflectance")
    ax.set_title(title, fontsize=10)
    if labels:
        ax.legend(fontsize=6, ncol=2)
    return ax


def plot_band(cube: np.ndarray, band: int = 0, title: Optional[str] = None):
    """Show one band of a ``(rows, columns, bands)`` cube."""
    data = np.asarray(cube)[:, :, int(band)]
    figure, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(data, cmap="gray", interpolation="nearest")
    ax.set_title(title or f"Band {band}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    figure.colorbar(image, ax=ax, fraction=0.046)
    figure.tight_layout()
    return figure


def plot_convergence(history: Sequence[float], title: str = "K-means convergence"):
    """Inertia per iteration."""
    figure, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(history) + 1), history, marker="o", markersize=3)
    ax.set_xlabel("iteration")
    ax.set_ylabel("inertia")
    ax.set_title(title, fontsize=10)
    figure.tight_layout()
    return figure


def _to_figure(obj):
    """Accept a Figure, an Axes or an Artist and return the parent Figure."""
    if hasattr(obj, "savefig"):
        return obj
    if hasattr(obj, "figure"):
        return obj.figure
    if hasattr(obj, "get_figure"):  # pragma: no cover - defensive
        return obj.get_figure()
    raise TypeError(f"cannot find a matplotlib figure in {type(obj)!r}")


def save_figure(figure, path: str, dpi: int = 150) -> str:
    """Save a figure (or any of its artists) and close it.

    The directory is created when needed and the absolute path of the written
    file is returned.
    """
    figure = _to_figure(figure)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return os.path.abspath(path)
