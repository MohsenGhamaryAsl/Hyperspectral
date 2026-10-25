"""Loading and handling hyperspectral data cubes.

A data cube is a ``(rows, columns, bands)`` array. The ground truth is a
``(rows, columns)`` label map where ``0`` conventionally means "unlabelled"
(background). All the algorithms in this package work on the *matrix* view of
the cube (``(bands, rows * columns)``), which is what :func:`to_dataset`
returns and :func:`to_cube` converts back.

Supported files
---------------
* MATLAB ``.mat`` (variable names ``Pixels``/``Image_DataCube``,
  ``Trained_Data``/``Ground_Truth``, optional ``Classes``/``Classes_List``)
* NumPy ``.npz`` with the same variable names
* plain ``.npy`` files containing just the cube

Examples
--------
>>> from hyperspectral.io import load_hypercube
>>> cube = load_hypercube("data/Indian_Pine.mat")   # doctest: +SKIP
>>> cube.shape                                      # doctest: +SKIP
(145, 145, 200)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "Hypercube",
    "load_hypercube",
    "save_hypercube",
    "to_dataset",
    "to_cube",
    "labels_from_dataset",
]

_CUBE_KEYS = ("Pixels", "Image_DataCube", "image", "cube", "data")
_GT_KEYS = ("Trained_Data", "Ground_Truth", "ground_truth", "labels", "gt")
_CLASS_KEYS = ("Classes", "Classes_List", "classes")


@dataclass
class Hypercube:
    """A hyperspectral data cube together with its ground truth.

    Attributes
    ----------
    pixels : numpy.ndarray
        ``(rows, columns, bands)`` data cube.
    ground_truth : numpy.ndarray
        ``(rows, columns)`` integer label map; ``0`` means "unlabelled".
    class_names : list of str, optional
        Human readable name of each label, indexed by ``label - 1`` when the
        labels start at 1 (otherwise indexed by position).
    path : str, optional
        File the cube was loaded from.
    metadata : dict
        Anything else that was found in the source file (e.g. sensor
        information).
    """

    pixels: np.ndarray
    ground_truth: np.ndarray
    class_names: List[str] = field(default_factory=list)
    path: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pixels = np.asarray(self.pixels)
        self.ground_truth = np.asarray(self.ground_truth)
        if self.pixels.ndim != 3:
            raise ValueError(
                f"pixels must be a 3-D array (rows, columns, bands), "
                f"got shape {self.pixels.shape}"
            )
        if self.ground_truth.shape != self.pixels.shape[:2]:
            raise ValueError(
                f"ground truth shape {self.ground_truth.shape} does not match "
                f"the spatial shape {self.pixels.shape[:2]}"
            )

    # -- basic properties -------------------------------------------------
    @property
    def shape(self) -> tuple:
        """``(rows, columns, bands)`` of the cube."""
        return self.pixels.shape

    @property
    def rows(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def columns(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def bands(self) -> int:
        return int(self.pixels.shape[2])

    @property
    def labels(self) -> np.ndarray:
        """Sorted array of the labelled classes (background excluded)."""
        return np.unique(self.ground_truth[self.ground_truth > 0])

    def label_counts(self) -> Dict[int, int]:
        """Number of pixels of every class (background excluded)."""
        values, counts = np.unique(
            self.ground_truth[self.ground_truth > 0], return_counts=True
        )
        return {int(v): int(c) for v, c in zip(values, counts)}

    def name_map(self) -> Dict[int, str]:
        """Mapping ``label -> class name``.

        Two layouts are supported, chosen automatically:

        * the name list contains an entry for the background as well
          (``len(names) == n_labels + 1``): it is indexed directly by the label;
        * the list only contains the classes (``len(names) == n_labels``):
          the labels are assumed to be ``1..n`` and the list is shifted by one.
        """
        if not self.class_names:
            return {}
        labels = [int(v) for v in self.labels]
        names = list(self.class_names)
        if labels and len(names) == len(labels) + 1:
            return {label: names[label] for label in labels if 0 <= label < len(names)}
        if labels and len(names) == len(labels):
            return {
                label: names[label - 1]
                for label in labels
                if 0 < label <= len(names)
            }
        # Unknown layout: keep the positional convention.
        return {
            label: (names[label - 1] if 0 < label <= len(names) else f"Class {label}")
            for label in labels
        }

    def name_of(self, label: int) -> str:
        """Name of *label*, falling back to ``"Class <label>"``."""
        return self.name_map().get(int(label), f"Class {label}")

    # -- conversions ------------------------------------------------------
    def dataset(self) -> np.ndarray:
        """The cube as a ``(bands, rows * columns)`` matrix."""
        return to_dataset(self.pixels)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Hypercube(shape={self.shape}, classes={len(self.labels)}, "
            f"path={os.path.basename(self.path) if self.path else None!r})"
        )


def _clean_class_names(raw) -> List[str]:
    """Normalise the class-name container found in a ``.mat``/``.npz`` file."""
    if raw is None:
        return []
    names: List[str] = []
    for item in np.asarray(raw, dtype=object).ravel():
        if isinstance(item, (list, tuple, np.ndarray)):
            inner = np.asarray(item).ravel()
            text = str(inner[0]) if inner.size else ""
        else:
            text = str(item)
        names.append(text.strip())
    return names


def load_hypercube(
    path: str,
    cube_key: Optional[str] = None,
    ground_truth_key: Optional[str] = None,
) -> Hypercube:
    """Load a data cube (and its ground truth) from ``.mat``, ``.npz`` or ``.npy``.

    Parameters
    ----------
    path : str
        File to read.
    cube_key, ground_truth_key : str, optional
        Variable names to use. When omitted the usual names are tried
        (``Pixels``/``Image_DataCube``/``cube`` and
        ``Trained_Data``/``Ground_Truth``/``labels``).

    Returns
    -------
    Hypercube
        The loaded cube.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If no data cube could be found inside the file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    extension = os.path.splitext(path)[1].lower()
    container: Dict[str, object] = {}

    if extension == ".npy":
        container["Pixels"] = np.load(path)
    elif extension == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            container.update({key: archive[key] for key in archive.files})
    elif extension in (".mat", ".h5", ".hdf5"):
        from scipy.io import loadmat

        raw = loadmat(path)
        container.update({k: v for k, v in raw.items() if not k.startswith("__")})
    else:
        raise ValueError(
            f"unsupported file type {extension!r} (use .mat, .npz or .npy)"
        )

    if cube_key is None:
        cube_key = next((k for k in _CUBE_KEYS if k in container), None)
    if cube_key is None or cube_key not in container:
        raise ValueError(
            f"no data cube found in {os.path.basename(path)!r}; "
            f"available variables: {sorted(container)}"
        )

    pixels = np.asarray(container[cube_key])
    if pixels.ndim == 2:  # (bands, rows*columns) matrices are accepted too
        raise ValueError("the cube must be 3-D (rows, columns, bands)")

    if ground_truth_key is None:
        ground_truth_key = next((k for k in _GT_KEYS if k in container), None)

    if ground_truth_key and ground_truth_key in container:
        ground_truth = np.asarray(container[ground_truth_key])
    else:
        ground_truth = np.zeros(pixels.shape[:2], dtype=np.int32)

    class_names = _clean_class_names(
        next((container[k] for k in _CLASS_KEYS if k in container), None)
    )

    metadata = {
        key: value
        for key, value in container.items()
        if key not in (cube_key, ground_truth_key) and not key.startswith("_")
        and not isinstance(value, np.ndarray)
    }

    return Hypercube(
        pixels=pixels,
        ground_truth=ground_truth,
        class_names=class_names,
        path=os.path.abspath(path),
        metadata=metadata,
    )


def save_hypercube(path: str, cube: Hypercube) -> None:
    """Save *cube* to ``.npz`` (fast) or ``.mat`` (MATLAB compatible)."""
    payload = {
        "Pixels": cube.pixels,
        "Trained_Data": cube.ground_truth,
        "Classes": np.asarray(cube.class_names, dtype=object),
    }
    extension = os.path.splitext(path)[1].lower()
    if extension == ".npz":
        np.savez_compressed(path, **payload)
    elif extension == ".mat":
        from scipy.io import savemat

        savemat(path, payload)
    else:
        raise ValueError("use .npz or .mat")


def to_dataset(cube: np.ndarray) -> np.ndarray:
    """Reshape ``(rows, columns, bands)`` into ``(bands, rows * columns)``."""
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"expected a 3-D cube, got shape {cube.shape}")
    rows, columns, bands = cube.shape
    return np.reshape(cube.transpose(2, 0, 1), (bands, rows * columns))


def to_cube(dataset: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Reshape a ``(bands, rows * columns)`` matrix back into a cube."""
    rows, columns = int(shape[0]), int(shape[1])
    bands = int(np.shape(dataset)[0])
    return np.reshape(np.asarray(dataset), (bands, rows, columns)).transpose(1, 2, 0)


def labels_from_dataset(dataset: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Reshape a per-pixel label vector into a ``(rows, columns)`` map."""
    rows, columns = int(shape[0]), int(shape[1])
    return np.reshape(np.asarray(dataset), (rows, columns))
