"""Maximum Tangent Discrimination (MTD) band selection.

Run:  python examples/05_band_selection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # make the package importable without installing
    sys.path.insert(0, str(ROOT))

from hyperspectral.datasets import class_means
from hyperspectral.features import maximum_tangent_discrimination, select_bands
from hyperspectral.io import load_hypercube
from hyperspectral.plotting import plot_spectra, save_figure
from hyperspectral.split import split_samples

DATA = Path(__file__).resolve().parents[1] / "data" / "Indian_Pine.mat"
OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    cube = load_hypercube(str(DATA))
    dataset = cube.dataset()
    split = split_samples(dataset, cube.ground_truth, 20, mode="percent", seed=0)

    means = class_means(split.train_dataset, split.train_counts)
    selected = maximum_tangent_discrimination(means)
    print(f"MTD selected {len(selected)} of {cube.bands} bands")
    print("band indices:", selected)

    subset = select_bands(split.train_dataset, selected)
    print(f"dataset reduced from {split.train_dataset.shape} to {subset.shape}")

    save_figure(plot_spectra(means, "Class mean spectra"),
                str(OUT / "class_mean_spectra.png"))
    print(f"\nfigure written to {OUT / 'class_mean_spectra.png'}")


if __name__ == "__main__":
    main()
