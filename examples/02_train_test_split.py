"""Random training/test sample selection (by number or by percentage).

Run:  python examples/02_train_test_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # make the package importable without installing
    sys.path.insert(0, str(ROOT))

from hyperspectral.io import load_hypercube
from hyperspectral.plotting import plot_samples, save_figure
from hyperspectral.split import split_samples

DATA = Path(__file__).resolve().parents[1] / "data" / "Indian_Pine.mat"
OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    cube = load_hypercube(str(DATA))
    dataset = cube.dataset()

    # 1) a fixed number of samples per class (small classes are skipped)
    by_number = split_samples(
        dataset, cube.ground_truth, samples=100, mode="number",
        seed=0, class_names=cube.name_map(), skip_insufficient=True,
    )
    print("by number :", by_number.summary())
    print("skipped   :", {cube.name_of(k): v for k, v in by_number.skipped_classes.items()})

    # 2) a percentage of every class, so all the classes take part
    by_percent = split_samples(
        dataset, cube.ground_truth, samples=20, mode="percent",
        seed=0, class_names=cube.name_map(),
    )
    print("by percent:", by_percent.summary())

    for label, train, test in zip(by_percent.labels,
                                  by_percent.train_counts,
                                  by_percent.test_counts):
        print(f"  {cube.name_of(label):26s} train {train:5d}   test {test:6d}")

    save_figure(plot_samples(by_percent), str(OUT / "samples.png"))
    print(f"\nfigure written to {OUT / 'samples.png'}")


if __name__ == "__main__":
    main()
