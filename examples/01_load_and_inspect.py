"""Load a hyperspectral cube and look at it.

Run:  python examples/01_load_and_inspect.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # make the package importable without installing
    sys.path.insert(0, str(ROOT))

from hyperspectral.io import load_hypercube
from hyperspectral.plotting import plot_band, plot_map, save_figure

DATA = Path(__file__).resolve().parents[1] / "data" / "Indian_Pine.mat"
OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    cube = load_hypercube(str(DATA))

    print(f"file      : {cube.path}")
    print(f"cube      : {cube.rows} x {cube.columns} x {cube.bands}")
    print(f"datatype  : {cube.pixels.dtype}, range "
          f"{cube.pixels.min()} .. {cube.pixels.max()}")
    print(f"classes   : {len(cube.labels)}")
    for label, count in cube.label_counts().items():
        print(f"  {label:2d}  {cube.name_of(label):26s} {count:6d} px")

    save_figure(plot_band(cube.pixels, 60, "Band 60"),
                str(OUT / "band60.png"))
    save_figure(plot_map(cube.ground_truth, "Ground truth"),
                str(OUT / "ground_truth.png"))
    print(f"\nfigures written to {OUT}")


if __name__ == "__main__":
    main()
