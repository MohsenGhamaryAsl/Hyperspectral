"""K-means clustering of a hyperspectral scene (unsupervised).

Run:  python examples/04_kmeans_clustering.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # make the package importable without installing
    sys.path.insert(0, str(ROOT))

from hyperspectral.cluster import kmeans
from hyperspectral.io import load_hypercube
from hyperspectral.plotting import plot_classification, plot_convergence, save_figure

DATA = Path(__file__).resolve().parents[1] / "data" / "Indian_Pine.mat"
OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    cube = load_hypercube(str(DATA))
    # one pixel per row: (rows * columns, bands)
    pixels = cube.dataset().T

    result = kmeans(pixels, n_clusters=8, max_iter=100, tol=1e-6, seed=0)
    print(f"clusters  : {result.n_clusters}")
    print(f"iterations: {result.n_iter}")
    print(f"inertia   : {result.inertia:.4f}")
    print("cluster sizes:", {
        label: int((result.labels == label).sum())
        for label in sorted(set(result.labels.tolist()))
    })

    cluster_map_2d = result.labels.reshape(cube.rows, cube.columns)
    save_figure(
        plot_classification(cluster_map_2d,
                            f"K-means ({result.n_clusters} clusters, {result.n_iter} iterations)"),
        str(OUT / "kmeans_map.png"),
    )
    save_figure(plot_convergence(result.history), str(OUT / "kmeans_convergence.png"))
    print(f"\nfigures written to {OUT}")


if __name__ == "__main__":
    main()
