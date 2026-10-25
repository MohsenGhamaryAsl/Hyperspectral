"""Command line interface: ``python -m hyperspectral ...``.

Examples
--------
Classify the bundled Indian Pines scene with 100 training samples per class::

    python -m hyperspectral --data data/Indian_Pine.mat --samples 100 --outdir results

Cluster it into 8 classes::

    python -m hyperspectral --data data/Indian_Pine.mat --task kmeans --clusters 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np

from .cluster import kmeans
from .datasets import class_means
from .features import maximum_tangent_discrimination, select_bands
from .io import load_hypercube
from .mlc import MaximumLikelihoodClassifier
from .plotting import plot_classification, plot_convergence, plot_samples, save_figure
from .split import split_samples

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hyperspectral",
        description="Hyperspectral image processing: sample selection, band "
                    "selection, Maximum Likelihood Classification and K-means.",
    )
    parser.add_argument("--data", required=True,
                        help="data cube (.mat, .npz or .npy)")
    parser.add_argument("--task", choices=("mlc", "kmeans"), default="mlc",
                        help="what to run (default: mlc)")
    parser.add_argument("--samples", type=float, default=100,
                        help="training samples per class (default: 100)")
    parser.add_argument("--percent", action="store_true",
                        help="interpret --samples as a percentage of each class")
    parser.add_argument("--background", default="0",
                        help="comma separated labels to ignore (default: 0)")
    parser.add_argument("--clusters", type=int, default=8,
                        help="number of clusters for --task kmeans (default: 8)")
    parser.add_argument("--bands", default="mtd",
                        help="'mtd' for Maximum Tangent Discrimination band "
                             "selection, 'all' to use every band, or a comma "
                             "separated list of band indices")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed of the sample selection (default: 0)")
    parser.add_argument("--outdir", default="results",
                        help="directory for the figures and the JSON report")
    parser.add_argument("--strict", action="store_true",
                        help="fail instead of dropping classes that have fewer "
                             "pixels than the requested number of samples")
    parser.add_argument("--quiet", action="store_true", help="only print the report")
    parser.add_argument("--version", action="version",
                        version="hyperspectral 1.0.0 (Mohsen Ghamary Asl, MIT)")
    return parser


def _parse_bands(text: str, dataset: np.ndarray, train_counts, labels) -> List[int]:
    """Resolve the ``--bands`` option into a list of band indices."""
    text = str(text).strip().lower()
    if text == "all":
        return list(range(dataset.shape[0]))
    if text == "mtd":
        means = class_means(dataset, train_counts)
        limit = max(1, int(min(train_counts)) - 1)
        return maximum_tangent_discrimination(means, max_features=limit)
    try:
        bands = [int(part) for part in text.split(",") if part.strip()]
    except ValueError as error:
        raise SystemExit(f"invalid --bands value: {text!r}") from error
    if not bands or min(bands) < 0 or max(bands) >= dataset.shape[0]:
        raise SystemExit(
            f"band indices must be between 0 and {dataset.shape[0] - 1}"
        )
    return bands


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point of ``python -m hyperspectral``."""
    args = build_parser().parse_args(argv)
    started = time.perf_counter()

    background = [int(v) for v in str(args.background).split(",") if v.strip()]
    cube = load_hypercube(args.data)
    dataset = cube.dataset()

    if not args.quiet:
        print(f"data      : {cube}")
        print(f"cube      : {cube.rows} x {cube.columns} x {cube.bands} bands")
        print(f"classes   : {len(cube.labels)} -> {cube.labels.tolist()}")

    os.makedirs(args.outdir, exist_ok=True)

    if args.task == "kmeans":
        result = kmeans(dataset.T, args.clusters, seed=args.seed)
        cluster_map_2d = result.labels.reshape(cube.rows, cube.columns)
        figure = plot_classification(
            cluster_map_2d, f"K-means ({result.n_clusters} clusters, "
                            f"{result.n_iter} iterations)"
        )
        save_figure(figure, os.path.join(args.outdir, "kmeans_map.png"))
        save_figure(plot_convergence(result.history),
                    os.path.join(args.outdir, "kmeans_convergence.png"))
        report = {
            "task": "kmeans",
            "clusters": result.n_clusters,
            "iterations": result.n_iter,
            "inertia": round(result.inertia, 4),
            "runtime_seconds": round(time.perf_counter() - started, 2),
        }
        print(json.dumps(report, indent=2))
        with open(os.path.join(args.outdir, "kmeans_report.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return 0

    split = split_samples(
        dataset,
        cube.ground_truth,
        samples=args.samples,
        mode="percent" if args.percent else "number",
        background=background,
        seed=args.seed,
        class_names=cube.name_map(),
        skip_insufficient=not args.strict,
    )
    if not args.quiet:
        print(f"samples   : {split.summary()}")
        if split.skipped_classes:
            skipped = ", ".join(
                f"{cube.name_of(label)} ({count} px)"
                for label, count in split.skipped_classes.items()
            )
            print(f"skipped   : {skipped}")

    bands = _parse_bands(args.bands, split.train_dataset, split.train_counts, split.labels)
    train = select_bands(split.train_dataset, bands)
    if not args.quiet:
        print(f"bands     : {len(bands)} used: {bands}")

    classifier = MaximumLikelihoodClassifier(shape=(cube.rows, cube.columns))
    classifier.fit(train, split.train_counts, split.labels)

    prediction = classifier.predict(select_bands(dataset, bands)).reshape(
        cube.rows, cube.columns
    )
    metrics = classifier.score(
        split.test_map,
        select_bands(dataset, bands),
        ignore=tuple(background),
        class_names=dict(zip(split.labels, split.class_names)) if split.class_names else None,
    )

    save_figure(plot_samples(split), os.path.join(args.outdir, "samples.png"))
    save_figure(
        plot_classification(
            prediction,
            f"Maximum Likelihood (OA = {metrics['overall_accuracy']:.2f} %)",
            ground_truth=cube.ground_truth,
        ),
        os.path.join(args.outdir, "classification_map.png"),
    )

    report = {
        "task": "mlc",
        "data": cube.path,
        "bands_used": len(bands),
        "band_indices": bands,
        "overall_accuracy": round(float(metrics["overall_accuracy"]), 2),
        "average_accuracy": round(float(metrics["average_accuracy"]), 2),
        "kappa": round(float(metrics["kappa"]), 4),
        "per_class_accuracy": {
            name: round(float(value), 2) for name, value in metrics["per_class"].items()
        },
        "train_samples": split.n_train,
        "test_samples": split.n_test,
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    print(json.dumps(report, indent=2))
    with open(os.path.join(args.outdir, "mlc_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
