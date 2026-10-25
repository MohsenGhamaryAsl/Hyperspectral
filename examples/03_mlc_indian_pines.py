"""Full pipeline: MTD band selection + Maximum Likelihood Classification.

Run:  python examples/03_mlc_indian_pines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # make the package importable without installing
    sys.path.insert(0, str(ROOT))

import json
from hyperspectral.datasets import class_means
from hyperspectral.features import maximum_tangent_discrimination, select_bands
from hyperspectral.io import load_hypercube
from hyperspectral.mlc import MaximumLikelihoodClassifier
from hyperspectral.plotting import plot_classification, save_figure
from hyperspectral.split import split_samples

DATA = Path(__file__).resolve().parents[1] / "data" / "Indian_Pine.mat"
OUT = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    cube = load_hypercube(str(DATA))
    dataset = cube.dataset()

    # 1) reproducible training/test split: 20 % of every class
    split = split_samples(
        dataset, cube.ground_truth, samples=20, mode="percent",
        seed=0, class_names=cube.name_map(),
    )

    # 2) band selection on the class-mean spectra
    means = class_means(split.train_dataset, split.train_counts)
    bands = maximum_tangent_discrimination(means)
    print(f"selected {len(bands)} of {cube.bands} bands: {bands}")

    # 3) train and classify
    classifier = MaximumLikelihoodClassifier(shape=(cube.rows, cube.columns))
    classifier.fit(select_bands(split.train_dataset, bands),
                   split.train_counts, split.labels)

    prediction = classifier.predict_map(select_bands(dataset, bands))
    metrics = classifier.score(
        split.test_map, select_bands(dataset, bands),
        class_names=dict(zip(split.labels, split.class_names)),
    )

    # 4) report
    print(f"\nOA    : {metrics['overall_accuracy']:.2f} %")
    print(f"AA    : {metrics['average_accuracy']:.2f} %")
    print(f"kappa : {metrics['kappa']:.4f}")
    print(f"tested: {metrics['n_samples']} pixels")
    print("\nper-class accuracy")
    for name, value in metrics["per_class"].items():
        print(f"  {name:26s} {value:6.2f} %")

    save_figure(
        plot_classification(
            prediction,
            f"Maximum Likelihood (OA = {metrics['overall_accuracy']:.2f} %)",
            ground_truth=cube.ground_truth,
        ),
        str(OUT / "classification_map.png"),
    )

    report = {
        "bands": bands,
        "overall_accuracy": round(float(metrics["overall_accuracy"]), 2),
        "average_accuracy": round(float(metrics["average_accuracy"]), 2),
        "kappa": round(float(metrics["kappa"]), 4),
        "per_class_accuracy": {
            name: round(float(value), 2) for name, value in metrics["per_class"].items()
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mlc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nresults written to {OUT}")


if __name__ == "__main__":
    main()
