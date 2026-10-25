"""Smoke tests of the command line interface."""

from __future__ import annotations

import json

import numpy as np

from hyperspectral.cli import main
from hyperspectral.io import save_hypercube


def test_cli_mlc(tmp_path, synthetic_cube):
    data_path = tmp_path / "cube.npz"
    save_hypercube(str(data_path), synthetic_cube)
    outdir = tmp_path / "results"

    exit_code = main([
        "--data", str(data_path),
        "--samples", "12",
        "--outdir", str(outdir),
        "--quiet",
    ])
    assert exit_code == 0

    report = json.loads((outdir / "mlc_report.json").read_text())
    assert report["task"] == "mlc"
    assert report["overall_accuracy"] > 90.0
    assert (outdir / "classification_map.png").is_file()
    assert (outdir / "samples.png").is_file()


def test_cli_kmeans(tmp_path, synthetic_cube):
    data_path = tmp_path / "cube.npz"
    save_hypercube(str(data_path), synthetic_cube)
    outdir = tmp_path / "clusters"

    exit_code = main([
        "--data", str(data_path),
        "--task", "kmeans",
        "--clusters", "3",
        "--outdir", str(outdir),
        "--quiet",
    ])
    assert exit_code == 0
    report = json.loads((outdir / "kmeans_report.json").read_text())
    assert report["clusters"] == 3
    assert report["iterations"] >= 1
    assert (outdir / "kmeans_map.png").is_file()


def test_cli_all_bands(tmp_path, synthetic_cube):
    data_path = tmp_path / "cube.npz"
    save_hypercube(str(data_path), synthetic_cube)
    outdir = tmp_path / "allbands"
    exit_code = main([
        "--data", str(data_path), "--samples", "10", "--bands", "all",
        "--outdir", str(outdir), "--quiet",
    ])
    assert exit_code == 0
