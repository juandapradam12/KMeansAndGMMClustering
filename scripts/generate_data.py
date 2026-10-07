#!/usr/bin/env python3
"""Regenerate synthetic clustering datasets (approximate originals).

The checked-in CSVs under ``data/`` are the canonical demo artifacts. This
script documents and recreates the same *family* of geometries with a fixed
seed so the project is fully reproducible end-to-end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _labeled(points: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "cat": labels})


def make_mv(rng: np.random.Generator, n_per: int = 1000) -> pd.DataFrame:
    """Three overlapping Gaussians with mild tilt (K-Means struggles, GMM helps)."""
    means = [np.array([-1.0, 0.0]), np.array([0.5, 0.0]), np.array([2.0, 0.0])]
    covs = [
        np.array([[0.31, -0.32], [-0.32, 0.78]]),
        np.array([[0.24, 0.0], [0.0, 0.23]]),
        np.array([[0.11, 0.0], [0.0, 0.76]]),
    ]
    chunks = [rng.multivariate_normal(m, c, size=n_per) for m, c in zip(means, covs, strict=False)]
    points = np.vstack(chunks)
    labels = np.repeat(np.arange(3), n_per)
    return _labeled(points, labels)


def make_mv2(rng: np.random.Generator, n_per: int = 1000) -> pd.DataFrame:
    """Three elongated / correlated Gaussians — strong GMM use case."""
    means = [np.array([-1.0, 0.0]), np.array([0.5, 0.0]), np.array([2.0, 0.0])]
    covs = [
        np.array([[0.22, -0.35], [-0.35, 0.81]]),
        np.array([[0.16, 0.31], [0.31, 0.84]]),
        np.array([[0.23, -0.34], [-0.34, 0.77]]),
    ]
    chunks = [rng.multivariate_normal(m, c, size=n_per) for m, c in zip(means, covs, strict=False)]
    points = np.vstack(chunks)
    labels = np.repeat(np.arange(3), n_per)
    return _labeled(points, labels)


def make_mv3(rng: np.random.Generator, n_per: int = 1000) -> pd.DataFrame:
    """Well-separated near-spherical blobs — friendly to K-Means."""
    means = [np.array([-2.0, -2.0]), np.array([2.0, -2.0]), np.array([0.0, 1.5])]
    covs = [
        np.array([[0.31, 0.0], [0.0, 0.29]]),
        np.array([[0.29, 0.0], [0.0, 0.28]]),
        np.array([[0.68, 0.0], [0.0, 0.72]]),
    ]
    chunks = [rng.multivariate_normal(m, c, size=n_per) for m, c in zip(means, covs, strict=False)]
    points = np.vstack(chunks)
    labels = np.repeat(np.arange(3), n_per)
    return _labeled(points, labels)


def make_unif(rng: np.random.Generator, n_per: int = 1000) -> pd.DataFrame:
    """Three overlapping uniform rectangles — intentionally difficult."""
    # Vertical strip left, horizontal band center, vertical strip right.
    c0 = np.column_stack(
        [rng.uniform(-1.0, -0.5, n_per), rng.uniform(-2.0, 2.0, n_per)]
    )
    c1 = np.column_stack(
        [rng.uniform(-1.2, 1.2, n_per), rng.uniform(-0.5, 0.5, n_per)]
    )
    c2 = np.column_stack(
        [rng.uniform(0.5, 1.0, n_per), rng.uniform(-2.0, 2.0, n_per)]
    )
    points = np.vstack([c0, c1, c2])
    labels = np.repeat(np.arange(3), n_per)
    return _labeled(points, labels)


GENERATORS = {
    "mv": make_mv,
    "mv2": make_mv2,
    "mv3": make_mv3,
    "unif": make_unif,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-class", type=int, default=1000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory for CSV outputs (defaults to data/).",
    )
    parser.add_argument(
        "--overwrite-canonical",
        action="store_true",
        help="Overwrite the checked-in CSVs. Off by default; writes *-generated.csv instead.",
    )
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, fn in GENERATORS.items():
        df = fn(rng, n_per=args.n_per_class)
        filename = f"{name}.csv" if args.overwrite_canonical else f"{name}-generated.csv"
        path = args.out_dir / filename
        df.to_csv(path)
        print(f"Wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
