"""Command-line entry points for demos and benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as SkKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture as SkGaussianMixture

from clustering import (
    DBSCAN,
    AgglomerativeClustering,
    GaussianMixtureModel,
    KMeans,
    SoftKMeans,
)

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "mv": ROOT / "data" / "mv.csv",
    "unif": ROOT / "data" / "unif.csv",
    "mv2": ROOT / "data" / "mv2.csv",
    "mv3": ROOT / "data" / "mv3.csv",
}


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, index_col=0)
    return df[["x", "y"]].to_numpy(dtype=float), df["cat"].to_numpy()


def run_benchmark(*, quick: bool = False) -> pd.DataFrame:
    """Compare custom estimators against sklearn (and density/hierarchical methods)."""
    names = ["mv3"] if quick else list(DATASETS)
    rows: list[dict[str, object]] = []

    for name in names:
        points, labels_true = _load(DATASETS[name])

        km = KMeans(n_clusters=3, random_state=0).fit(points)
        sk_km = SkKMeans(n_clusters=3, n_init=10, random_state=0).fit(points)
        soft = SoftKMeans(n_clusters=3, beta=0.3, random_state=0).fit(points)
        gmm = GaussianMixtureModel(n_components=3, random_state=0).fit(points)
        sk_gmm = SkGaussianMixture(n_components=3, random_state=0).fit(points)
        agg = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(points)

        # DBSCAN hyperparameters tuned lightly per dataset geometry.
        eps = 0.35 if name == "mv3" else 0.25
        db = DBSCAN(eps=eps, min_samples=12).fit(points)

        rows.extend(
            [
                {
                    "dataset": name,
                    "model": "custom_kmeans",
                    "ari": adjusted_rand_score(labels_true, km.labels_),
                    "iters": km.n_iter_,
                },
                {
                    "dataset": name,
                    "model": "sklearn_kmeans",
                    "ari": adjusted_rand_score(labels_true, sk_km.labels_),
                    "iters": sk_km.n_iter_,
                },
                {
                    "dataset": name,
                    "model": "soft_kmeans",
                    "ari": adjusted_rand_score(labels_true, soft.labels_),
                    "iters": soft.n_iter_,
                },
                {
                    "dataset": name,
                    "model": "custom_gmm",
                    "ari": adjusted_rand_score(labels_true, gmm.labels_),
                    "iters": gmm.n_iter_,
                },
                {
                    "dataset": name,
                    "model": "sklearn_gmm",
                    "ari": adjusted_rand_score(labels_true, sk_gmm.predict(points)),
                    "iters": sk_gmm.n_iter_,
                },
                {
                    "dataset": name,
                    "model": "agglomerative",
                    "ari": adjusted_rand_score(labels_true, agg.labels_),
                    "iters": None,
                },
                {
                    "dataset": name,
                    "model": "dbscan",
                    "ari": adjusted_rand_score(labels_true, db.labels_),
                    "iters": None,
                },
            ]
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cluster-bench",
        description="Benchmark from-scratch clustering algorithms on the bundled datasets.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only on mv3 for a fast CI smoke test.",
    )
    args = parser.parse_args(argv)
    frame = run_benchmark(quick=args.quick)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 12)
    print(frame.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))


if __name__ == "__main__":
    main()
