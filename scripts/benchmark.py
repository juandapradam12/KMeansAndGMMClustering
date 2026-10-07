#!/usr/bin/env python3
"""Benchmark custom clustering algorithms against scikit-learn."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans as SkKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture as SkGaussianMixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clustering import GaussianMixtureModel, KMeans, SoftKMeans  # noqa: E402


DATASETS = {
    "mv": ROOT / "data" / "mv.csv",
    "unif": ROOT / "data" / "unif.csv",
    "mv2": ROOT / "data" / "mv2.csv",
    "mv3": ROOT / "data" / "mv3.csv",
}


def load(path: Path):
    df = pd.read_csv(path, index_col=0)
    return df[["x", "y"]].to_numpy(dtype=float), df["cat"].to_numpy(), df


def main() -> None:
    rows = []
    for name, path in DATASETS.items():
        points, labels_true, _ = load(path)

        km = KMeans(n_clusters=3, random_state=0).fit(points)
        sk_km = SkKMeans(n_clusters=3, n_init=10, random_state=0).fit(points)
        soft = SoftKMeans(n_clusters=3, beta=0.3, random_state=0).fit(points)
        gmm = GaussianMixtureModel(n_components=3, random_state=0).fit(points)
        sk_gmm = SkGaussianMixture(n_components=3, random_state=0).fit(points)

        rows.extend(
            [
                {
                    "dataset": name,
                    "model": "custom_kmeans",
                    "ari": adjusted_rand_score(labels_true, km.labels_),
                    "inertia": km.inertia_,
                    "iters": km.n_iter_,
                },
                {
                    "dataset": name,
                    "model": "sklearn_kmeans",
                    "ari": adjusted_rand_score(labels_true, sk_km.labels_),
                    "inertia": float(sk_km.inertia_),
                    "iters": sk_km.n_iter_,
                },
                {
                    "dataset": name,
                    "model": "soft_kmeans",
                    "ari": adjusted_rand_score(labels_true, soft.labels_),
                    "inertia": None,
                    "iters": soft.n_iter_,
                },
                {
                    "dataset": name,
                    "model": "custom_gmm",
                    "ari": adjusted_rand_score(labels_true, gmm.labels_),
                    "inertia": None,
                    "iters": gmm.n_iter_,
                },
                {
                    "dataset": name,
                    "model": "sklearn_gmm",
                    "ari": adjusted_rand_score(labels_true, sk_gmm.predict(points)),
                    "inertia": None,
                    "iters": sk_gmm.n_iter_,
                },
            ]
        )

    frame = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 10)
    print(frame.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))


if __name__ == "__main__":
    main()
