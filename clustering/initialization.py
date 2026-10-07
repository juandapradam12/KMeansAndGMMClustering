"""Centroid initialization strategies."""

from __future__ import annotations

import numpy as np


def kmeans_plusplus(
    points: np.ndarray,
    n_clusters: int,
    *,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Select initial centroids with the K-Means++ procedure.

    The probability of selecting a candidate point is proportional to its
    squared Euclidean distance to the nearest already-chosen centroid.

    Parameters
    ----------
    points:
        Array of shape ``(n_samples, n_features)``.
    n_clusters:
        Number of centroids to return.
    random_state:
        Seed or NumPy generator for reproducibility.

    Returns
    -------
    centroids:
        Array of shape ``(n_clusters, n_features)``.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2-D array")
    n_samples = points.shape[0]
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if n_clusters > n_samples:
        raise ValueError("n_clusters cannot exceed the number of samples")

    rng = np.random.default_rng(random_state)
    centroids = np.empty((n_clusters, points.shape[1]), dtype=float)

    # First center chosen uniformly.
    first_idx = int(rng.integers(0, n_samples))
    centroids[0] = points[first_idx]

    # Closest squared distance from each point to the chosen centers so far.
    closest_sq = np.full(n_samples, np.inf)
    for k in range(1, n_clusters):
        diff = points - centroids[k - 1]
        dist_sq = np.einsum("ij,ij->i", diff, diff)
        closest_sq = np.minimum(closest_sq, dist_sq)

        total = closest_sq.sum()
        if total <= 0 or not np.isfinite(total):
            # Degenerate case: remaining points coincide with existing centers.
            remaining = np.setdiff1d(
                np.arange(n_samples),
                np.flatnonzero(closest_sq == 0),
                assume_unique=False,
            )
            if len(remaining) == 0:
                centroids[k:] = points[rng.choice(n_samples, size=n_clusters - k)]
                break
            idx = int(rng.choice(remaining))
        else:
            probs = closest_sq / total
            idx = int(rng.choice(n_samples, p=probs))
        centroids[k] = points[idx]

    return centroids
