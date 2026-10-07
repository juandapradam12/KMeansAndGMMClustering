"""Clustering quality and comparison utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def pairwise_squared_distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return squared Euclidean distances of shape ``(n_samples, n_centers)``."""
    points = np.asarray(points, dtype=float)
    centers = np.asarray(centers, dtype=float)
    # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
    points_sq = np.sum(points**2, axis=1, keepdims=True)
    centers_sq = np.sum(centers**2, axis=1)
    return np.maximum(points_sq + centers_sq - 2.0 * points @ centers.T, 0.0)


def inertia(points: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """Within-cluster sum of squared distances."""
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels)
    centers = np.asarray(centers, dtype=float)
    assigned = centers[labels]
    return float(np.sum((points - assigned) ** 2))


def clustering_report(
    points: np.ndarray,
    labels_true: np.ndarray | None,
    labels_pred: np.ndarray,
    centers: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute a compact evaluation dictionary for a clustering result."""
    points = np.asarray(points, dtype=float)
    labels_pred = np.asarray(labels_pred)
    report: dict[str, Any] = {
        "n_samples": int(points.shape[0]),
        "n_clusters": int(len(np.unique(labels_pred))),
    }

    if centers is not None:
        report["inertia"] = inertia(points, labels_pred, centers)

    n_labels = len(np.unique(labels_pred))
    if 1 < n_labels < len(points):
        report["silhouette"] = float(silhouette_score(points, labels_pred))
    else:
        report["silhouette"] = None

    if labels_true is not None:
        labels_true = np.asarray(labels_true)
        report["adjusted_rand_index"] = float(
            adjusted_rand_score(labels_true, labels_pred)
        )
        report["normalized_mutual_info"] = float(
            normalized_mutual_info_score(labels_true, labels_pred)
        )

    return report
