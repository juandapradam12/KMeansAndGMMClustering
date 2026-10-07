"""Density-Based Spatial Clustering of Applications with Noise (DBSCAN)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import KDTree


@dataclass
class DBSCAN:
    """From-scratch DBSCAN with a KD-tree neighbor query.

    Parameters
    ----------
    eps:
        Neighborhood radius.
    min_samples:
        Minimum neighbors (including the point itself) to form a core point.
    metric:
        Currently only ``\"euclidean\"`` is supported.
    """

    eps: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"

    labels_: np.ndarray | None = field(default=None, init=False, repr=False)
    core_sample_indices_: np.ndarray | None = field(default=None, init=False, repr=False)
    n_clusters_: int = field(default=0, init=False, repr=False)

    def fit(self, points: np.ndarray) -> DBSCAN:
        if self.metric != "euclidean":
            raise ValueError("Only metric='euclidean' is supported")
        points = np.asarray(points, dtype=float)
        n_samples = len(points)
        tree = KDTree(points)
        neighbors = tree.query_ball_point(points, r=self.eps)

        labels = np.full(n_samples, -1, dtype=int)
        core_mask = np.array([len(nbrs) >= self.min_samples for nbrs in neighbors])
        cluster_id = 0

        for i in range(n_samples):
            if labels[i] != -1 or not core_mask[i]:
                continue

            # Expand a new cluster from core point i (BFS).
            labels[i] = cluster_id
            seed = list(neighbors[i])
            in_seed = set(seed)
            j = 0
            while j < len(seed):
                q = seed[j]
                if labels[q] == -1:
                    labels[q] = cluster_id
                    if core_mask[q]:
                        for nbr in neighbors[q]:
                            if nbr not in in_seed:
                                in_seed.add(nbr)
                                seed.append(nbr)
                j += 1
            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.flatnonzero(core_mask)
        self.n_clusters_ = cluster_id
        return self

    def fit_predict(self, points: np.ndarray) -> np.ndarray:
        self.fit(points)
        assert self.labels_ is not None
        return self.labels_
