"""Agglomerative hierarchical clustering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

LinkageName = Literal["ward", "average", "complete", "single"]


@dataclass
class AgglomerativeClustering:
    """Bottom-up hierarchical clustering via SciPy linkage.

    This wrapper keeps the same educational API as the other estimators while
    exposing classic linkage choices. Ward linkage requires Euclidean space.

    Parameters
    ----------
    n_clusters:
        Number of flat clusters to extract from the dendrogram.
    linkage:
        ``ward``, ``average``, ``complete``, or ``single``.
    """

    n_clusters: int = 3
    linkage: LinkageName = "ward"

    labels_: np.ndarray | None = field(default=None, init=False, repr=False)
    children_: np.ndarray | None = field(default=None, init=False, repr=False)
    distances_: np.ndarray | None = field(default=None, init=False, repr=False)
    n_clusters_: int = field(default=0, init=False, repr=False)

    def fit(self, points: np.ndarray) -> AgglomerativeClustering:
        points = np.asarray(points, dtype=float)
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if self.n_clusters > len(points):
            raise ValueError("n_clusters cannot exceed n_samples")

        condensed = pdist(points, metric="euclidean")
        z = linkage(condensed, method=self.linkage)
        # SciPy linkage matrix: [i, j, dist, count]
        labels = fcluster(z, t=self.n_clusters, criterion="maxclust") - 1

        self.labels_ = labels.astype(int)
        self.children_ = z[:, :2].astype(int)
        self.distances_ = z[:, 2]
        self.n_clusters_ = self.n_clusters
        return self

    def fit_predict(self, points: np.ndarray) -> np.ndarray:
        self.fit(points)
        assert self.labels_ is not None
        return self.labels_
