"""Hard and soft K-Means clustering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .initialization import kmeans_plusplus
from .metrics import pairwise_squared_distances


@dataclass
class KMeans:
    """Vectorized K-Means with K-Means++ initialization.

    Parameters
    ----------
    n_clusters:
        Number of clusters.
    max_iter:
        Maximum EM-style assign/update iterations.
    tol:
        Stop when the maximum centroid movement falls below this value.
    random_state:
        Seed for reproducible initialization.
    init:
        ``\"k-means++\"`` (default) or ``\"random\"``.
    """

    n_clusters: int = 3
    max_iter: int = 300
    tol: float = 1e-4
    random_state: int | None = None
    init: str = "k-means++"

    cluster_centers_: np.ndarray | None = field(default=None, init=False, repr=False)
    labels_: np.ndarray | None = field(default=None, init=False, repr=False)
    inertia_: float | None = field(default=None, init=False, repr=False)
    n_iter_: int = field(default=0, init=False, repr=False)

    def _init_centers(self, points: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        if self.init == "k-means++":
            return kmeans_plusplus(
                points, self.n_clusters, random_state=self.random_state
            )
        if self.init == "random":
            idx = rng.choice(len(points), size=self.n_clusters, replace=False)
            return points[idx].copy()
        raise ValueError("init must be 'k-means++' or 'random'")

    def fit(self, points: np.ndarray, initial_centers: np.ndarray | None = None) -> "KMeans":
        points = np.asarray(points, dtype=float)
        centers = (
            np.asarray(initial_centers, dtype=float).copy()
            if initial_centers is not None
            else self._init_centers(points)
        )

        labels = np.zeros(len(points), dtype=int)
        for iteration in range(1, self.max_iter + 1):
            distances = pairwise_squared_distances(points, centers)
            labels = np.argmin(distances, axis=1)

            new_centers = centers.copy()
            for k in range(self.n_clusters):
                members = points[labels == k]
                if len(members) == 0:
                    # Re-seed empty clusters with the farthest point.
                    farthest = int(np.argmax(np.min(distances, axis=1)))
                    new_centers[k] = points[farthest]
                else:
                    new_centers[k] = members.mean(axis=0)

            shift = np.linalg.norm(new_centers - centers, axis=1).max()
            centers = new_centers
            self.n_iter_ = iteration
            if shift <= self.tol:
                break

        distances = pairwise_squared_distances(points, centers)
        labels = np.argmin(distances, axis=1)
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = float(np.sum(distances[np.arange(len(points)), labels]))
        return self

    def predict(self, points: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("Model has not been fit yet")
        distances = pairwise_squared_distances(np.asarray(points, dtype=float), self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def fit_predict(
        self, points: np.ndarray, initial_centers: np.ndarray | None = None
    ) -> np.ndarray:
        return self.fit(points, initial_centers=initial_centers).labels_


@dataclass
class SoftKMeans:
    """Soft (fuzzy) K-Means with exponential responsibilities.

    Responsibility of point ``i`` for cluster ``k``:

    .. math::

        \\phi_i(k) =
        \\frac{\\exp(-\\|x_i-\\mu_k\\|^2 / \\beta)}
             {\\sum_j \\exp(-\\|x_i-\\mu_j\\|^2 / \\beta)}

    ``beta`` controls softness: smaller values approach hard K-Means.
    """

    n_clusters: int = 3
    beta: float = 0.3
    max_iter: int = 300
    tol: float = 1e-4
    random_state: int | None = None
    init: str = "k-means++"

    cluster_centers_: np.ndarray | None = field(default=None, init=False, repr=False)
    responsibilities_: np.ndarray | None = field(default=None, init=False, repr=False)
    labels_: np.ndarray | None = field(default=None, init=False, repr=False)
    n_iter_: int = field(default=0, init=False, repr=False)

    def _init_centers(self, points: np.ndarray) -> np.ndarray:
        if self.init == "k-means++":
            return kmeans_plusplus(
                points, self.n_clusters, random_state=self.random_state
            )
        if self.init == "random":
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(points), size=self.n_clusters, replace=False)
            return points[idx].copy()
        raise ValueError("init must be 'k-means++' or 'random'")

    def _responsibilities(self, points: np.ndarray, centers: np.ndarray) -> np.ndarray:
        # Numerically stable softmax over -dist^2 / beta.
        logits = -pairwise_squared_distances(points, centers) / self.beta
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=1, keepdims=True)

    def fit(
        self, points: np.ndarray, initial_centers: np.ndarray | None = None
    ) -> "SoftKMeans":
        points = np.asarray(points, dtype=float)
        centers = (
            np.asarray(initial_centers, dtype=float).copy()
            if initial_centers is not None
            else self._init_centers(points)
        )

        responsibilities = None
        for iteration in range(1, self.max_iter + 1):
            responsibilities = self._responsibilities(points, centers)
            mass = responsibilities.sum(axis=0)
            # Avoid divide-by-zero for collapsed clusters.
            mass = np.maximum(mass, 1e-12)
            new_centers = (responsibilities.T @ points) / mass[:, None]

            shift = np.linalg.norm(new_centers - centers, axis=1).max()
            centers = new_centers
            self.n_iter_ = iteration
            if shift <= self.tol:
                break

        self.cluster_centers_ = centers
        self.responsibilities_ = responsibilities
        self.labels_ = np.argmax(responsibilities, axis=1)
        return self

    def predict_proba(self, points: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("Model has not been fit yet")
        return self._responsibilities(np.asarray(points, dtype=float), self.cluster_centers_)

    def predict(self, points: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(points), axis=1)

    def fit_predict(
        self, points: np.ndarray, initial_centers: np.ndarray | None = None
    ) -> np.ndarray:
        return self.fit(points, initial_centers=initial_centers).labels_
