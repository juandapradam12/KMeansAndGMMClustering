"""Gaussian Mixture Model fitted with Expectation-Maximization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import multivariate_normal

from .initialization import kmeans_plusplus
from .kmeans import KMeans


@dataclass
class GaussianMixtureModel:
    """From-scratch Gaussian Mixture Model (EM).

    Each component is parameterized by ``(mean, weight, covariance)``.
    Means are initialized with K-Means++ (optionally refined by a short
    K-Means warm start), weights are uniform, and covariances start as
    identity matrices.

    Parameters
    ----------
    n_components:
        Number of Gaussian components.
    max_iter:
        Maximum EM iterations.
    tol:
        Absolute tolerance on the mean parameter movement.
    reg_covar:
        Diagonal ridge added to each covariance for numerical stability.
    random_state:
        Seed for reproducible initialization.
    warm_start_kmeans:
        If True, refine K-Means++ seeds with a short hard K-Means run.
    """

    n_components: int = 3
    max_iter: int = 300
    tol: float = 1e-4
    reg_covar: float = 1e-6
    random_state: int | None = None
    warm_start_kmeans: bool = True

    means_: np.ndarray | None = field(default=None, init=False, repr=False)
    weights_: np.ndarray | None = field(default=None, init=False, repr=False)
    covariances_: np.ndarray | None = field(default=None, init=False, repr=False)
    responsibilities_: np.ndarray | None = field(default=None, init=False, repr=False)
    labels_: np.ndarray | None = field(default=None, init=False, repr=False)
    lower_bound_: float | None = field(default=None, init=False, repr=False)
    n_iter_: int = field(default=0, init=False, repr=False)

    def _initialize(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_features = points.shape[1]
        means = kmeans_plusplus(
            points, self.n_components, random_state=self.random_state
        )
        if self.warm_start_kmeans:
            km = KMeans(
                n_clusters=self.n_components,
                random_state=self.random_state,
                max_iter=25,
            ).fit(points, initial_centers=means)
            assert km.cluster_centers_ is not None
            means = km.cluster_centers_

        weights = np.full(self.n_components, 1.0 / self.n_components)
        covariances = np.array(
            [np.eye(n_features) for _ in range(self.n_components)], dtype=float
        )
        return means, weights, covariances

    def _e_step(
        self,
        points: np.ndarray,
        means: np.ndarray,
        weights: np.ndarray,
        covariances: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        n_samples = points.shape[0]
        log_prob = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            # allow_singular helps when clusters temporarily collapse.
            rv = multivariate_normal(
                mean=means[k],
                cov=covariances[k],
                allow_singular=True,
            )
            log_prob[:, k] = np.log(weights[k] + 1e-16) + rv.logpdf(points)

        # Log-sum-exp normalization for responsibilities.
        max_log = np.max(log_prob, axis=1, keepdims=True)
        stabilized = np.exp(log_prob - max_log)
        denom = stabilized.sum(axis=1, keepdims=True)
        responsibilities = stabilized / denom
        log_likelihood = float(np.sum(max_log.ravel() + np.log(denom.ravel())))
        return responsibilities, log_likelihood

    def _m_step(
        self, points: np.ndarray, responsibilities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_features = points.shape
        nk = responsibilities.sum(axis=0) + 1e-12
        weights = nk / n_samples
        means = (responsibilities.T @ points) / nk[:, None]

        covariances = np.empty((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = points - means[k]
            weighted = responsibilities[:, k][:, None] * diff
            cov = (weighted.T @ diff) / nk[k]
            cov.flat[:: n_features + 1] += self.reg_covar
            covariances[k] = cov
        return means, weights, covariances

    def fit(
        self,
        points: np.ndarray,
        initial_means: np.ndarray | None = None,
    ) -> GaussianMixtureModel:
        points = np.asarray(points, dtype=float)
        if initial_means is not None:
            means = np.asarray(initial_means, dtype=float).copy()
            weights = np.full(self.n_components, 1.0 / self.n_components)
            covariances = np.array(
                [np.eye(points.shape[1]) for _ in range(self.n_components)],
                dtype=float,
            )
        else:
            means, weights, covariances = self._initialize(points)

        prev_ll = -np.inf
        responsibilities = None
        log_likelihood = prev_ll
        for iteration in range(1, self.max_iter + 1):
            responsibilities, log_likelihood = self._e_step(
                points, means, weights, covariances
            )
            new_means, weights, covariances = self._m_step(points, responsibilities)

            mean_shift = np.linalg.norm(new_means - means, axis=1).max()
            means = new_means
            self.n_iter_ = iteration

            if abs(log_likelihood - prev_ll) <= self.tol and mean_shift <= self.tol:
                break
            prev_ll = log_likelihood

        if responsibilities is None:
            responsibilities, log_likelihood = self._e_step(
                points, means, weights, covariances
            )
        self.means_ = means
        self.weights_ = weights
        self.covariances_ = covariances
        self.responsibilities_ = responsibilities
        self.labels_ = np.argmax(responsibilities, axis=1)
        self.lower_bound_ = log_likelihood
        return self

    def predict_proba(self, points: np.ndarray) -> np.ndarray:
        if self.means_ is None or self.weights_ is None or self.covariances_ is None:
            raise RuntimeError("Model has not been fit yet")
        responsibilities, _ = self._e_step(
            np.asarray(points, dtype=float),
            self.means_,
            self.weights_,
            self.covariances_,
        )
        return responsibilities

    def predict(self, points: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(points), axis=1)

    def fit_predict(
        self, points: np.ndarray, initial_means: np.ndarray | None = None
    ) -> np.ndarray:
        self.fit(points, initial_means=initial_means)
        assert self.labels_ is not None
        return self.labels_

    def bic(self, points: np.ndarray) -> float:
        """Bayesian Information Criterion (lower is better)."""
        points = np.asarray(points, dtype=float)
        if self.lower_bound_ is None or self.means_ is None:
            raise RuntimeError("Model has not been fit yet")
        n_samples, n_features = points.shape
        # Free params: means + unique cov entries + mixing weights - 1
        n_params = (
            self.n_components * n_features
            + self.n_components * n_features * (n_features + 1) / 2
            + (self.n_components - 1)
        )
        return float(-2.0 * self.lower_bound_ + n_params * np.log(n_samples))

    def aic(self, points: np.ndarray) -> float:
        """Akaike Information Criterion (lower is better)."""
        points = np.asarray(points, dtype=float)
        if self.lower_bound_ is None or self.means_ is None:
            raise RuntimeError("Model has not been fit yet")
        n_features = points.shape[1]
        n_params = (
            self.n_components * n_features
            + self.n_components * n_features * (n_features + 1) / 2
            + (self.n_components - 1)
        )
        return float(-2.0 * self.lower_bound_ + 2.0 * n_params)

    @property
    def clusters(self) -> list[tuple[np.ndarray, float, np.ndarray]]:
        """Legacy tuple view: ``(mu, pi, Sigma)`` per component."""
        if self.means_ is None or self.weights_ is None or self.covariances_ is None:
            raise RuntimeError("Model has not been fit yet")
        return [
            (self.means_[k], float(self.weights_[k]), self.covariances_[k])
            for k in range(self.n_components)
        ]
