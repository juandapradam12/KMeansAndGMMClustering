"""Unit tests for from-scratch clustering implementations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans as SkKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture as SkGaussianMixture

from clustering import (
    GaussianMixtureModel,
    KMeans,
    SoftKMeans,
    clustering_report,
    kmeans_plusplus,
)
from clustering.model_selection import elbow_curve, gmm_information_criteria, suggest_k

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def mv3_points_labels():
    """Well-separated spherical clusters — friendly to K-Means."""
    df = pd.read_csv(DATA_DIR / "mv3.csv", index_col=0)
    return df[["x", "y"]].to_numpy(dtype=float), df["cat"].to_numpy()


@pytest.fixture(scope="module")
def mv2_points_labels():
    """Overlapping / anisotropic clusters — GMM should excel."""
    df = pd.read_csv(DATA_DIR / "mv2.csv", index_col=0)
    return df[["x", "y"]].to_numpy(dtype=float), df["cat"].to_numpy()


def test_kmeans_plusplus_shape_and_uniqueness():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(100, 2))
    centers = kmeans_plusplus(points, 4, random_state=0)
    assert centers.shape == (4, 2)
    assert np.isfinite(centers).all()


def test_kmeans_matches_sklearn_and_recovers_labels(mv3_points_labels):
    points, labels_true = mv3_points_labels
    init = kmeans_plusplus(points, 3, random_state=7)
    custom = KMeans(n_clusters=3, random_state=7).fit(points, initial_centers=init)
    sk = SkKMeans(n_clusters=3, init=init, n_init=1, random_state=7).fit(points)

    ari_custom = adjusted_rand_score(labels_true, custom.labels_)
    ari_between = adjusted_rand_score(custom.labels_, sk.labels_)

    assert ari_custom > 0.95
    assert ari_between > 0.95
    assert abs(custom.inertia_ - sk.inertia_) / sk.inertia_ < 0.02
    assert np.allclose(custom.cluster_centers_, sk.cluster_centers_, atol=0.2)


def test_soft_kmeans_uses_squared_distance_and_converges(mv3_points_labels):
    points, labels_true = mv3_points_labels
    model = SoftKMeans(n_clusters=3, beta=0.3, random_state=0).fit(points)
    assert model.responsibilities_.shape == (len(points), 3)
    assert np.allclose(model.responsibilities_.sum(axis=1), 1.0)
    assert adjusted_rand_score(labels_true, model.labels_) > 0.95


def test_gmm_recovers_anisotropic_clusters(mv2_points_labels):
    points, labels_true = mv2_points_labels
    model = GaussianMixtureModel(n_components=3, random_state=0).fit(points)
    assert model.responsibilities_.shape == (len(points), 3)
    assert np.allclose(model.weights_.sum(), 1.0, atol=1e-5)
    assert adjusted_rand_score(labels_true, model.labels_) > 0.8
    assert model.bic(points) > 0
    assert model.aic(points) > 0


def test_gmm_comparable_to_sklearn(mv2_points_labels):
    points, labels_true = mv2_points_labels
    custom = GaussianMixtureModel(n_components=3, random_state=1).fit(points)
    sk = SkGaussianMixture(n_components=3, random_state=1).fit(points)
    ari_custom = adjusted_rand_score(labels_true, custom.labels_)
    ari_sk = adjusted_rand_score(labels_true, sk.predict(points))
    assert ari_custom > 0.8
    assert abs(ari_custom - ari_sk) < 0.15


def test_empty_cluster_reseed():
    points = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [20.0, 20.0],
        ],
        dtype=float,
    )
    model = KMeans(n_clusters=3, random_state=0, max_iter=50).fit(points)
    assert len(np.unique(model.labels_)) == 3
    assert np.isfinite(model.cluster_centers_).all()


def test_clustering_report_and_model_selection(mv3_points_labels):
    points, labels_true = mv3_points_labels
    model = KMeans(n_clusters=3, random_state=0).fit(points)
    report = clustering_report(
        points, labels_true, model.labels_, model.cluster_centers_
    )
    assert "inertia" in report
    assert report["adjusted_rand_index"] > 0.95

    elbow = elbow_curve(points, range(2, 6), random_state=0)
    assert suggest_k(elbow, "silhouette") in {2, 3, 4, 5}

    ic = gmm_information_criteria(points, range(2, 5), random_state=0)
    assert suggest_k(ic, "bic") in {2, 3, 4}
