"""Plotting helpers for clustering demos."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from sklearn.cluster import KMeans as SkKMeans
from sklearn.mixture import GaussianMixture as SkGaussianMixture

from .gmm import GaussianMixtureModel
from .kmeans import KMeans

PALETTE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def plot_datasets(
    dataframes: Sequence[pd.DataFrame],
    titles: Sequence[str] | None = None,
    *,
    figsize: tuple[float, float] = (8, 8),
) -> plt.Figure:
    """Scatter-plot labeled 2-D datasets in a 2x2 grid."""
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    for i, (ax, df) in enumerate(zip(axs.flatten(), dataframes, strict=False)):
        for cat, col in zip(df["cat"].unique(), PALETTE, strict=False):
            subset = df[df.cat == cat]
            ax.scatter(subset.x, subset.y, c=col, label=str(cat), alpha=0.25, s=12)
        ax.legend(markerscale=1.5, fontsize=8)
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.tight_layout()
    return fig


def plot_kmeans_comparison(
    dataframes: Sequence[pd.DataFrame],
    *,
    n_clusters: int = 3,
    random_state: int = 0,
    figsize: tuple[float, float] = (8, 8),
) -> plt.Figure:
    """Compare custom K-Means centers against scikit-learn."""
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    for ax, df in zip(axs.flatten(), dataframes, strict=False):
        points = df.iloc[:, :2].to_numpy(dtype=float)
        custom = KMeans(
            n_clusters=n_clusters, random_state=random_state
        ).fit(points)
        sk = SkKMeans(
            n_clusters=n_clusters,
            init=custom.cluster_centers_,
            n_init=1,
            random_state=random_state,
        ).fit(points)

        for cat, col in zip(df["cat"].unique(), PALETTE, strict=False):
            subset = df[df.cat == cat]
            ax.scatter(subset.x, subset.y, c=col, alpha=0.2, s=12)
        assert custom.cluster_centers_ is not None
        ax.scatter(
            custom.cluster_centers_[:, 0],
            custom.cluster_centers_[:, 1],
            c="k",
            marker="x",
            s=90,
            label="Custom",
        )
        ax.scatter(
            sk.cluster_centers_[:, 0],
            sk.cluster_centers_[:, 1],
            c="red",
            marker="+",
            s=90,
            label="sklearn",
        )
        ax.legend()
    fig.suptitle(f"K-Means: custom vs sklearn (k={n_clusters})")
    fig.tight_layout()
    return fig


def _blend_colors(responsibilities: np.ndarray, colors: np.ndarray) -> list[str]:
    return [to_hex(responsibilities[i] @ colors) for i in range(len(responsibilities))]


def plot_soft_assignments(
    dataframes: Sequence[pd.DataFrame],
    *,
    model: str = "gmm",
    n_clusters: int = 3,
    random_state: int = 0,
    figsize: tuple[float, float] = (8, 8),
    use_sklearn: bool = False,
) -> plt.Figure:
    """Color points by hard labels (K-Means) or soft responsibilities (GMM)."""
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    colors = np.array([to_rgb(c) for c in PALETTE[:n_clusters]])

    for ax, df in zip(axs.flatten(), dataframes, strict=False):
        points = df.iloc[:, :2].to_numpy(dtype=float)
        if model == "gmm":
            if use_sklearn:
                fitted = SkGaussianMixture(
                    n_components=n_clusters, random_state=random_state
                ).fit(points)
                probs = fitted.predict_proba(points)
            else:
                fitted = GaussianMixtureModel(
                    n_components=n_clusters, random_state=random_state
                ).fit(points)
                probs = fitted.responsibilities_
            point_colors = _blend_colors(probs, colors)
        elif model == "kmeans":
            if use_sklearn:
                fitted = SkKMeans(
                    n_clusters=n_clusters, n_init=10, random_state=random_state
                ).fit(points)
                labels = fitted.labels_
            else:
                fitted = KMeans(
                    n_clusters=n_clusters, random_state=random_state
                ).fit(points)
                labels = fitted.labels_
            point_colors = colors[labels]
        else:
            raise ValueError("model must be 'gmm' or 'kmeans'")

        ax.scatter(df.x, df.y, c=point_colors, alpha=0.35, s=12)

    backend = "sklearn" if use_sklearn else "custom"
    fig.suptitle(f"{model.upper()} soft/hard assignments ({backend}, k={n_clusters})")
    fig.tight_layout()
    return fig
