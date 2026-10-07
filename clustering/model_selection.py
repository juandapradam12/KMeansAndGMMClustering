"""Helpers for choosing the number of clusters / components."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.metrics import silhouette_score

from .gmm import GaussianMixtureModel
from .kmeans import KMeans


def elbow_curve(
    points: np.ndarray,
    k_range: range | list[int] = range(1, 9),
    *,
    random_state: int | None = 0,
) -> list[dict[str, Any]]:
    """Fit K-Means for each ``k`` and return inertia / silhouette scores."""
    points = np.asarray(points, dtype=float)
    rows: list[dict[str, Any]] = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state).fit(points)
        row: dict[str, Any] = {
            "k": int(k),
            "inertia": model.inertia_,
            "n_iter": model.n_iter_,
        }
        if 1 < k < len(points):
            row["silhouette"] = float(silhouette_score(points, model.labels_))
        else:
            row["silhouette"] = None
        rows.append(row)
    return rows


def gmm_information_criteria(
    points: np.ndarray,
    k_range: range | list[int] = range(1, 9),
    *,
    random_state: int | None = 0,
) -> list[dict[str, Any]]:
    """Fit GMMs across ``k`` and return AIC / BIC for model selection."""
    points = np.asarray(points, dtype=float)
    rows: list[dict[str, Any]] = []
    for k in k_range:
        model = GaussianMixtureModel(
            n_components=k, random_state=random_state
        ).fit(points)
        rows.append(
            {
                "k": int(k),
                "aic": model.aic(points),
                "bic": model.bic(points),
                "log_likelihood": model.lower_bound_,
                "n_iter": model.n_iter_,
            }
        )
    return rows


def suggest_k(
    curves: list[dict[str, Any]],
    criterion: Literal["silhouette", "bic", "aic"] = "silhouette",
) -> int:
    """Pick the best ``k`` from an elbow / IC curve."""
    if criterion == "silhouette":
        valid = [r for r in curves if r.get("silhouette") is not None]
        if not valid:
            raise ValueError("No silhouette scores available")
        best = max(valid, key=lambda r: r["silhouette"])
        return int(best["k"])
    if criterion in {"bic", "aic"}:
        best = min(curves, key=lambda r: r[criterion])
        return int(best["k"])
    raise ValueError("criterion must be 'silhouette', 'bic', or 'aic'")
