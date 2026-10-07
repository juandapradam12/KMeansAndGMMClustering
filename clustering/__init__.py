"""From-scratch clustering algorithms with scikit-learn comparisons.

Implements K-Means, Soft K-Means, and Gaussian Mixture Models (EM)
with K-Means++ initialization, model-selection helpers, and evaluation
metrics against labeled synthetic datasets.
"""

from .gmm import GaussianMixtureModel
from .initialization import kmeans_plusplus
from .kmeans import KMeans, SoftKMeans
from .metrics import clustering_report, inertia, pairwise_squared_distances

__all__ = [
    "GaussianMixtureModel",
    "KMeans",
    "SoftKMeans",
    "kmeans_plusplus",
    "clustering_report",
    "inertia",
    "pairwise_squared_distances",
]

__version__ = "1.0.0"
