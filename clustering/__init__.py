"""From-scratch clustering algorithms with scikit-learn comparisons.

Implements K-Means, Soft K-Means, Gaussian Mixture Models (EM), DBSCAN,
and agglomerative hierarchical clustering, plus K-Means++ initialization,
model-selection helpers, and evaluation metrics.
"""

from .dbscan import DBSCAN
from .gmm import GaussianMixtureModel
from .hierarchical import AgglomerativeClustering
from .initialization import kmeans_plusplus
from .kmeans import KMeans, SoftKMeans
from .metrics import clustering_report, inertia, pairwise_squared_distances

__all__ = [
    "AgglomerativeClustering",
    "DBSCAN",
    "GaussianMixtureModel",
    "KMeans",
    "SoftKMeans",
    "kmeans_plusplus",
    "clustering_report",
    "inertia",
    "pairwise_squared_distances",
]

__version__ = "1.1.0"
