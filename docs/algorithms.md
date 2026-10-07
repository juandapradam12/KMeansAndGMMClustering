# Algorithms

This project implements classical clustering methods and compares them with
scikit-learn on labeled 2-D synthetic datasets.

## K-Means++

K-Means is sensitive to centroid initialization. K-Means++ reduces poor local
minima by sampling successive centers with probability proportional to the
**squared** distance to the nearest already-chosen center:

1. Pick the first center uniformly from the data.
2. For each remaining center, sample point \(x\) with

\[
P(x) \propto \min_{c \in C} \|x - c\|^2
\]

3. Repeat until \(k\) centers are chosen.

## Hard K-Means

Alternates between:

- **Assignment:** give each point to the nearest centroid (hard one-hot).
- **Update:** set each centroid to the mean of its assigned points.

Empty clusters are re-seeded on the farthest point from existing centers so
the algorithm remains stable. Convergence is declared when the maximum
centroid movement falls below a tolerance.

## Soft K-Means

Soft K-Means replaces hard assignments with exponential responsibilities:

\[
\phi_i(k) =
\frac{\exp(-\|x_i-\mu_k\|^2 / \beta)}
     {\sum_j \exp(-\|x_i-\mu_j\|^2 / \beta)}
\]

Centroids are then weighted means of the data. The temperature \(\beta\)
controls softness: smaller values approach hard K-Means; larger values blend
clusters more aggressively. Responsibilities are computed with a log-sum-exp
trick for numerical stability.

## Gaussian Mixture Models (EM)

A GMM models the density as

\[
p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)
\]

Expectation-Maximization iterates:

- **E-step:** compute posterior responsibilities
  \(\phi_i(k) \propto \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)\).
- **M-step:** update \(\pi_k\), \(\mu_k\), and \(\Sigma_k\) from weighted
  sufficient statistics.

Implementation details that improve robustness:

- K-Means++ (optionally warm-started with short K-Means) for means
- Uniform initial mixture weights and identity covariances
- Diagonal covariance regularization (`reg_covar`)
- Log-domain responsibility normalization
- AIC / BIC helpers for choosing the number of components

## DBSCAN

Density-Based Spatial Clustering of Applications with Noise:

1. Mark points with at least `min_samples` neighbors within radius `eps` as **core**.
2. Grow clusters by BFS through neighborhoods of core points.
3. Label non-core, non-reachable points as **noise** (`-1`).

Neighbor queries use a KD-tree. DBSCAN does not require choosing \(k\) up front,
but `eps` / `min_samples` are sensitive to scale.

## Agglomerative hierarchical clustering

Bottom-up merging of clusters using SciPy linkage (`ward`, `average`,
`complete`, or `single`), then cutting the dendrogram into `n_clusters` flat
groups. Ward linkage is a strong baseline on compact spherical blobs.

## When to prefer which model

| Method | Strengths | Limitations |
|--------|-----------|-------------|
| K-Means | Fast, simple, strong spherical clusters | Hard boundaries; sensitive to scale |
| Soft K-Means | Graded memberships; tunable softness | Still isotropic distance geometry |
| GMM | Elliptical clusters; probabilistic density | More parameters; needs regularization |
| DBSCAN | Arbitrary shapes; explicit noise label | Scale-sensitive hyperparameters |
| Agglomerative | No random init; dendrogram view | \(O(n^2)\) memory/time; linkage choice matters |

## Evaluation

Because the included CSVs carry ground-truth `cat` labels, models are scored with:

- **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)**
- **Inertia** (K-Means) and **silhouette**
- **AIC / BIC** (GMM model selection)
- Side-by-side comparison against scikit-learn estimators
