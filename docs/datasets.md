# Datasets in the case

CSVs in [`data/`](../data/): `x`, `y`, and ground-truth `cat` (3 × 1000 points). Labels exist so unsupervised partitions can be scored with ARI / NMI against scikit-learn.

<p align="center">
  <img src="../assets/cluster_categories.png" alt="Four synthetic datasets with ground-truth labels" width="640"/>
</p>

**Figure — Panel layout.** Each subplot is one CSV; color encodes `cat`. Reading left-to-right, top-to-bottom: `mv`, `unif`, `mv2`, `mv3`.

| File | Role |
|------|------|
| `mv3.csv` | Control — separated near-spherical Gaussians; all methods succeed |
| `mv.csv` | Mild overlap + tilt — hard K-Means slips; GMM recovers |
| `mv2.csv` | Elongated / correlated Gaussians — largest K-Means ↔ GMM gap |
| `unif.csv` | Overlapping rectangles — stress test; absolute ARI stays low |

<p align="center">
  <img src="../assets/mv2_kmeans_vs_gmm.png" alt="mv2 ground truth vs K-Means vs GMM" width="780"/>
</p>

**Figure — Why `mv2` is the headline dataset.** Truth (left) is three diagonal ellipses. K-Means (middle) imposes spherical Voronoi cells. GMM (right) recovers the axes of correlation.

**Generation (same family as the checked-in files):** `mv*` = three 2-D Gaussians with distinct means/covariances; `unif` = three overlapping axis-aligned rectangles. Draws: `scripts/generate_data.py`.
