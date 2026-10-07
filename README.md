# K-Means & GMM Clustering from Scratch

**Learn clustering by building it** — then prove your implementation against scikit-learn on labeled synthetic data.

This repository is a clean, educational Python project that re-implements:

- **K-Means++** initialization
- **Hard K-Means**
- **Soft (fuzzy) K-Means**
- **Gaussian Mixture Models** with Expectation-Maximization

…and benchmarks them against `sklearn.cluster.KMeans` and `sklearn.mixture.GaussianMixture` using Adjusted Rand Index, silhouette, inertia, AIC, and BIC.

<p align="center">
  <img src="assets/Cluster%20Categories.png" alt="Four synthetic clustering datasets" width="520"/>
</p>

## Why this project stands out

Most clustering tutorials stop at calling `fit()`. This one goes further:

1. **Algorithms you can read** — vectorized NumPy implementations with clear math in [`docs/algorithms.md`](docs/algorithms.md).
2. **Fair sklearn comparisons** — same initializations where it matters; side-by-side centers and labels.
3. **The right model for the geometry** — spherical blobs favor K-Means; elongated / correlated clusters need GMMs. The included datasets make that trade-off obvious.
4. **Model selection tools** — elbow / silhouette curves for K-Means and AIC/BIC for GMMs.
5. **Tests that lock correctness** — `pytest` checks recovery on well-separated data and agreement with scikit-learn.

## Results at a glance

| Dataset | Geometry | Custom K-Means ARI | Custom GMM ARI | Takeaway |
|---------|----------|--------------------|----------------|----------|
| `mv3` | Separated spherical blobs | **0.98** | **0.99** | Both methods shine |
| `mv2` | Anisotropic, correlated | ~0.42 | **0.85** | Covariance modeling wins |
| `mv` | Mild overlap + tilt | ~0.62 | **0.84** | Soft elliptical boundaries help |
| `unif` | Overlapping uniforms | ~0.25 | ~0.57 | Hard problem; GMM still better |

Custom K-Means tracks scikit-learn inertia and labels closely when started from the same seeds.

<p align="center">
  <img src="assets/KMeans:%20Custom%20vs%20Sklearn.png" alt="Custom vs sklearn K-Means centers" width="520"/>
</p>

## Project layout

```text
clustering/                 # Importable package (algorithms + viz + metrics)
data/                       # Synthetic CSVs with ground-truth labels
notebooks/                  # Narrative walkthrough
docs/                       # Algorithm notes & usage guide
scripts/benchmark.py        # Cross-dataset ARI / inertia table
tests/                      # pytest suite
assets/                     # Figure outputs
```

## Quick start

```bash
python3 -m pip install -r requirements.txt

# Sanity checks
python3 -m pytest tests/ -v

# End-to-end comparison table
python3 scripts/benchmark.py
```

```python
import pandas as pd
from clustering import KMeans, GaussianMixtureModel, clustering_report

df = pd.read_csv("data/mv3.csv", index_col=0)
X, y = df[["x", "y"]].to_numpy(), df["cat"].to_numpy()

km = KMeans(n_clusters=3, random_state=0).fit(X)
print(clustering_report(X, y, km.labels_, km.cluster_centers_))

gmm = GaussianMixtureModel(n_components=3, random_state=0).fit(X)
print({"ari_ready_labels": gmm.labels_[:5], "bic": gmm.bic(X)})
```

Open [`notebooks/Clustering_Methods.ipynb`](notebooks/Clustering_Methods.ipynb) for plots, soft assignments, and model-selection curves.

<p align="center">
  <img src="assets/Sklearn:%20GMM%20%26%20KMeans.png" alt="Sklearn GMM and KMeans soft/hard coloring" width="520"/>
</p>

## What improved vs. the original notebook

The original exploratory notebook was preserved in spirit and upgraded into a maintainable project:

| Area | Improvement |
|------|-------------|
| Structure | Package + tests + docs instead of a single notebook |
| K-Means++ | Uses **squared** distances (standard algorithm) |
| Soft K-Means | Uses **squared** Euclidean distances + log-sum-exp stability |
| GMM | Vectorized M-step, covariance regularization, AIC/BIC |
| K-Means | Empty-cluster re-seeding; sklearn-compatible API (`fit` / `predict`) |
| Evaluation | ARI, NMI, silhouette, inertia, information criteria |
| Performance | Broadcasting / `einsum` instead of nested `apply_along_axis` loops |

## Documentation

- [Algorithm reference](docs/algorithms.md) — math, design choices, when to use each model
- [Usage guide](docs/usage.md) — install, API snippets, visualization helpers

## Requirements

Python 3.10+ recommended. See [`requirements.txt`](requirements.txt) for NumPy, SciPy, pandas, scikit-learn, Matplotlib, Jupyter, and pytest.

## License / authorship

Educational clustering project by **Juan David Prada Malagón**.
Use freely for learning, portfolios, and experimentation.
