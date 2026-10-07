# Usage

## Install

```bash
python3 -m pip install -e ".[dev]"
```

This installs the `clustering` package and the `cluster-bench` CLI.

For a minimal runtime-only install:

```bash
python3 -m pip install -e .
```

## Quick start

```python
import pandas as pd
from clustering import (
    KMeans,
    SoftKMeans,
    GaussianMixtureModel,
    DBSCAN,
    AgglomerativeClustering,
    clustering_report,
)

df = pd.read_csv("data/mv3.csv", index_col=0)
X = df[["x", "y"]].to_numpy()
y = df["cat"].to_numpy()

km = KMeans(n_clusters=3, random_state=0).fit(X)
print(clustering_report(X, y, km.labels_, km.cluster_centers_))

gmm = GaussianMixtureModel(n_components=3, random_state=0).fit(X)
print("BIC:", gmm.bic(X), "AIC:", gmm.aic(X))

db = DBSCAN(eps=0.35, min_samples=12).fit(X)
agg = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(X)
print(db.n_clusters_, agg.n_clusters_)
```

## Choosing \(k\)

```python
from clustering.model_selection import elbow_curve, gmm_information_criteria, suggest_k

elbow = elbow_curve(X, range(2, 8), random_state=0)
print("Suggested k (silhouette):", suggest_k(elbow, "silhouette"))

ics = gmm_information_criteria(X, range(1, 8), random_state=0)
print("Suggested k (BIC):", suggest_k(ics, "bic"))
```

## Visualizations

```python
from clustering.viz import plot_datasets, plot_kmeans_comparison, plot_soft_assignments

dfs = [pd.read_csv(f"data/{name}.csv", index_col=0) for name in ("mv", "unif", "mv2", "mv3")]
plot_datasets(dfs, titles=["mv", "unif", "mv2", "mv3"])
plot_kmeans_comparison(dfs, n_clusters=3)
plot_soft_assignments(dfs, model="gmm", n_clusters=3)
```

## CLI, tests, and lint

```bash
cluster-bench           # full table
cluster-bench --quick   # mv3 only (CI smoke)

pytest -v
ruff check clustering tests scripts
mypy clustering
```

## Datasets

See [datasets.md](datasets.md). Regenerate similar geometries with:

```bash
python3 scripts/generate_data.py
```

## Notebook

Open `notebooks/Clustering_Methods.ipynb` for a narrative walkthrough with
plots, soft assignments, and model-selection curves.
