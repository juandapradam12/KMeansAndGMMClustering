# Usage

## Install

```bash
python3 -m pip install -r requirements.txt
```

The `clustering` package is importable from the repository root (no install
step required for local use).

## Quick start

```python
import pandas as pd
from clustering import KMeans, SoftKMeans, GaussianMixtureModel, clustering_report

df = pd.read_csv("data/mv.csv", index_col=0)
X = df[["x", "y"]].to_numpy()
y = df["cat"].to_numpy()

km = KMeans(n_clusters=3, random_state=0).fit(X)
print(clustering_report(X, y, km.labels_, km.cluster_centers_))

gmm = GaussianMixtureModel(n_components=3, random_state=0).fit(X)
print("BIC:", gmm.bic(X), "AIC:", gmm.aic(X))
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

## Benchmarks & tests

```bash
python3 scripts/benchmark.py
python3 -m pytest tests/ -v
```

## Notebook

Open `notebooks/Clustering_Methods.ipynb` for a narrative walkthrough that
loads the datasets, fits all three custom models, compares them with
scikit-learn, and plots model-selection curves.
