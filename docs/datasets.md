# Datasets

All demo CSVs live in [`data/`](../data/) with columns `x`, `y`, and ground-truth
`cat` (three classes × 1000 points).

| File | Geometry | Hard for | Friendly to |
|------|----------|----------|-------------|
| `mv3.csv` | Separated near-spherical Gaussians | — | K-Means, Soft K-Means, GMM, Ward |
| `mv.csv` | Mildly overlapping Gaussians (one tilted) | Hard K-Means | GMM |
| `mv2.csv` | Elongated / correlated Gaussians | K-Means | GMM |
| `unif.csv` | Overlapping uniform rectangles | Most methods | Density methods (sometimes) |

## Generative recipes

The checked-in files are the canonical artifacts used by tests and README
numbers. To recreate the **same family** of geometries:

```bash
python3 scripts/generate_data.py
# writes data/*-generated.csv

python3 scripts/generate_data.py --overwrite-canonical  # replace data/*.csv
```

Approximate recipes (see `scripts/generate_data.py` for exact covariances):

- **mv / mv2 / mv3** — draws from three 2-D multivariate normals with different
  means and covariance orientations.
- **unif** — three axis-aligned uniform rectangles that deliberately overlap:
  left vertical strip, center horizontal band, right vertical strip.

## Why keep labeled data?

Ground-truth `cat` labels let us score unsupervised models with Adjusted Rand
Index and NMI — the key proof that custom implementations recover structure
comparably to scikit-learn.
