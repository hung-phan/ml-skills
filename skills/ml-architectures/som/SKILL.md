---
name: SOM
description: Self-Organizing Maps for topology-preserving dimensionality reduction, clustering, and visualization. Covers competitive learning, U-matrix, minisom patterns, and comparison with t-SNE/UMAP.
---

## Why This Exists

**Problem**: You have high-dimensional data and need to visualize its cluster structure on a 2D map while preserving topological relationships — similar data points should map to nearby neurons. t-SNE/UMAP show structure but don't provide a fixed grid for ongoing classification or monitoring.

**Key insight**: Competitive learning on a fixed grid — each input activates its closest neuron (BMU), which drags its neighbors closer to the input. Over time, the grid unfolds to cover the data manifold while preserving neighborhood relationships, creating an interpretable topographic map.

**Reach for this when**: You need topology-preserving visualization with a fixed grid structure (dashboards, monitoring), want interpretable cluster boundaries via U-matrix, or need a non-parametric density estimator for anomaly detection. Prefer t-SNE/UMAP for pure visualization, SOMs for ongoing operational use with stable grid coordinates.


# Self-Organizing Maps (SOM)

Unsupervised neural network for topology-preserving dimensionality reduction, clustering, and visualization. Introduced by Teuvo Kohonen (1982).

## Algorithm

### Core Loop (Competitive Learning)

```
For each epoch:
  For each input vector x:
    1. Competition: Find Best Matching Unit (BMU)
       bmu = argmin_i ||x - w_i||  (Euclidean distance)
    2. Cooperation: Determine neighborhood of BMU
       h(bmu, j, t) = exp(-d(bmu,j)² / (2σ(t)²))
    3. Adaptation: Update weights
       w_j(t+1) = w_j(t) + α(t) · h(bmu,j,t) · (x - w_j(t))
```

### Key Parameters
- **Grid dimensions**: (rows, cols) — typically 5√N neurons for N samples
- **σ (sigma)**: Initial neighborhood radius — typically max(rows, cols) / 2
- **α (learning_rate)**: Initial learning rate — typically 0.5
- **Iterations**: At least 500 × num_neurons for convergence

### Learning Rate Decay

```python
α(t) = α₀ · exp(-t / τ)           # exponential (most common)
α(t) = α₀ / (1 + t / (T/2))       # inverse
α(t) = α₀ · (1 - t/T)             # linear
```

### Neighborhood Radius Decay

```python
σ(t) = σ₀ · exp(-t / τ)           # exponential shrink
# τ = T / log(σ₀) ensures radius → 1 by end of training
```

## Quality Metrics

### Quantization Error (QE)
Average distance between each data point and its BMU weight vector.
```
QE = (1/N) Σᵢ ||xᵢ - w_bmu(xᵢ)||
```
Lower = better data representation.

### Topographic Error (TE)
Fraction of samples where first and second BMUs are NOT adjacent on the grid.
```
TE = (1/N) Σᵢ u(xᵢ)
u(xᵢ) = 1 if BMU₁ and BMU₂ are not neighbors, else 0
```
Lower = better topology preservation. Good maps: TE < 0.05.

## minisom Python Patterns

```python
from minisom import MiniSom
import numpy as np

# --- Setup ---
data = np.array(...)  # shape (n_samples, n_features)
grid_size = int(np.ceil(np.sqrt(5 * np.sqrt(len(data)))))
som = MiniSom(grid_size, grid_size, data.shape[1],
              sigma=grid_size/2, learning_rate=0.5,
              topology='rectangular',  # or 'hexagonal'
              neighborhood_function='gaussian',
              activation_distance='euclidean')

# --- Initialize ---
som.pca_weights_init(data)

# --- Train ---
som.train(data, num_iteration=10000)       # online random
# som.train_batch(data, num_iteration=500) # batch mode

# --- Evaluate ---
qe = som.quantization_error(data)  # lower = better
te = som.topographic_error(data)   # lower = better, target < 0.05

# --- Map data ---
winner = som.winner(data[0])       # (row, col) of BMU
winners = [som.winner(x) for x in data]

# --- Visualize ---
umatrix = som.distance_map()       # U-matrix (normalized 0-1)

# --- Clustering on SOM ---
from sklearn.cluster import KMeans
weights = som.get_weights().reshape(-1, data.shape[1])
km = KMeans(n_clusters=5).fit(weights)
cluster_map = km.labels_.reshape(grid_size, grid_size)

# --- Anomaly detection ---
distances = [np.linalg.norm(x - som.get_weights()[som.winner(x)])
             for x in data]
threshold = np.percentile(distances, 95)
anomalies = data[np.array(distances) > threshold]
```

## SOM vs t-SNE vs UMAP

| Property | SOM | t-SNE | UMAP |
|----------|-----|-------|------|
| Output | Fixed grid (discrete) | Scatter plot (continuous) | Scatter plot (continuous) |
| Topology | Grid topology preserved | Local only | Local + some global |
| New data | Map directly via winner() | Requires re-fit | transform() available |
| Interpretability | High (grid cells = prototypes) | Low (distances meaningless) | Medium |
| Clustering | Native (U-matrix + k-means on weights) | Post-hoc only | Post-hoc only |
| Speed | O(N·M·D) per epoch, M=neurons | O(N² or N·log N) | O(N·log N) |
| Scalability | Good (M fixed) | Poor >10K points | Good |
| Determinism | Yes (with PCA init + batch) | No (random init) | No |
| Anomaly detection | Native (QE threshold) | Not designed for it | Not designed for it |

## When to Use

| ✅ Use SOM | ❌ Don't Use |
|---|---|
| Topology-preserving 2D visualization of high-dim data | When you need class labels (use supervised) |
| Clustering without knowing K ahead of time | Large datasets (>100K samples, too slow) |
| Exploratory data analysis, finding structure | When clusters aren't spatially meaningful |
| Sensor/IoT grid mapping | High-dimensional output needed |
| Customer segmentation visualization | When UMAP/t-SNE gives better separation |

**Typical domains**: Customer segmentation heatmaps, sensor monitoring, color quantization, document clustering, biological data exploration.

**Decision rule**: Need a 2D map preserving neighborhood topology? SOM. Just need clusters? K-means/HDBSCAN. Just need 2D projection? UMAP.

---

## References

- [MiniSom](https://github.com/JustGlowing/minisom) — Lightweight Self-Organizing Map library for Python
- [Kohonen's Self-Organizing Map (original paper, 1982)](https://ieeexplore.ieee.org/document/6726486) — Foundational SOM algorithm
- [ESOM: Tools for Visualizing High-Dimensional Data (Ultsch, 2005)](https://www.uni-marburg.de/fb12/arbeitsgruppen/datenbionik/pdf/pubs/2005/ultsch05esom) — U-matrix and emergent SOM visualization
