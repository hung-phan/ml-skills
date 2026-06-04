---
name: sklearn
description: Scikit-learn patterns for ML pipelines, preprocessing, model selection, ensembles, and calibration
triggers:
  - sklearn
  - scikit-learn
  - pipeline
  - gridsearch
  - cross validation
  - feature engineering
  - model selection
  - ensemble
  - calibration
  - preprocessing
  - ColumnTransformer
  - VotingClassifier
  - StackingClassifier
---

# Scikit-learn Patterns

## Why This Exists

**Problem**: Building end-to-end ML pipelines is error-prone without a consistent API — preprocessing fitted on training data can silently leak into validation/test sets, different algorithm implementations have incompatible interfaces, and there is no standard way to compose preprocessing with models for cross-validation.

**Key insight**: The universal `fit`/`predict`/`transform` estimator contract means any preprocessor or model can be snapped into a `Pipeline`, and that pipeline integrates cleanly with cross-validation and hyperparameter search, eliminating data leakage by design.

**Reach for this when**: Working on classical ML with tabular data — classification, regression, clustering, feature engineering, model selection, or calibration. Use HuggingFace/PyTorch instead for deep learning or pretrained models; use XGBoost/LightGBM directly (with sklearn wrappers) when gradient boosting is the primary model.

## 1 — Why sklearn Exists

Problem: ML requires dozens of algorithms, each with different APIs, fit/predict semantics, and preprocessing needs. Sklearn provides a **consistent estimator interface** (`fit`, `predict`, `transform`, `score`) across 100+ algorithms so you can swap models without rewriting code.

Core contract:
```python
estimator.fit(X, y)          # learn from data
estimator.predict(X)         # inference
estimator.score(X, y)        # default metric
estimator.get_params()       # introspection
estimator.set_params(**kw)   # hyperparameter setting
```

---

## 2 — Pipeline + ColumnTransformer (The Core Pattern)

**Always use pipelines.** They prevent data leakage, simplify deployment, and compose with cross-validation.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

num_cols = ["age", "income", "tenure"]
cat_cols = ["city", "plan_type"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
], remainder="drop")  # or "passthrough"

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", GradientBoostingClassifier(n_estimators=200)),
])

pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

**Key rules:**
- `remainder="drop"` is default — explicitly list all columns you want
- Use `sparse_output=False` for OneHotEncoder when downstream doesn't handle sparse
- Access steps: `pipe.named_steps["model"]` or `pipe["model"]`
- Set nested params: `pipe.set_params(model__n_estimators=500)`

---

## 3 — Custom Transformers

```python
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    """Log1p transform for skewed numeric features."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self  # stateless

    def transform(self, X):
        X = X.copy()
        cols = self.columns or X.columns
        X[cols] = np.log1p(X[cols])
        return X
```

**Rules:**
- `__init__` params must match `get_params()` keys exactly (no mutation in `__init__`)
- `fit` must return `self`
- `transform` must not modify input in-place
- For stateful transforms, compute state in `fit`, apply in `transform`

For function-based transforms (no state):
```python
from sklearn.preprocessing import FunctionTransformer

log_tf = FunctionTransformer(np.log1p, inverse_func=np.expm1)
```

---

## 4 — Model Selection

### Cross-validation
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
print(f"AUC: {scores.mean():.3f} ± {scores.std():.3f}")
```

### GridSearchCV (exhaustive, small param spaces)
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__n_estimators": [100, 200, 500],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.1],
}

search = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
```

### RandomizedSearchCV (large/continuous param spaces)
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    "model__n_estimators": randint(50, 500),
    "model__max_depth": randint(2, 10),
    "model__learning_rate": uniform(0.001, 0.3),
}

search = RandomizedSearchCV(pipe, param_dist, n_iter=50, cv=5,
                            scoring="roc_auc", n_jobs=-1, random_state=42)
search.fit(X_train, y_train)
```

| Scenario | Use |
|----------|-----|
| <20 combinations | `GridSearchCV` |
| >20 combinations or continuous params | `RandomizedSearchCV` |
| Need early stopping | `HalvingRandomSearchCV` |
| Quick baseline | `cross_val_score` with defaults |

---

## 5 — Preprocessing

| Transformer | When to Use | Gotcha |
|-------------|-------------|--------|
| `StandardScaler` | Linear models, SVMs, KNN | Sensitive to outliers |
| `RobustScaler` | Data with outliers | Uses median/IQR |
| `MinMaxScaler` | Neural nets, bounded features | Outliers compress range |
| `OneHotEncoder` | Nominal categories, <15 cardinality | Explodes dimensions |
| `OrdinalEncoder` | Tree-based models, ordinal categories | Imposes false ordering for non-ordinal |
| `TargetEncoder` | High-cardinality categoricals | Needs `cv` to avoid leakage (built-in since 1.3) |
| `KBinsDiscretizer` | Non-linear relationships for linear models | Loses information |
| `PolynomialFeatures` | Interaction terms for linear models | Explodes dimensions quickly |

```python
from sklearn.preprocessing import TargetEncoder

# High-cardinality: TargetEncoder replaces mean-encoding with built-in CV
cat_pipe = Pipeline([
    ("target_enc", TargetEncoder(smooth="auto")),
    ("scale", StandardScaler()),
])
```

---

## 6 — Ensemble Methods

| Method | When to Use | Key Param |
|--------|-------------|-----------|
| `VotingClassifier` | Combine diverse model types | `voting="soft"` (probabilities) |
| `StackingClassifier` | Learn optimal combination weights | `final_estimator` (meta-learner) |
| `BaggingClassifier` | Reduce variance of unstable models | `n_estimators`, `max_samples` |
| `AdaBoostClassifier` | Sequential error correction | `learning_rate` |
| `GradientBoostingClassifier` | Best single-model accuracy | `n_estimators`, `max_depth` |
| `HistGradientBoostingClassifier` | Large datasets (>10K rows), native NaN handling | `max_iter`, `max_depth` |

### Stacking (the power pattern)
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

stack = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=100)),
        ("gb", GradientBoostingClassifier(n_estimators=200)),
        ("svc", SVC(probability=True)),
    ],
    final_estimator=LogisticRegression(),
    cv=5,  # internal CV to generate meta-features
    passthrough=False,
)
```

### Voting
```python
from sklearn.ensemble import VotingClassifier

vote = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
    voting="soft",  # use predict_proba, not majority vote
    weights=[2, 3, 1],  # optional weighting
)
```

---

## 7 — Calibration

**Problem:** Many models output scores that aren't true probabilities (RF, SVM, GBM).

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Wrap any classifier for calibrated probabilities
cal_model = CalibratedClassifierCV(
    estimator=GradientBoostingClassifier(),
    method="isotonic",  # or "sigmoid" (Platt scaling)
    cv=5,
)
cal_model.fit(X_train, y_train)
probs = cal_model.predict_proba(X_test)[:, 1]

# Diagnostic: reliability diagram
fraction_pos, mean_predicted = calibration_curve(y_test, probs, n_bins=10)
```

| Method | When |
|--------|------|
| `"sigmoid"` (Platt) | Small datasets, roughly sigmoid-shaped distortion |
| `"isotonic"` | Larger datasets (>1000), non-parametric correction |

**When to calibrate:**
- You need actual probabilities (risk scoring, cost-sensitive decisions)
- Comparing probabilities across different models
- NOT needed if you only care about ranking (AUC is calibration-invariant)

---

## 8 — Estimator Decision Table

| Problem | Dataset Size | First Try | If Underfitting | If Overfitting |
|---------|-------------|-----------|-----------------|----------------|
| Classification, tabular | <10K rows | `LogisticRegression` | `GradientBoostingClassifier` | Regularize (`C`, `max_depth`) |
| Classification, tabular | 10K-1M | `HistGradientBoostingClassifier` | More `max_iter`, deeper trees | `min_samples_leaf`, `l2_regularization` |
| Classification, tabular | >1M | `HistGradientBoostingClassifier` | Increase `max_iter` | `max_leaf_nodes`, subsampling |
| Regression, tabular | Any | `Ridge` → `HistGradientBoostingRegressor` | Feature engineering | Regularization |
| Clustering | Known K | `KMeans` | More clusters | Fewer clusters, `MiniBatchKMeans` |
| Clustering | Unknown K | `HDBSCAN` (sklearn 1.3+) | Lower `min_cluster_size` | Higher `min_cluster_size` |
| Anomaly detection | Mostly normal data | `IsolationForest` | Lower `contamination` | Higher `contamination` |
| Dimensionality reduction | Visualization | `PCA` → `TSNE` / `UMAP` | More components | Fewer components |
| Text classification | Any | `TfidfVectorizer` + `LogisticRegression` | `SGDClassifier` | More regularization |

---

## 9 — Gotchas

1. **Data leakage**: Never fit preprocessors on full data before splitting. Use `Pipeline` + `cross_val_score` — sklearn handles this correctly.
2. **Sparse vs dense**: `OneHotEncoder` outputs sparse by default. Some estimators (e.g., `HistGradientBoosting`) don't accept sparse. Use `sparse_output=False` or `set_output(transform="pandas")`.
3. **Feature names**: Use `pipe.get_feature_names_out()` (sklearn ≥1.0) to trace transformed feature names back to originals.
4. **Reproducibility**: Always set `random_state` on estimators, splitters, and search objects.
5. **n_jobs=-1**: Parallelizes across CPU cores. Don't nest (e.g., `GridSearchCV(n_jobs=-1)` with `RandomForest(n_jobs=-1)`) — use `n_jobs` on outer loop only.
6. **Memory**: `Pipeline(memory="cache_dir")` caches expensive transform steps. Use for large datasets with slow preprocessing.
7. **pandas output**: `pipe.set_output(transform="pandas")` preserves DataFrame structure through transforms (sklearn ≥1.2).

---

## 10 — References

- Docs: https://scikit-learn.org/stable/
- Source: https://github.com/scikit-learn/scikit-learn
- User Guide: https://scikit-learn.org/stable/user_guide.html
- API Reference: https://scikit-learn.org/stable/modules/classes.html
- Pipeline Guide: https://scikit-learn.org/stable/modules/compose.html
- Model Selection: https://scikit-learn.org/stable/model_selection.html
- Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
- Ensemble Methods: https://scikit-learn.org/stable/modules/ensemble.html
- Calibration: https://scikit-learn.org/stable/modules/calibration.html
- Choosing Estimator Flowchart: https://scikit-learn.org/stable/machine_learning_map.html

## References

- Official docs: https://scikit-learn.org/stable/user_guide.html
- GitHub: https://github.com/scikit-learn/scikit-learn
- Paper: https://jmlr.org/papers/v12/pedregosa11a.html (Pedregosa et al., 2011 — original scikit-learn JMLR paper)
