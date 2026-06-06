---
name: feature-selection
description: Feature selection methods — filter (variance, mutual information, chi-square), wrapper (RFE, Boruta, SFS), embedded (Lasso, tree importance, SHAP), and VIF/stability selection. Use when selecting informative features, removing noise, detecting multicollinearity, or reducing dimensionality before training.
---

# Feature Selection

## Why This Exists

Raw datasets often contain hundreds or thousands of features where many are noise, redundant, or collinear. Training on all of them leads to overfitting, slow training, poor generalization, and uninterpretable models. Feature selection solves this by identifying the minimal subset that carries predictive signal — reducing dimensionality, speeding up training, improving accuracy, and making models explainable. Without it, you're fitting noise.

---

Systematic approaches to identify the most informative features, remove redundant/noisy variables, and build parsimonious models that generalize better.

---

## 1. Filter Methods

Univariate statistical tests that score features independently of any model.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
    chi2,
    f_classif,
    f_regression,
    SelectKBest,
    SelectPercentile,
)

# --- Variance Threshold (remove near-constant features) ---
X, y = make_classification(n_samples=1000, n_features=50, n_informative=10, random_state=42)

selector = VarianceThreshold(threshold=0.01)  # remove features with var < 0.01
X_filtered = selector.fit_transform(X)
kept_mask = selector.get_support()
print(f"Kept {kept_mask.sum()}/{X.shape[1]} features")

# --- Mutual Information (works for non-linear relationships) ---
# Classification
mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
mi_ranking = np.argsort(mi_scores)[::-1]
print("Top 10 MI features:", mi_ranking[:10])

# Regression
X_reg, y_reg = make_regression(n_samples=1000, n_features=50, n_informative=10, random_state=42)
mi_reg_scores = mutual_info_regression(X_reg, y_reg, random_state=42)

# Using SelectKBest wrapper
selector_mi = SelectKBest(score_func=mutual_info_classif, k=15)
X_mi = selector_mi.fit_transform(X, y)

# --- Chi-squared (non-negative features only, e.g. counts/TF-IDF) ---
X_positive = np.abs(X)  # chi2 requires non-negative
selector_chi2 = SelectKBest(score_func=chi2, k=10)
X_chi2 = selector_chi2.fit_transform(X_positive, y)
print("Chi2 scores:", selector_chi2.scores_[:5])

# --- F-test (ANOVA for classification, F-regression for regression) ---
selector_f = SelectKBest(score_func=f_classif, k=10)
X_f = selector_f.fit_transform(X, y)

selector_f_reg = SelectPercentile(score_func=f_regression, percentile=30)
X_f_reg = selector_f_reg.fit_transform(X_reg, y_reg)

# --- Correlation-based removal (drop highly correlated pairs) ---
def remove_correlated_features(df: pd.DataFrame, threshold: float = 0.90) -> list:
    """Remove features with pairwise correlation above threshold."""
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    return to_drop

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
drop_cols = remove_correlated_features(df, threshold=0.90)
df_filtered = df.drop(columns=drop_cols)
print(f"Dropped {len(drop_cols)} correlated features")
```

---

## 2. Wrapper Methods

Use a model to evaluate feature subsets iteratively.

```python
from sklearn.feature_selection import RFE, RFECV, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

X, y = make_classification(n_samples=500, n_features=30, n_informative=8, random_state=42)

# --- Recursive Feature Elimination (RFE) ---
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=estimator, n_features_to_select=10, step=1)
rfe.fit(X, y)
print("RFE selected features:", np.where(rfe.support_)[0])
print("Feature rankings:", rfe.ranking_)

# --- RFECV (RFE with cross-validation to find optimal k) ---
rfecv = RFECV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    step=1,
    cv=StratifiedKFold(5),
    scoring="accuracy",
    min_features_to_select=3,
    n_jobs=-1,
)
rfecv.fit(X, y)
print(f"Optimal features: {rfecv.n_features_}")
print(f"Selected: {np.where(rfecv.support_)[0]}")

# --- Sequential Feature Selector (forward) ---
sfs_forward = SequentialFeatureSelector(
    estimator=GradientBoostingClassifier(n_estimators=50, random_state=42),
    n_features_to_select=10,
    direction="forward",
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
)
sfs_forward.fit(X, y)
print("Forward SFS features:", np.where(sfs_forward.get_support())[0])

# --- Sequential Feature Selector (backward) ---
sfs_backward = SequentialFeatureSelector(
    estimator=GradientBoostingClassifier(n_estimators=50, random_state=42),
    n_features_to_select=10,
    direction="backward",
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
)
sfs_backward.fit(X, y)
print("Backward SFS features:", np.where(sfs_backward.get_support())[0])

# --- Boruta (all-relevant feature selection) ---
from boruta import BorutaPy

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
boruta = BorutaPy(
    estimator=rf,
    n_estimators="auto",
    max_iter=100,
    random_state=42,
    verbose=0,
)
boruta.fit(X, y)
print(f"Boruta confirmed: {np.where(boruta.support_)[0]}")
print(f"Boruta tentative: {np.where(boruta.support_weak_)[0]}")
print(f"Feature ranking: {boruta.ranking_}")
```

---

## 3. Embedded Methods

Feature importance derived during model training.

```python
import shap
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=40, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Lasso / L1 regularization (drives coefficients to zero) ---
# Regression
X_reg, y_reg = make_regression(n_samples=1000, n_features=40, n_informative=10, random_state=42)
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_reg, y_reg)
print(f"Optimal alpha: {lasso_cv.alpha_:.4f}")
print(f"Non-zero coefficients: {np.sum(lasso_cv.coef_ != 0)}/{len(lasso_cv.coef_)}")
selected_lasso = np.where(lasso_cv.coef_ != 0)[0]

# Classification with L1
lr_l1 = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=2000, random_state=42)
lr_l1.fit(X_train, y_train)
selected_l1 = np.where(lr_l1.coef_[0] != 0)[0]
print(f"L1 LogReg selected: {len(selected_l1)} features")

# --- SelectFromModel (threshold-based) ---
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

sfm = SelectFromModel(rf, threshold="median")  # or "mean", "1.25*mean", specific float
sfm.fit(X_train, y_train)
X_sfm = sfm.transform(X_train)
print(f"SelectFromModel kept: {X_sfm.shape[1]} features")
print(f"Importances: {rf.feature_importances_[sfm.get_support()]}")

# --- Permutation Importance (model-agnostic, computed on held-out data) ---
perm_imp = permutation_importance(
    rf, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1
)
perm_ranking = np.argsort(perm_imp.importances_mean)[::-1]
print("Top 10 by permutation importance:", perm_ranking[:10])

# Filter features with positive importance (mean > 0)
significant_features = np.where(perm_imp.importances_mean > 0)[0]

# --- SHAP feature importance (global) ---
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train, y_train)

explainer = shap.TreeExplainer(gbr)
shap_values = explainer.shap_values(X_test)

# Mean absolute SHAP values as importance
shap_importance = np.abs(shap_values).mean(axis=0)
shap_ranking = np.argsort(shap_importance)[::-1]
print("Top 10 SHAP features:", shap_ranking[:10])

# Select features above a threshold
shap_threshold = np.percentile(shap_importance, 50)
shap_selected = np.where(shap_importance > shap_threshold)[0]
print(f"SHAP selected: {len(shap_selected)} features")

# Visualization
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
```

---

## 4. Stability Selection

Bootstrap subsampling to identify features that are consistently selected across perturbations.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

def stability_selection(X, y, estimator_fn, n_bootstrap=100, threshold=0.6, sample_fraction=0.5):
    """
    Run stability selection: fit L1 model on bootstrap subsamples,
    track how often each feature is selected.
    
    Args:
        X: feature matrix
        y: target
        estimator_fn: callable returning a fitted estimator with coef_ attribute
        n_bootstrap: number of bootstrap iterations
        threshold: selection frequency threshold (0.6-0.9 recommended)
        sample_fraction: fraction of data per subsample
    
    Returns:
        stable_features: indices of stable features
        selection_frequencies: per-feature selection probability
    """
    n_samples, n_features = X.shape
    subsample_size = int(n_samples * sample_fraction)
    selection_counts = np.zeros(n_features)
    
    for i in range(n_bootstrap):
        # Subsample without replacement
        indices = np.random.choice(n_samples, size=subsample_size, replace=False)
        X_sub, y_sub = X[indices], y[indices]
        
        # Fit L1-regularized model
        model = estimator_fn()
        model.fit(X_sub, y_sub)
        
        # Count non-zero coefficients
        coef = model.coef_.ravel() if hasattr(model.coef_, 'ravel') else model.coef_
        selection_counts += (coef != 0).astype(int)
    
    selection_frequencies = selection_counts / n_bootstrap
    stable_features = np.where(selection_frequencies >= threshold)[0]
    
    return stable_features, selection_frequencies

# Usage
X, y = make_classification(n_samples=500, n_features=50, n_informative=8, random_state=42)

stable_feats, freqs = stability_selection(
    X, y,
    estimator_fn=lambda: LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=1000, random_state=None),
    n_bootstrap=200,
    threshold=0.7,
    sample_fraction=0.5,
)
print(f"Stable features (freq >= 0.7): {stable_feats}")
print(f"Their frequencies: {freqs[stable_feats]}")

# Using stability-selection library
from stability_selection import StabilitySelection

selector = StabilitySelection(
    base_estimator=LogisticRegression(penalty="l1", solver="saga", max_iter=1000),
    lambda_name="C",
    lambda_grid=np.logspace(-2, 0, 20),
    threshold=0.7,
    n_bootstrap_iterations=100,
    random_state=42,
)
selector.fit(X, y)
stable_features_lib = selector.get_support(indices=True)
print(f"stability-selection library selected: {stable_features_lib}")
```

---

## 5. Multicollinearity Detection (VIF)

Variance Inflation Factor measures how much a feature's variance is inflated due to collinearity with other features.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Calculate VIF for all features. VIF > 5-10 indicates multicollinearity."""
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    return vif_data.sort_values("VIF", ascending=False)

def iterative_vif_removal(X: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """Iteratively remove features with VIF above threshold until all pass."""
    X_current = X.copy()
    dropped = []
    
    while True:
        vif_df = calculate_vif(X_current)
        max_vif = vif_df["VIF"].max()
        
        if max_vif <= threshold:
            break
        
        worst_feature = vif_df.loc[vif_df["VIF"].idxmax(), "feature"]
        dropped.append((worst_feature, max_vif))
        X_current = X_current.drop(columns=[worst_feature])
        print(f"Dropped '{worst_feature}' (VIF={max_vif:.1f})")
    
    print(f"\nRemoved {len(dropped)} features. Remaining: {X_current.shape[1]}")
    return X_current

# Usage
X, _ = make_regression(n_samples=500, n_features=20, n_informative=10, random_state=42)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

# Add collinear features
df["f20"] = df["f0"] * 0.95 + np.random.normal(0, 0.1, 500)
df["f21"] = df["f1"] + df["f2"] + np.random.normal(0, 0.05, 500)

print("Before VIF removal:")
print(calculate_vif(df).head(10))

df_clean = iterative_vif_removal(df, threshold=10.0)
print("\nAfter VIF removal:")
print(calculate_vif(df_clean).head(10))
```

---

## 6. Dimensionality Reduction for Feature Selection

Use component loadings to identify which original features contribute most.

```python
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

X, y = make_classification(n_samples=1000, n_features=50, n_informative=10, n_classes=3,
                           n_clusters_per_class=1, random_state=42)
feature_names = [f"f{i}" for i in range(X.shape[1])]

# --- PCA with loadings analysis ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)  # retain 95% variance
X_pca = pca.fit_transform(X_scaled)
print(f"PCA: {X.shape[1]} -> {X_pca.shape[1]} components (95% variance)")

# Feature importance from loadings (absolute contribution to top components)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=feature_names,
)

# Weight loadings by explained variance ratio
weighted_importance = np.abs(loadings).multiply(pca.explained_variance_ratio_, axis=1).sum(axis=1)
top_pca_features = weighted_importance.nlargest(15).index.tolist()
print(f"Top PCA features: {top_pca_features}")

# --- Truncated SVD (for sparse data, e.g. TF-IDF) ---
from scipy.sparse import random as sparse_random

X_sparse = sparse_random(1000, 200, density=0.1, format="csr", random_state=42)
svd = TruncatedSVD(n_components=20, random_state=42)
X_svd = svd.fit_transform(X_sparse)
print(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}")

svd_importance = np.abs(svd.components_).sum(axis=0)
top_svd_features = np.argsort(svd_importance)[::-1][:20]

# --- LDA (supervised, maximizes class separation) ---
lda = LDA(n_components=2)  # max = n_classes - 1
X_lda = lda.fit_transform(X_scaled, y)

# LDA coefficients indicate feature importance for class separation
lda_importance = np.abs(lda.coef_).mean(axis=0)
top_lda_features = np.argsort(lda_importance)[::-1][:15]
print(f"Top LDA features: {top_lda_features}")
```

---

## 7. Complete Pipeline: Filter → Embedded → Final Model

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, mutual_info_classif, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import make_classification
import numpy as np

# Generate data with noise
X, y = make_classification(
    n_samples=2000, n_features=100, n_informative=15,
    n_redundant=10, n_clusters_per_class=2, random_state=42
)

# --- Multi-stage selection pipeline ---
pipeline = Pipeline([
    # Stage 1: Remove constant/near-constant features
    ("variance", VarianceThreshold(threshold=0.01)),
    
    # Stage 2: Scale for MI computation
    ("scaler", StandardScaler()),
    
    # Stage 3: Filter with mutual information (keep top 50%)
    ("mi_filter", SelectKBest(score_func=mutual_info_classif, k=50)),
    
    # Stage 4: Embedded selection with tree importance
    ("embedded", SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        threshold="mean"
    )),
    
    # Stage 5: Final classifier
    ("classifier", GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
    )),
])

# Evaluate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"Pipeline accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# Compare with no selection
baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
    )),
])
baseline_scores = cross_val_score(baseline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"No selection accuracy: {baseline_scores.mean():.4f} ± {baseline_scores.std():.4f}")

# --- Inspect what was selected ---
pipeline.fit(X, y)
after_variance = pipeline.named_steps["variance"].get_support()
after_mi = pipeline.named_steps["mi_filter"].get_support()
after_embedded = pipeline.named_steps["embedded"].get_support()

# Trace back to original indices
idx_after_var = np.where(after_variance)[0]
idx_after_mi = idx_after_var[after_mi]
idx_final = idx_after_mi[after_embedded]
print(f"\nSelection funnel: {X.shape[1]} -> {after_variance.sum()} -> {after_mi.sum()} -> {len(idx_final)}")
print(f"Final selected feature indices: {idx_final}")
```

---

## When to Use

| Method | Best For | Scales To | Captures Non-linear |
|--------|----------|-----------|-------------------|
| Variance Threshold | Removing constants/near-constants | Very large (O(n·p)) | No |
| Mutual Information | Non-linear univariate relationships | Large | Yes |
| Chi-squared | Categorical/count features | Large | No |
| F-test (ANOVA/regression) | Linear univariate relationships | Very large | No |
| Correlation removal | Removing redundant features | Large (O(p²)) | No |
| RFE / RFECV | Finding optimal subset with any model | Medium (retrains p times) | Depends on model |
| Sequential (SFS) | Small-medium feature sets | Small-medium | Depends on model |
| Boruta | All-relevant selection (not minimal) | Medium | Yes (tree-based) |
| Lasso / L1 | Sparse linear models | Large | No |
| Tree importance | Quick embedded baseline | Large | Yes |
| Permutation importance | Model-agnostic post-hoc | Medium (refit-free) | Depends on model |
| SHAP importance | Interpretable global importance | Medium (slow for large N) | Yes |
| Stability selection | Robust selection under perturbation | Medium | Depends on base |
| VIF | Detecting multicollinearity | Medium (O(p²) fits) | No |
| PCA loadings | Identifying variance drivers | Large | No (linear) |
| LDA | Supervised class-separation features | Large | No (linear) |

---

## Common Gotchas

1. **Data leakage**: Always fit selectors inside cross-validation (use `Pipeline`). Never select features on full data then split.
2. **Scaling before selection**: MI, chi2, and tree methods don't need scaling. L1/VIF/PCA do. Put `StandardScaler` in the right pipeline position.
3. **Chi-squared requires non-negative inputs**: Apply to raw counts or TF-IDF, not standardized data.
4. **Mutual information is stochastic**: Set `random_state` and use multiple runs or `SelectKBest` with enough `k` to be robust.
5. **RFE with unstable estimators**: Tree-based RFE rankings can vary between runs. Use RFECV or stability selection for robustness.
6. **Boruta is slow**: O(n_iterations × model_fit). Use `max_iter=100` and subsample large datasets.
7. **VIF with dummy variables**: One-hot encoded categoricals inflate VIF. Drop one level per category before computing VIF.
8. **Permutation importance on training data**: Overestimates importance of overfit features. Always compute on held-out test set.
9. **SHAP ≠ causal**: High SHAP importance means predictive association, not causal effect. Correlated features split importance.
10. **Correlation threshold too aggressive**: Dropping at r=0.7 can remove genuinely informative features. Start at 0.90-0.95 and validate with model performance.
11. **Lasso path instability**: With correlated features, Lasso arbitrarily picks one. Use Elastic Net (l1_ratio=0.5-0.9) or stability selection.
12. **PCA for selection loses interpretability**: PCA components are linear combos. Use loadings to trace back, but the selected "features" aren't original columns.

---

## References

- sklearn feature_selection: https://scikit-learn.org/stable/modules/feature_selection.html
- Boruta: https://github.com/scikit-learn-contrib/boruta_py
- SHAP: https://shap.readthedocs.io/
- stability-selection: https://github.com/scikit-learn-contrib/stability-selection
- statsmodels VIF: https://www.statsmodels.org/stable/stats.html#variance-inflation-factor
- Guyon & Elisseeff (2003) "An Introduction to Variable and Feature Selection": https://jmlr.org/papers/v3/guyon03a.html
