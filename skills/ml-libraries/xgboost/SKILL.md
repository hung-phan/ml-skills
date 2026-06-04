---
name: gradient-boosting
description: XGBoost, LightGBM, and CatBoost for tabular data — hyperparameter tuning, feature importance, GPU training
triggers:
  - xgboost
  - lightgbm
  - catboost
  - gradient boosting
  - tabular data
  - gbm
  - gbdt
  - boosted trees
version: 1.0.0
---

# Gradient Boosting (XGBoost / LightGBM / CatBoost)

Ensemble of sequentially-trained weak learners (decision trees) where each tree corrects residual errors of the previous. Dominates tabular/structured data where deep learning typically underperforms due to lack of spatial/temporal structure.

## Why This Exists

**Problem**: Deep learning underperforms on structured/tabular data with limited samples and mixed feature types — it requires spatial or sequential inductive biases that don't exist in row-column data, and it cannot natively handle missing values, mixed numeric/categorical features, or small datasets without extensive preprocessing.

**Key insight**: Gradient boosting fits each new tree to the residual errors of the ensemble so far, progressively correcting bias while controlling variance through regularization — this makes it the de facto standard for tabular ML across Kaggle, fraud detection, ranking, and credit scoring.

**Reach for this when**: Your data is tabular/structured with numeric and categorical columns, especially with <1M rows or significant missing values. Choose XGBoost as the safe default, LightGBM when speed or dataset size (>100K rows) matters, and CatBoost when you have many high-cardinality categoricals and want to skip encoding. Reach for deep learning (TabNet, etc.) only when you have very large datasets (>10M rows) or known feature interactions that trees miss.

---

## 1 · Why Gradient Boosting Exists

- **Problem**: Tabular data lacks the spatial/sequential patterns that CNNs/RNNs exploit. Single decision trees overfit; random forests average but don't correct errors.
- **Solution**: Additive training — each new tree fits the negative gradient (residuals) of the loss function, progressively reducing bias while controlling variance via regularization.
- **Result**: State-of-the-art on Kaggle competitions, fraud detection, ranking, recommendation, credit scoring — any structured/tabular domain.

---

## 2 · When to Use Each

| Criterion | XGBoost | LightGBM | CatBoost |
|-----------|---------|----------|----------|
| **Best for** | General default, wide compatibility | Large datasets (>100K rows), speed | Categorical-heavy data, minimal preprocessing |
| **Tree growth** | Level-wise (balanced) | Leaf-wise (faster, deeper) | Symmetric (oblivious) trees |
| **Categoricals** | Requires encoding | Supports native (`categorical_feature`) | Native ordered target statistics — no encoding needed |
| **Speed** | Moderate | Fastest (histogram + GOSS + EFB) | Moderate (ordered boosting overhead) |
| **Overfitting risk** | Medium | Higher (leaf-wise) — tune `num_leaves` | Lowest (ordered boosting, built-in regularization) |
| **GPU support** | Yes (`tree_method='gpu_hist'`) | Yes (`device='gpu'`) | Yes (`task_type='GPU'`) |
| **Missing values** | Native handling | Native handling | Native handling |
| **Ranking** | LambdaMART built-in | LambdaMART built-in | YetiRank built-in |

**Decision rule**:
- Default / don't know → XGBoost
- >1M rows or need speed → LightGBM
- Many categorical features or want zero preprocessing → CatBoost

---

## 3 · Code Examples with Optuna Tuning

### 3.1 XGBoost + Optuna

```python
import xgboost as xgb
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_openml

X, y = fetch_openml("adult", version=2, return_X_y=True, as_frame=True)
X = X.select_dtypes(include="number").fillna(0)
y = (y == ">50K").astype(int)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "tree_method": "hist",
        "eval_metric": "logloss",
        "random_state": 42,
    }
    model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
    scores = cross_val_score(model, X, y, cv=3, scoring="roc_auc")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### 3.2 LightGBM + Optuna

```python
import lightgbm as lgb
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "verbose": -1,
        "random_state": 42,
    }
    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X, y, cv=3, scoring="roc_auc")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

### 3.3 CatBoost + Optuna (Native Categoricals)

```python
import catboost as cb
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_openml

X, y = fetch_openml("adult", version=2, return_X_y=True, as_frame=True)
y = (y == ">50K").astype(int)
cat_features = X.select_dtypes(include="category").columns.tolist()

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "cat_features": cat_features,
        "verbose": 0,
        "random_seed": 42,
    }
    model = cb.CatBoostClassifier(**params)
    scores = cross_val_score(model, X, y, cv=3, scoring="roc_auc")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

---

## 4 · Feature Importance

### 4.1 Built-in Importance

```python
# XGBoost
model.feature_importances_  # gain-based (default)
xgb.plot_importance(model, importance_type="weight")  # split count

# LightGBM
lgb.plot_importance(model, importance_type="gain")

# CatBoost
model.get_feature_importance(prettified=True)
```

**Types**: `weight` (split count), `gain` (avg loss reduction), `cover` (avg samples affected).

### 4.2 SHAP (Model-Agnostic, Additive)

```python
import shap

# TreeExplainer — exact Shapley values for tree models (fast)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot — global feature importance + direction
shap.summary_plot(shap_values, X_test)

# Dependence plot — single feature interaction
shap.dependence_plot("age", shap_values, X_test)

# Single prediction explanation
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

**Why SHAP over built-in**: Built-in importance tells you which features split often but not direction or magnitude of effect. SHAP gives signed, additive, per-sample attributions grounded in game theory.

---

## 5 · Early Stopping

Prevents overfitting by monitoring validation metric and stopping when no improvement for N rounds.

```python
# XGBoost
model = xgb.XGBClassifier(n_estimators=10000, early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f"Best iteration: {model.best_iteration}")

# LightGBM
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]
model = lgb.LGBMClassifier(n_estimators=10000)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)

# CatBoost
model = cb.CatBoostClassifier(iterations=10000, early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)
```

**Rule of thumb**: Set `n_estimators` high (5000–10000), rely on early stopping to find optimal count. Use 10–20% of data as eval set.

---

## 6 · GPU Training

```python
# XGBoost — GPU histogram
model = xgb.XGBClassifier(tree_method="gpu_hist", gpu_id=0)

# LightGBM — GPU
model = lgb.LGBMClassifier(device="gpu", gpu_use_dp=False)

# CatBoost — GPU (easiest, no extra install)
model = cb.CatBoostClassifier(task_type="GPU", devices="0")
```

**GPU speedup guide**:
| Dataset size | CPU vs GPU |
|---|---|
| <50K rows | CPU faster (GPU overhead dominates) |
| 50K–500K | GPU ~2–5× faster |
| >500K | GPU ~5–20× faster |

**CatBoost GPU** works out of the box with pip install. XGBoost/LightGBM GPU requires building from source or specific pip wheels with CUDA support.

---

## 7 · Comparison Table

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Training speed (1M rows) | ~120s | ~45s | ~90s |
| Memory efficiency | High | Highest (histogram) | Moderate |
| Categorical handling | Manual encode | Native (limited) | Best (ordered TS) |
| Default regularization | L1 + L2 | L1 + L2 | L2 + ordered boosting |
| Distributed training | Dask, Spark, Ray | Dask, Spark, Ray | Spark (limited) |
| Monotone constraints | ✅ | ✅ | ✅ |
| Custom loss functions | ✅ | ✅ | ✅ |
| Interaction constraints | ✅ | ✅ | ❌ |
| Built-in cross-validation | `xgb.cv()` | `lgb.cv()` | `cv()` method |
| Prediction speed | Fast | Fastest | Moderate |
| Model serialization | JSON, UBJSON, pickle | txt, pickle | cbm, JSON, ONNX |

---

## 8 · Gotchas

- **LightGBM `num_leaves`**: Controls complexity more than `max_depth`. Default 31 is often too high for small datasets → overfitting. Start with `num_leaves = 2^(max_depth) - 1`.
- **CatBoost slow first run**: Ordered boosting creates permutations of training data on first fit. Subsequent fits with same data are cached.
- **XGBoost `scale_pos_weight`**: For imbalanced classification, set to `sum(negative) / sum(positive)`. LightGBM uses `is_unbalance=True`. CatBoost uses `auto_class_weights='Balanced'`.
- **Feature names with special chars**: LightGBM crashes on `[`, `]`, `{`, `}` in column names. Sanitize before training.
- **Optuna + early stopping**: When using both, don't put `n_estimators` in search space. Fix it high and let early stopping decide.
- **SHAP + CatBoost**: Use `shap.TreeExplainer(model)` directly — CatBoost has native SHAP support, no need for slow `KernelExplainer`.
- **Categorical cardinality**: CatBoost handles high cardinality well. LightGBM's native categoricals degrade above ~1000 unique values — use target encoding instead.

---

## 9 · References

- XGBoost docs: https://xgboost.readthedocs.io
- LightGBM docs: https://lightgbm.readthedocs.io
- CatBoost docs: https://catboost.ai/docs/
- SHAP: https://shap.readthedocs.io
- Optuna: https://optuna.readthedocs.io
- "Why do tree-based models still outperform deep learning on tabular data?" (Grinsztajn et al., 2022): https://arxiv.org/abs/2207.08815
- XGBoost paper (Chen & Guestrin, 2016): https://arxiv.org/abs/1603.02754
- LightGBM paper (Ke et al., 2017): https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- CatBoost paper (Prokhorenkova et al., 2018): https://arxiv.org/abs/1706.09516

## References

- Official docs (XGBoost): https://xgboost.readthedocs.io/en/stable/
- Official docs (LightGBM): https://lightgbm.readthedocs.io/en/stable/
- Official docs (CatBoost): https://catboost.ai/docs/
- GitHub (XGBoost): https://github.com/dmlc/xgboost
- Paper: https://arxiv.org/abs/1603.02754 (Chen & Guestrin, XGBoost: A Scalable Tree Boosting System, 2016)
- Paper: https://arxiv.org/abs/2207.08815 (Grinsztajn et al., Why tree-based models still outperform deep learning on tabular data, 2022)
