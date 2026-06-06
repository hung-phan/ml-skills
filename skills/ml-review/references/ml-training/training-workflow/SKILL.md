---
name: training-workflow
description: End-to-end model training workflow — cross-validation, hyperparameter tuning (Optuna, Ray Tune), experiment tracking (MLflow, W&B), sklearn Pipelines, model serialization (joblib, ONNX), early stopping, and learning curves. Use when setting up a reproducible training pipeline or choosing CV/HPO/tracking tooling.
---

# Model Training Workflow

## Why This Exists

The gap between "model.fit(X, y)" and production-ready ML is enormous. Without proper splitting you get data leakage, without cross-validation you get unreliable estimates, without hyperparameter tuning you leave performance on the table, without experiment tracking you can't reproduce results, and without serialization you can't deploy. This skill provides the complete scaffold — from first split to ONNX export — so every training run is reproducible, properly evaluated, and deployment-ready.

---

End-to-end patterns for training ML models: splitting, cross-validation, tuning, tracking, reproducibility, pipelines, and serialization.

---

## 1. Splitting Strategies

```python
import numpy as np
from sklearn.model_selection import (
    train_test_split, TimeSeriesSplit, GroupKFold, GroupShuffleSplit
)

# Stratified train/test split (classification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Time series split (no future leakage)
tscv = TimeSeriesSplit(n_splits=5, gap=0)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

# GroupKFold (no group leakage across folds)
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=groups):
    X_train, X_val = X[train_idx], X[val_idx]

# GroupShuffleSplit (random group-aware splits)
gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
for train_idx, val_idx in gss.split(X, y, groups=groups):
    X_train, X_val = X[train_idx], X[val_idx]
```

---

## 2. Cross-Validation

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, cross_validate
)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

# Basic KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print(f"KFold: {scores.mean():.4f} ± {scores.std():.4f}")

# StratifiedKFold (preserves class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')

# RepeatedStratifiedKFold (more robust estimate)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
results = cross_validate(model, X, y, cv=rskf,
                         scoring=['accuracy', 'f1_macro'],
                         return_train_score=True)
print(f"Test F1: {results['test_f1_macro'].mean():.4f}")
print(f"Train F1: {results['train_f1_macro'].mean():.4f}")  # check overfitting

# Nested CV for unbiased model selection (Cawley & Talbot 2010)
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs = GridSearchCV(model, param_grid, cv=inner_cv, scoring='f1_macro', n_jobs=-1)
nested_scores = cross_val_score(gs, X, y, cv=outer_cv, scoring='f1_macro')
print(f"Nested CV F1: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
```

### When to reach for nested CV (vs single-loop CV)

| Goal | Use |
|------|-----|
| Pick the best hyperparameters of *one* model family | Single-loop `GridSearchCV` / `Optuna` — the inner CV's best score is fine |
| Estimate generalization error of a *fully tuned* pipeline | Nested CV — outer fold gives an unbiased estimate, inner fold tunes |
| Compare *different model families* (e.g. RF vs XGBoost vs MLP) head-to-head | Nested CV — without it, the family with the largest tuning surface wins by overfitting the inner CV |
| Final model for production | Refit on all training data with the hyperparameters chosen by inner CV; use nested CV's outer score as the honest estimate |

The reason: the inner CV's best score is *biased upward* (it's been optimized over). Reporting it as your generalization estimate is the same mistake as reporting the training score. Nested CV's outer loop is the only honest answer when hyperparameters were tuned. For the live-rollout analog (offline → A/B), see [`../online-experimentation/`](../online-experimentation/).

---

## 3. Hyperparameter Tuning

```python
# --- GridSearchCV ---
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']}
gs = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print(f"Best: {gs.best_score_:.4f} with {gs.best_params_}")

# --- RandomizedSearchCV ---
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9),
}
rs = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist, n_iter=100, cv=5, scoring='f1_macro',
    random_state=42, n_jobs=-1
)
rs.fit(X_train, y_train)
print(f"Best: {rs.best_score_:.4f} with {rs.best_params_}")

# --- Optuna with pruning ---
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
    }
    model = GradientBoostingClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100, show_progress_bar=True)
print(f"Best trial: {study.best_trial.value:.4f}")
print(f"Best params: {study.best_trial.params}")

# --- Optuna + sklearn integration (OptunaSearchCV) ---
from optuna.integration import OptunaSearchCV

param_dist = {
    'n_estimators': optuna.distributions.IntDistribution(50, 500),
    'max_depth': optuna.distributions.IntDistribution(3, 15),
    'learning_rate': optuna.distributions.FloatDistribution(1e-3, 0.3, log=True),
}
optuna_cv = OptunaSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_dist, cv=5, n_trials=50, scoring='f1_macro', random_state=42
)
optuna_cv.fit(X_train, y_train)
print(f"Best: {optuna_cv.best_score_:.4f}")

# --- Ray Tune (brief) ---
from ray import tune
from ray.tune.sklearn import TuneSearchCV

param_dist = {
    'n_estimators': tune.randint(50, 500),
    'max_depth': tune.randint(3, 15),
    'learning_rate': tune.loguniform(1e-3, 0.3),
}
tune_search = TuneSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_dist, n_trials=50, cv=5, scoring='f1_macro',
    early_stopping=True, max_iters=10
)
tune_search.fit(X_train, y_train)
```

---

## 4. Early Stopping

```python
# --- sklearn (GradientBoosting with warm_start) ---
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
import numpy as np

# Manual early stopping with sklearn
best_score = -np.inf
patience_counter = 0
patience = 10
best_n = 0

for n_est in range(10, 510, 10):
    model = GradientBoostingClassifier(
        n_estimators=n_est, learning_rate=0.1, max_depth=5, random_state=42
    )
    model.fit(X_train, y_train)
    val_score = f1_score(y_val, model.predict(X_val), average='macro')
    if val_score > best_score:
        best_score = val_score
        best_n = n_est
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"Early stop at n_estimators={n_est}, best={best_n} (score={best_score:.4f})")
        break

# --- PyTorch early stopping ---
import torch

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Usage in training loop
early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)
    early_stopping(val_loss)
    if early_stopping.should_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## 5. Learning Curves

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, cv=5, scoring='f1_macro',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1, shuffle=True, random_state=42
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
plt.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('F1 Macro')
plt.title('Learning Curve')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=150)
plt.show()

# Diagnosis:
# - High bias: both curves converge low -> need more complex model or features
# - High variance: large gap between curves -> need more data or regularization
```

---

## 6. Experiment Tracking

```python
# --- MLflow autolog ---
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-classification")
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="rf-baseline"):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    # autolog captures params, metrics, model artifact

# --- MLflow manual logging ---
with mlflow.start_run(run_name="rf-tuned"):
    params = {'n_estimators': 200, 'max_depth': 15, 'min_samples_leaf': 3}
    mlflow.log_params(params)

    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
    mlflow.log_metric("cv_f1_mean", scores.mean())
    mlflow.log_metric("cv_f1_std", scores.std())
    mlflow.log_metric("test_f1", f1_score(y_test, model.predict(X_test), average='macro'))

    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("learning_curve.png")

# --- Weights & Biases ---
import wandb

wandb.init(project="my-classification", name="rf-tuned", config={
    'n_estimators': 200, 'max_depth': 15, 'model': 'RandomForest'
})

model = RandomForestClassifier(**wandb.config, random_state=42)
model.fit(X_train, y_train)

wandb.log({
    "cv_f1_mean": scores.mean(),
    "test_f1": f1_score(y_test, model.predict(X_test), average='macro'),
})
wandb.sklearn.plot_confusion_matrix(y_test, model.predict(X_test), labels=class_names)
wandb.finish()

# --- TensorBoard (PyTorch) ---
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment_1")
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
writer.close()
# Run: tensorboard --logdir runs/
```

---

## 7. Reproducibility

```python
import os
import random
import numpy as np

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    except ImportError:
        pass

set_seed(42)

# For DataLoader reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True,
    worker_init_fn=seed_worker, generator=g
)
```

---

## 8. Pipeline Construction

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# Basic pipeline
pipe = make_pipeline(
    StandardScaler(),
    SelectKBest(f_classif, k=20),
    RandomForestClassifier(n_estimators=100, random_state=42)
)
pipe.fit(X_train, y_train)

# ColumnTransformer for mixed types
numeric_features = ['age', 'income', 'score']
categorical_features = ['city', 'gender', 'category']

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), categorical_features),
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
full_pipeline.fit(X_train, y_train)
print(f"Test accuracy: {full_pipeline.score(X_test, y_test):.4f}")

# Custom transformer
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1.0):
        self.offset = offset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X + self.offset - 1)

# Use in pipeline
pipe = make_pipeline(LogTransformer(offset=1.0), StandardScaler(), RandomForestClassifier())
```

---

## 9. Model Serialization

```python
# --- joblib (sklearn standard) ---
import joblib

joblib.dump(full_pipeline, 'model_pipeline.joblib')
loaded_pipeline = joblib.load('model_pipeline.joblib')
assert (loaded_pipeline.predict(X_test) == full_pipeline.predict(X_test)).all()

# --- pickle ---
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# --- ONNX export from sklearn ---
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(full_pipeline, initial_types=initial_type)
with open('model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# Inference with ONNX Runtime
sess = ort.InferenceSession('model.onnx')
input_name = sess.get_inputs()[0].name
pred = sess.run(None, {input_name: X_test.astype(np.float32)})[0]

# --- ONNX export from PyTorch ---
import torch

dummy_input = torch.randn(1, input_dim)
torch.onnx.export(
    model, dummy_input, 'model.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=17
)

# --- TorchScript ---
scripted = torch.jit.script(model)
scripted.save('model_scripted.pt')
loaded = torch.jit.load('model_scripted.pt')
output = loaded(dummy_input)
```

---

## PyTorch Training Loop

Complete PyTorch training workflow with Dataset, DataLoader, optimizer, scheduler, train/val loop, early stopping, and checkpointing.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path

# --- Custom Dataset ---
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Model ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, checkpoint_path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = Path(checkpoint_path)
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def load_best(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path, weights_only=True))

# --- Training Loop ---
def train_model(X_train, y_train, X_val, y_val, config=None):
    config = config or {}
    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", 64)
    max_epochs = config.get("max_epochs", 200)
    patience = config.get("patience", 15)
    weight_decay = config.get("weight_decay", 1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoaders
    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    # Model, optimizer, scheduler
    model = MLP(input_dim=X_train.shape[1], num_classes=len(np.unique(y_train))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_ds)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_loss += criterion(logits, y_batch).item() * len(y_batch)
                correct += (logits.argmax(1) == y_batch).sum().item()
        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best weights
    early_stopping.load_best(model)
    return model

# --- Usage ---
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=5000, n_features=30, n_informative=15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = train_model(X_train, y_train, X_val, y_val, config={
    "lr": 3e-4, "batch_size": 128, "max_epochs": 100, "patience": 10
})
```

---

## When to Use

| Scenario | Technique |
|----------|-----------|
| Classification with class imbalance | StratifiedKFold, RepeatedStratifiedKFold |
| Time series data | TimeSeriesSplit (never shuffle) |
| Patient/user-level grouping | GroupKFold, GroupShuffleSplit |
| Unbiased model selection estimate | Nested CV (inner tune, outer evaluate) |
| Small param grid (<100 combos) | GridSearchCV |
| Large search space | RandomizedSearchCV or Optuna |
| Need pruning/Bayesian search | Optuna with MedianPruner |
| Distributed tuning | Ray Tune |
| Prevent overfitting deep models | Early stopping with patience |
| Diagnose under/overfitting | Learning curves |
| Team experiment tracking | MLflow (self-hosted) or W&B (cloud) |
| Production serving | ONNX export + ONNXRuntime |
| Python-only deployment | joblib serialization |
| GPU inference without Python | TorchScript |

---

## Common Gotchas

1. **Data leakage in CV**: Fit preprocessors inside each fold, not on full data. Use `Pipeline` to ensure transformations are fold-aware.
2. **Optimistic nested CV**: Using the same CV splitter for inner and outer loops inflates scores. Always use different random states.
3. **Time series shuffle**: Never `shuffle=True` with time series data — future leaks into past.
4. **GroupKFold + stratification**: `GroupKFold` doesn't stratify. Use `StratifiedGroupKFold` (sklearn ≥1.1) if you need both.
5. **Optuna SQLite concurrency**: Default in-memory storage isn't parallel-safe. Use `optuna.storages.RDBStorage("sqlite:///study.db")` for multi-worker.
6. **ONNX float types**: sklearn uses float64 internally but ONNX prefers float32. Always cast input to `float32` before inference.
7. **Reproducibility with DataLoader**: Must set `worker_init_fn` AND `generator` — `torch.manual_seed` alone doesn't seed workers.
8. **joblib across versions**: Models serialized with one sklearn version may fail on another. Pin sklearn version in requirements.
9. **Early stopping restore**: Always restore best weights, not final weights after patience exhaustion.
10. **W&B secrets**: Never log API keys. Use `WANDB_API_KEY` env var or `wandb login` once.

---

## References

- sklearn model_selection: https://scikit-learn.org/stable/modules/cross_validation.html
- Optuna: https://optuna.readthedocs.io/
- Ray Tune: https://docs.ray.io/en/latest/tune/index.html
- MLflow: https://mlflow.org/docs/latest/index.html
- Weights & Biases: https://docs.wandb.ai/
- ONNX: https://onnxruntime.ai/docs/
- PyTorch reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
- Cawley & Talbot (2010) "On Over-fitting in Model Selection": https://jmlr.org/papers/v11/cawley10a.html
- sklearn Pipelines: https://scikit-learn.org/stable/modules/compose.html
- skl2onnx: https://onnx.ai/sklearn-onnx/
