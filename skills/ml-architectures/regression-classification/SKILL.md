---
name: regression-classification
description: Output layer and loss design for regression and classification — MSE/CE/Focal losses, metrics, class imbalance handling, multi-label and ordinal patterns, calibration, threshold tuning, and complete sklearn/PyTorch training loops. Use when picking the right output head, loss, and metric for a supervised task.
---

## Why This Exists

**Problem**: Every supervised learning task ultimately requires mapping features to either a continuous value (regression) or discrete categories (classification), but choosing the right loss function, output activation, handling class imbalance, and calibrating predictions requires careful design decisions that drastically affect performance.

**Key insight**: The output layer, loss function, and evaluation metric form a tightly coupled system — sigmoid+BCE for binary, softmax+CE for multi-class, linear+MSE for regression — and getting this wrong (e.g., using accuracy on imbalanced data, or MSE on classification) silently degrades results.

**Reach for this when**: You're designing the final prediction layer of any neural network or ML model, dealing with class imbalance (Focal loss, SMOTE), need probability calibration, multi-label classification, ordinal regression, or threshold tuning. This is a reference for the output stage of any supervised model.


# Regression & Classification Patterns

## Output Layer Design

| Task | Output Activation | Output Size | Notes |
|------|------------------|-------------|-------|
| Regression | Linear (none) | 1 (or N for multi-output) | Unbounded output |
| Binary classification | Sigmoid | 1 | Output ∈ [0,1] |
| Multi-class | Softmax | C (num classes) | Outputs sum to 1 |
| Multi-label | Sigmoid | C | Independent per-label probability |
| Ordinal regression | Sigmoid | K-1 (thresholds) | Cumulative probabilities |

```python
# PyTorch output layers
import torch.nn as nn

# Regression
nn.Linear(hidden, 1)  # no activation, loss handles it

# Binary classification
nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
# Or: raw logits + BCEWithLogitsLoss (preferred, numerically stable)

# Multi-class
nn.Linear(hidden, num_classes)  # raw logits + CrossEntropyLoss

# Multi-label
nn.Linear(hidden, num_labels)  # raw logits + BCEWithLogitsLoss
```

## Loss Function Selection

### Regression

| Loss | Formula | When to Use |
|------|---------|-------------|
| MSE (L2) | `(y - ŷ)²` | Default; penalizes large errors heavily |
| MAE (L1) | `|y - ŷ|` | Robust to outliers; median-seeking |
| Huber | MSE if `|e|<δ`, else MAE | Balanced; δ controls transition |
| Log-cosh | `log(cosh(y - ŷ))` | Smooth approximation of Huber |
| Quantile | Asymmetric MAE | Prediction intervals |

```python
# PyTorch
nn.MSELoss()
nn.L1Loss()
nn.HuberLoss(delta=1.0)
nn.SmoothL1Loss()  # equivalent to Huber with delta=1

# scikit-learn
from sklearn.metrics import mean_squared_error, mean_absolute_error
```

### Classification

| Loss | When to Use |
|------|-------------|
| BCE (Binary Cross-Entropy) | Binary or multi-label |
| CE (Cross-Entropy) | Multi-class (mutually exclusive) |
| Focal Loss | Severe class imbalance |
| Label Smoothing CE | Prevent overconfidence |
| Hinge / SVM Loss | Max-margin classification |

```python
# PyTorch — always prefer logit versions for numerical stability
nn.BCEWithLogitsLoss()                    # binary/multi-label
nn.CrossEntropyLoss()                     # multi-class (expects raw logits)
nn.CrossEntropyLoss(label_smoothing=0.1)  # with label smoothing

# Focal loss (manual)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()
```

## Metrics

### Regression

| Metric | Interpretation | Pitfall |
|--------|---------------|---------|
| R² | Variance explained (1=perfect) | Can be negative; meaningless without baseline |
| RMSE | Same units as target | Sensitive to outliers |
| MAE | Median error magnitude | More robust than RMSE |
| MAPE | Percentage error | Undefined when y=0 |

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

### Classification

| Metric | Use Case |
|--------|----------|
| Accuracy | Balanced classes only |
| Precision | Cost of false positives high (spam) |
| Recall | Cost of false negatives high (fraud, disease) |
| F1 | Balance precision/recall |
| AUC-ROC | Ranking quality, threshold-independent |
| AUC-PR | Imbalanced datasets (prefer over ROC) |
| Log Loss | Probabilistic calibration quality |

```python
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, classification_report
)

# Multi-class AUC
roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
```

## Class Imbalance Handling

### Weighted Loss

```python
# PyTorch — pass class weights inversely proportional to frequency
weights = 1.0 / class_counts
weights = weights / weights.sum() * len(weights)
nn.CrossEntropyLoss(weight=torch.tensor(weights))

# BCE with pos_weight for binary
pos_weight = num_neg / num_pos
nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
```

### Sampling Strategies

```python
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# SMOTE (synthetic minority oversampling)
sm = SMOTE(sampling_strategy='minority', k_neighbors=5)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Combined pipeline
pipeline = ImbPipeline([
    ('over', SMOTE(sampling_strategy=0.5)),
    ('under', RandomUnderSampler(sampling_strategy=0.8)),
    ('model', LogisticRegression())
])

# PyTorch weighted sampler
from torch.utils.data import WeightedRandomSampler
sample_weights = [1/class_counts[y] for y in labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels))
```

### Focal Loss

Downweights easy examples, focuses on hard ones. γ=2 is standard starting point.

## Multi-Label vs Multi-Class

| Aspect | Multi-Class | Multi-Label |
|--------|-------------|-------------|
| Constraint | Exactly one class | Zero or more labels |
| Activation | Softmax | Sigmoid (per label) |
| Loss | CrossEntropyLoss | BCEWithLogitsLoss |
| Metric | Accuracy, macro-F1 | Subset accuracy, hamming loss, per-label AUC |
| Threshold | argmax | Per-label threshold (default 0.5) |

```python
# scikit-learn multi-label
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import hamming_loss, multilabel_confusion_matrix

# PyTorch multi-label training loop
logits = model(x)  # shape: (batch, num_labels)
loss = nn.BCEWithLogitsLoss()(logits, labels.float())
preds = (torch.sigmoid(logits) > 0.5).int()
```

## Ordinal Regression

For ordered categories (e.g., star ratings 1-5, severity low/med/high).

```python
# CORAL approach: K-1 binary classifiers for cumulative probabilities
# P(Y > k) for k = 1, ..., K-1
class OrdinalModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.features = nn.Linear(input_dim, 128)
        self.thresholds = nn.Linear(128, num_classes - 1)

    def forward(self, x):
        h = torch.relu(self.features(x))
        cumprobs = torch.sigmoid(self.thresholds(h))
        return cumprobs

    def predict(self, x):
        cumprobs = self.forward(x)
        return (cumprobs > 0.5).sum(dim=1)  # predicted class

# Loss: sum of binary cross-entropies per threshold
def ordinal_loss(cumprobs, targets, num_classes):
    # targets: integer class labels 0..K-1
    # Create binary targets: Y > k for each k
    binary_targets = (targets.unsqueeze(1) > torch.arange(num_classes - 1)).float()
    return nn.BCELoss()(cumprobs, binary_targets)
```

## Calibration

Calibrated models output probabilities that match empirical frequencies.

### Temperature Scaling (post-hoc, preserves ranking)

```python
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

# Fit on validation set
temp_model = TemperatureScaling()
optimizer = torch.optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)

def closure():
    optimizer.zero_grad()
    scaled = temp_model(val_logits)
    loss = nn.CrossEntropyLoss()(scaled, val_labels)
    loss.backward()
    return loss

optimizer.step(closure)
```

### Platt Scaling (binary)

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

# Wrap any classifier with Platt scaling (sigmoid method)
calibrated = CalibratedClassifierCV(base_estimator, method='sigmoid', cv=5)
calibrated.fit(X_train, y_train)

# Isotonic regression (non-parametric, needs more data)
calibrated = CalibratedClassifierCV(base_estimator, method='isotonic', cv=5)
```

### Reliability Diagram

```python
from sklearn.calibration import calibration_curve
fraction_pos, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10)
# Plot mean_predicted vs fraction_pos; perfect = diagonal
```

## Threshold Tuning

Default 0.5 is rarely optimal. Tune on validation set.

```python
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1s)]

# Or optimize for business metric (e.g., cost-sensitive)
# cost = FP_cost * FP_count + FN_cost * FN_count
```

## Feature Scaling

| Algorithm | Needs Scaling? | Why |
|-----------|---------------|-----|
| Linear/Logistic Regression | Yes | Gradient-based; regularization assumes comparable scale |
| SVM | Yes | Distance-based |
| KNN | Yes | Distance-based |
| Neural Networks | Yes | Gradient stability, faster convergence |
| Tree-based (RF, XGBoost) | No | Split-based, scale-invariant |
| Naive Bayes | No | Probability-based |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler: zero mean, unit variance (default choice)
# MinMaxScaler: [0,1] range (use for bounded activations)
# RobustScaler: uses median/IQR (robust to outliers)

# ALWAYS fit on train, transform train+val+test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Regularization

| Method | Effect | Use Case |
|--------|--------|----------|
| L1 (Lasso) | Sparse weights (feature selection) | High-dimensional, few relevant features |
| L2 (Ridge) | Small weights (shrinkage) | Default; prevents overfitting |
| ElasticNet | L1 + L2 mix | Groups of correlated features |
| Dropout | Random neuron zeroing | Neural networks |
| Weight Decay | L2 on optimizer level | PyTorch standard |

```python
# scikit-learn
from sklearn.linear_model import LogisticRegression, ElasticNet
LogisticRegression(penalty='l1', C=0.1, solver='saga')
LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga')
ElasticNet(alpha=0.1, l1_ratio=0.5)  # regression

# PyTorch weight decay (L2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# PyTorch L1 manually
l1_lambda = 1e-5
l1_reg = sum(p.abs().sum() for p in model.parameters())
loss = criterion(output, target) + l1_lambda * l1_reg
```

## Complete scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier())
])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.01, 0.1]
}

search = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
search.fit(X_train, y_train)
```

## Complete PyTorch Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1 if num_classes == 2 else num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Setup
model = Classifier(input_dim=20, num_classes=2)
criterion = nn.BCEWithLogitsLoss()  # binary
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Training
for epoch in range(50):
    model.train()
    for X_batch, y_batch in train_loader:
        logits = model(X_batch).squeeze()
        loss = criterion(logits, y_batch.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

# Inference with threshold
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_test).squeeze())
    preds = (probs > best_threshold).int()
```

## Decision Guide

1. **Start simple**: LogisticRegression / LinearRegression as baseline
2. **Scale features** if using linear/distance/gradient methods
3. **Choose loss** matching your error tolerance (MSE vs Huber, CE vs Focal)
4. **Handle imbalance** before training (weighted loss > sampling > SMOTE)
5. **Calibrate** if you need reliable probabilities (temperature scaling)
6. **Tune threshold** on validation set using business-relevant metric
7. **Regularize** to prevent overfitting (start with L2/weight decay)
8. **Evaluate** with appropriate metrics (never accuracy alone on imbalanced data)

## When to Use

| Task | ✅ Use When | Model Escalation |
|---|---|---|
| **Regression** | Continuous numeric target (price, temperature, score) | Linear → Ridge/Lasso → XGBoost → MLP → deep |
| **Classification** | Discrete categories (spam/not, species, sentiment) | Logistic → RF/XGBoost → MLP → CNN/Transformer |
| **Multi-label** | Item can belong to multiple classes | BCE loss, sigmoid per class |
| **Ordinal** | Ordered categories (star ratings) | Ordinal regression, cumulative link |

**Decision rule**: Always start with a simple baseline (linear/logistic). If it works within 2% of deep learning, keep it. Tabular → tree-based first. Unstructured → deep learning.

---

## References

- [scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) — Metrics, scoring, and cross-validation
- [PyTorch Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions) — Complete loss function API reference
