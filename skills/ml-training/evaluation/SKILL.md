---
name: evaluation
description: Model evaluation and class imbalance handling — metrics beyond accuracy (F1, ROC-AUC, PR-AUC, MCC), probability calibration, threshold optimization, statistical model comparison, and SMOTE/ADASYN sampling. Use when evaluating classifiers under imbalance or picking the right metric for a deployment decision.
---

# Evaluation and Class Imbalance

## Why This Exists

Most real-world classification problems have imbalanced classes (fraud detection, medical diagnosis, churn prediction) where accuracy is meaningless and naive models predict majority class. This skill solves three critical problems: (1) choosing metrics that actually reflect minority-class performance (PR-AUC, MCC, F1), (2) rebalancing training data without leaking information across CV folds, and (3) statistically proving one model is better than another rather than relying on single-split luck. Without these techniques, you ship models that look good on paper but fail on the cases that matter most.

---

Comprehensive toolkit for handling imbalanced datasets, computing robust evaluation metrics, calibrating probabilities, optimizing decision thresholds, and statistically comparing models.

---

## 1. Class Imbalance Techniques

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create imbalanced dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15,
                           n_redundant=5, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     stratify=y, random_state=42)

# --- Oversampling ---
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler

smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

adasyn = ADASYN(sampling_strategy='auto', n_neighbors=5, random_state=42)
X_res, y_res = adasyn.fit_resample(X_train, y_train)

bsmote = BorderlineSMOTE(sampling_strategy='auto', kind='borderline-1', random_state=42)
X_res, y_res = bsmote.fit_resample(X_train, y_train)

ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)

# --- Undersampling ---
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)

# --- Combined ---
from imblearn.combine import SMOTETomek

smt = SMOTETomek(sampling_strategy='auto', random_state=42)
X_res, y_res = smt.fit_resample(X_train, y_train)

# --- class_weight='balanced' ---
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# --- Focal Loss (PyTorch) ---
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal

# Usage
loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
logits = torch.randn(32, 1)
targets = torch.randint(0, 2, (32, 1))
loss = loss_fn(logits, targets)
```

---

## 2. Evaluation Metrics

```python
from sklearn.metrics import (
    classification_report, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, log_loss, brier_score_loss, confusion_matrix
)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Classification report
print(classification_report(y_test, y_pred, digits=4))

# Individual metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average='binary')

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)  # PR-AUC
mcc = matthews_corrcoef(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
logloss = log_loss(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
print(f"MCC: {mcc:.4f} | Kappa: {kappa:.4f}")
print(f"Log Loss: {logloss:.4f} | Brier: {brier:.4f}")
```

---

## 3. Multi-Class Metrics and Confusion Matrix

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Multi-class averaging strategies
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test, y_pred, average='macro')      # Unweighted mean across classes
precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
    y_test, y_pred, average='micro')      # Global TP/FP/FN
precision_wt, recall_wt, f1_wt, _ = precision_recall_fscore_support(
    y_test, y_pred, average='weighted')   # Support-weighted mean

# Multi-class ROC-AUC (One-vs-Rest)
# For multi-class: y_prob_mc = clf.predict_proba(X_test)  # shape (n, n_classes)
# roc_auc_ovr = roc_auc_score(y_test, y_prob_mc, multi_class='ovr', average='macro')

# Confusion matrix visualization
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues',
                                         normalize='true')
ax.set_title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
```

---

## 4. Probability Calibration

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Platt scaling (sigmoid) - best for SVMs, boosted trees
base_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
cal_clf_platt = CalibratedClassifierCV(base_clf, method='sigmoid', cv=5)
cal_clf_platt.fit(X_train, y_train)

# Isotonic regression - more flexible, needs more data
cal_clf_iso = CalibratedClassifierCV(base_clf, method='isotonic', cv=5)
cal_clf_iso.fit(X_train, y_train)

# Calibration curve (reliability diagram)
y_prob_cal = cal_clf_platt.predict_proba(X_test)[:, 1]
fraction_pos, mean_predicted = calibration_curve(y_test, y_prob_cal, n_bins=10)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(mean_predicted, fraction_pos, 's-', label='Calibrated (Platt)')
ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curve")
ax.legend()
plt.tight_layout()
plt.savefig("calibration_curve.png", dpi=150)
plt.close()

# Expected Calibration Error (ECE)
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true)

ece = expected_calibration_error(y_test, y_prob_cal)
print(f"ECE: {ece:.4f}")
```

---

## 5. Threshold Optimization

```python
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score

# --- Youden's J Statistic (maximizes TPR - FPR) ---
fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
threshold_youden = thresholds_roc[best_idx]
print(f"Youden's J threshold: {threshold_youden:.4f} (J={j_scores[best_idx]:.4f})")

# --- F1-Optimal Threshold ---
precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_f1_idx = np.argmax(f1_scores)
threshold_f1 = thresholds_pr[best_f1_idx]
print(f"F1-optimal threshold: {threshold_f1:.4f} (F1={f1_scores[best_f1_idx]:.4f})")

# --- Cost-Matrix Threshold ---
def cost_threshold(y_true, y_prob, cost_fp=1.0, cost_fn=5.0, n_thresholds=100):
    """Find threshold minimizing total cost: cost_fp*FP + cost_fn*FN."""
    thresholds = np.linspace(0, 1, n_thresholds)
    costs = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        fp = ((y_pred_t == 1) & (y_true == 0)).sum()
        fn = ((y_pred_t == 0) & (y_true == 1)).sum()
        costs.append(cost_fp * fp + cost_fn * fn)
    best_idx = np.argmin(costs)
    return thresholds[best_idx], costs[best_idx]

threshold_cost, min_cost = cost_threshold(y_test, y_prob, cost_fp=1, cost_fn=10)
print(f"Cost-optimal threshold: {threshold_cost:.4f} (cost={min_cost:.0f})")

# Apply chosen threshold
y_pred_opt = (y_prob >= threshold_f1).astype(int)
```

---

## 6. Statistical Model Comparison

```python
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

# Setup: two models, repeated CV scores
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
scores_a = cross_val_score(RandomForestClassifier(random_state=42),
                           X_train, y_train, cv=cv, scoring='f1')
scores_b = cross_val_score(GradientBoostingClassifier(random_state=42),
                           X_train, y_train, cv=cv, scoring='f1')

# --- Paired t-test (naive, tends to overestimate significance) ---
t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")

# --- Corrected Resampled t-test (Nadeau & Bengio, 2003) ---
def corrected_resampled_ttest(scores_a, scores_b, n_train, n_test, n_splits=10, n_repeats=10):
    """Accounts for non-independence of overlapping training sets."""
    diffs = scores_a - scores_b
    mean_diff = diffs.mean()
    var_diff = diffs.var(ddof=1)
    n = n_splits * n_repeats
    # Correction factor for repeated CV
    correction = (1/n + n_test/n_train)
    t_stat = mean_diff / np.sqrt(correction * var_diff)
    df = n - 1
    p_value = 2 * stats.t.sf(abs(t_stat), df)
    return t_stat, p_value

n_train_fold = int(len(X_train) * 0.9)
n_test_fold = len(X_train) - n_train_fold
t_corr, p_corr = corrected_resampled_ttest(scores_a, scores_b, n_train_fold, n_test_fold)
print(f"Corrected t-test: t={t_corr:.4f}, p={p_corr:.4f}")

# --- Wilcoxon Signed-Rank (non-parametric) ---
w_stat, p_wilcox = stats.wilcoxon(scores_a, scores_b)
print(f"Wilcoxon: W={w_stat:.4f}, p={p_wilcox:.4f}")

# --- McNemar's Test (compares two classifiers on same test set) ---
from statsmodels.stats.contingency_tables import mcnemar

clf_a = RandomForestClassifier(random_state=42).fit(X_train, y_train)
clf_b = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
pred_a = clf_a.predict(X_test)
pred_b = clf_b.predict(X_test)

# Contingency table: [both_correct, a_correct_b_wrong; a_wrong_b_correct, both_wrong]
correct_a = (pred_a == y_test)
correct_b = (pred_b == y_test)
contingency = np.array([
    [(correct_a & correct_b).sum(), (correct_a & ~correct_b).sum()],
    [(~correct_a & correct_b).sum(), (~correct_a & ~correct_b).sum()]
])
result = mcnemar(contingency, exact=False, correction=True)
print(f"McNemar's: chi2={result.statistic:.4f}, p={result.pvalue:.4f}")

# --- Friedman + Nemenyi (3+ models) ---
from scipy.stats import friedmanchisquare

scores_c = cross_val_score(LogisticRegression(max_iter=1000, random_state=42),
                           X_train, y_train, cv=cv, scoring='f1')
# Reshape: each row = one fold, each column = one model
stat_f, p_friedman = friedmanchisquare(scores_a, scores_b, scores_c)
print(f"Friedman: chi2={stat_f:.4f}, p={p_friedman:.4f}")

# Nemenyi post-hoc (requires scikit-posthocs)
# pip install scikit-posthocs
import scikit_posthocs as sp
scores_matrix = np.column_stack([scores_a, scores_b, scores_c])
nemenyi = sp.posthoc_nemenyi_friedman(scores_matrix)
print("Nemenyi pairwise p-values:")
print(nemenyi)
```

---

## 7. Learning Curves for Diagnosis

```python
from sklearn.model_selection import learning_curve, validation_curve

# --- Learning Curve (bias vs variance) ---
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train, y_train, cv=5, scoring='f1',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
ax.plot(train_sizes, train_mean, 'o-', label='Training')
ax.plot(train_sizes, val_mean, 'o-', label='Validation')
ax.set_xlabel("Training Set Size")
ax.set_ylabel("F1 Score")
ax.set_title("Learning Curve (High Bias = both low, High Variance = gap)")
ax.legend()
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
plt.close()

# --- Validation Curve (hyperparameter effect) ---
param_range = [10, 50, 100, 200, 500]
train_scores_vc, val_scores_vc = validation_curve(
    RandomForestClassifier(random_state=42),
    X_train, y_train, param_name='n_estimators',
    param_range=param_range, cv=5, scoring='f1', n_jobs=-1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(param_range, train_scores_vc.mean(axis=1), 'o-', label='Training')
ax.plot(param_range, val_scores_vc.mean(axis=1), 'o-', label='Validation')
ax.set_xlabel("n_estimators")
ax.set_ylabel("F1 Score")
ax.set_title("Validation Curve")
ax.legend()
plt.tight_layout()
plt.savefig("validation_curve.png", dpi=150)
plt.close()
```

---

## 8. Complete Evaluation Pipeline with Imbalanced-Learn

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, average_precision_score
import numpy as np

# Build imbalanced-learn pipeline (sampling happens inside CV correctly)
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('sampler', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                    random_state=42))
])

# Define scoring
scoring = {
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
    'pr_auc': make_scorer(average_precision_score, needs_proba=True),
}

# Stratified CV preserves class ratio in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate(pipeline, X_train, y_train, cv=cv,
                         scoring=scoring, return_train_score=True, n_jobs=-1)

print("=== Cross-Validation Results ===")
for metric in ['f1', 'roc_auc', 'pr_auc']:
    train_key = f'train_{metric}'
    test_key = f'test_{metric}'
    print(f"{metric:>8}: train={results[train_key].mean():.4f}±{results[train_key].std():.4f} "
          f"| val={results[test_key].mean():.4f}±{results[test_key].std():.4f}")

# Final fit and evaluation on held-out test set
pipeline.fit(X_train, y_train)
y_pred_final = pipeline.predict(X_test)
y_prob_final = pipeline.predict_proba(X_test)[:, 1]

print("\n=== Test Set Results ===")
print(classification_report(y_test, y_pred_final, digits=4))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_final):.4f}")
print(f"PR-AUC:  {average_precision_score(y_test, y_prob_final):.4f}")
print(f"MCC:     {matthews_corrcoef(y_test, y_pred_final):.4f}")
```

---

## When to Use

| Scenario | Technique |
|----------|-----------|
| Minority class < 5% of data | SMOTE + class_weight='balanced' |
| Very small minority (< 100 samples) | RandomOverSampler (SMOTE needs k neighbors) |
| Noisy minority boundary | BorderlineSMOTE or ADASYN |
| Need clean decision boundary | SMOTETomek (oversample + remove Tomek links) |
| Metric for imbalanced data | PR-AUC, F1, MCC (not accuracy or ROC-AUC alone) |
| Probabilities needed for ranking | Calibrate with Platt/isotonic, use Brier/ECE |
| Cost-sensitive domain (fraud, medical) | Cost-matrix threshold + focal loss |
| Comparing 2 models rigorously | Corrected resampled t-test or McNemar's |
| Comparing 3+ models | Friedman test + Nemenyi post-hoc |
| Model underfitting (high bias) | Learning curve shows both scores low |
| Model overfitting (high variance) | Learning curve shows large train-val gap |
| Pipeline with sampling + CV | imblearn.pipeline.Pipeline (NOT sklearn Pipeline) |

---

---

## 9. "Too-Good-To-Be-True" — Leakage and Sanity-Check Playbook

Before celebrating a strong CV score, run this checklist. Almost every ML team has shipped a model that "won" CV because of a leak the validation pipeline couldn't see. The pattern is universal: a feature secretly encodes the label.

### Red flags that should trigger a leakage hunt

| Red flag | Likely cause | Diagnostic |
|----------|--------------|-----------|
| CV score >> domain prior / human baseline | Target leak | Check feature importance — single feature dominates |
| One feature has importance > all others combined | Direct or near-direct label encoding | Drop it, retrain — score collapses if true leak |
| AUC stays high even with random labels shuffled | Bug in eval / shuffled labels not actually shuffled | Sanity test: shuffle y, AUC must drop to ~0.5 |
| Train AUC ≈ Test AUC and both are very high | Group leak (same entity in train and test) | Use `GroupKFold` on the entity ID |
| Performance degrades sharply from CV to live | Train uses a feature the serving pipeline can't compute at request-time | Compare train-time and serve-time feature schemas exactly |
| Time-aware CV score >> shuffled-CV score on the SAME data | Future leak via post-event features | Audit each feature: is it knowable at the prediction timestamp? |

### Leakage families to audit explicitly

1. **Post-event features** — `payment_type` for fraud is a leak if it's only populated *after* the fraud is flagged. Test: redact the feature on a held-out window and re-score.
2. **Group leak** — multiple rows per user/store/patient end up split across train and val. Use `GroupKFold` / `GroupShuffleSplit`.
3. **Target encoding** — encoding categorical with the global mean leaks the labels of held-out rows. Use expanding-mean (past-only) target encoding.
4. **Pre-split preprocessing** — fitting `StandardScaler` / imputation on the full dataset then splitting. Always fit transformers inside the fold (use `Pipeline`).
5. **Duplicate rows** — same row in train and test (common in scraped data). Hash + dedupe before splitting.
6. **Time leak** — random shuffle on a time-indexed dataset; future bleeds into past. See [`../../data-prep/time-series-features/`](../../data-prep/time-series-features/) for purged + embargoed CV.

### Negative-control / shuffle test (cheap, definitive)

```python
from sklearn.utils import shuffle
import numpy as np

# Shuffle y. A model that trains on (X, shuffled_y) MUST score ~chance.
# If it doesn't, your "score" is from a leak, not signal.
y_shuffled = shuffle(y, random_state=0)
score_shuffled = cross_val_score(model, X, y_shuffled, cv=5, scoring='roc_auc').mean()
print(f"Shuffled-label AUC: {score_shuffled:.4f}")  # expect ~0.50
```

If the shuffled-label model scores meaningfully above 0.5, your CV pipeline is broken (most often: target encoding fitted before the split, or a feature derived from `y` itself).

For the live-rollout side of this story (offline → A/B test, guardrails, CUPED), see [`../online-experimentation/`](../online-experimentation/). Offline beats live by default — confirm with a real experiment.

---

## Common Gotchas

1. **Data leakage with SMOTE**: Never apply SMOTE before cross-validation. Use `imblearn.pipeline.Pipeline` so resampling happens inside each fold.
2. **ROC-AUC misleading on imbalanced data**: ROC-AUC can be high even with poor minority recall. Always pair with PR-AUC.
3. **Accuracy paradox**: 95% accuracy means nothing if baseline is 95% majority class. Use MCC or balanced accuracy.
4. **Calibration after threshold**: If you calibrate probabilities then apply a threshold ≠ 0.5, the calibration guarantee no longer holds at that operating point.
5. **SMOTE on high-dimensional sparse data**: SMOTE interpolates in feature space — unreliable for sparse/text features. Use RandomOverSampler or class weights instead.
6. **Corrected t-test, not naive t-test**: Repeated CV folds share training data, violating independence. Always use the Nadeau & Bengio correction.
7. **McNemar's requires same test set**: Both classifiers must predict the exact same test instances.
8. **Isotonic calibration needs 1000+ samples**: With fewer calibration samples, Platt (sigmoid) is more stable.
9. **Focal loss gamma tuning**: gamma=2 is a starting point. Higher gamma focuses more on hard examples but can destabilize training.
10. **imblearn Pipeline vs sklearn Pipeline**: `sklearn.pipeline.Pipeline` does NOT support sampling steps. You must use `imblearn.pipeline.Pipeline`.

---

## References

- **imbalanced-learn documentation**: https://imbalanced-learn.org/stable/
- **scikit-learn model evaluation**: https://scikit-learn.org/stable/modules/model_evaluation.html
- **scikit-learn calibration**: https://scikit-learn.org/stable/modules/calibration.html
- **Chawla et al. (2002)** — SMOTE: Synthetic Minority Over-sampling Technique: https://arxiv.org/abs/1106.1813
- **He & Garcia (2009)** — Learning from Imbalanced Data: https://doi.org/10.1109/TKDE.2008.239
- **Nadeau & Bengio (2003)** — Inference for the Generalization Error (corrected resampled t-test): https://link.springer.com/article/10.1023/A:1024068626366
- **Niculescu-Mizil & Caruana (2005)** — Predicting Good Probabilities with Supervised Learning: https://dl.acm.org/doi/10.1145/1102351.1102430
- **Demšar (2006)** — Statistical Comparisons of Classifiers over Multiple Data Sets: https://jmlr.org/papers/v7/demsar06a.html
