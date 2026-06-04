---
name: ann
description: Multi-layer perceptrons (MLPs) for tabular classification and regression, plus activation, initialization, and normalization reference. Use when building feedforward networks for tabular data or need a quick reference for ReLU/GELU/SwiGLU, He/Xavier init, or BatchNorm/LayerNorm choices.
---

## Why This Exists

**Problem**: Tabular data (customer features, sensor readings, financial metrics) needs a flexible function approximator that can learn arbitrary non-linear relationships between input features and outputs — decision trees handle this but can't extrapolate, and linear models miss interactions.

**Key insight**: Stacking layers of linear transforms with non-linear activations creates a universal function approximator — the network learns its own feature combinations rather than requiring manual feature engineering.

**Reach for this when**: You have structured/tabular data, need a simple baseline before trying tree ensembles (XGBoost), or need a differentiable model that can be embedded as a component in a larger neural architecture.


# Artificial Neural Networks (ANN) — From Perceptron to Deep MLP

## Perceptron to MLP

**Single Perceptron**: Linear classifier — `y = σ(w·x + b)`. Can only learn linearly separable functions (fails XOR).

**Multi-Layer Perceptron (MLP)**: Stack of fully-connected layers with non-linear activations. Each layer transforms: `h = activation(W @ x + b)`.

Architecture notation: `[input_dim] → [hidden1] → [hidden2] → ... → [output_dim]`

```python
# PyTorch MLP — shape annotations
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: list[int], out_features: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]  # [B, prev] → [B, h]
            prev = h
        layers.append(nn.Linear(prev, out_features))  # [B, prev] → [B, out]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [B, in_features]
        return self.net(x)  # out: [B, out_features]
```

```python
# Keras equivalent
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_mlp(in_features, hidden, out_features, dropout=0.1):
    inputs = tf.keras.Input(shape=(in_features,))  # [B, in_features]
    x = inputs
    for h in hidden:
        x = layers.Dense(h, activation='relu')(x)   # [B, h]
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(out_features)(x)          # [B, out_features]
    return Model(inputs, outputs)
```

---

## Activation Functions

| Function | Formula | Pros | Cons | Use When |
|----------|---------|------|------|----------|
| **ReLU** | `max(0, x)` | Fast, sparse activation, mitigates vanishing gradient | Dead neurons (output=0 forever if weights push input negative) | Default for CNNs, simple MLPs |
| **GELU** | `x · Φ(x)` ≈ `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))` | Smooth, probabilistic gating, better for NLP | Slightly slower than ReLU | Transformers (BERT, GPT, ViT) |
| **SiLU/Swish** | `x · σ(x)` | Smooth, non-monotonic, self-gated | Unbounded below (can produce small negatives) | EfficientNet, modern vision |
| **Leaky ReLU** | `max(αx, x)`, α=0.01 | No dead neurons | Marginal improvement over ReLU | When dead neurons are a problem |

**Practical rule**: GELU for transformers, SiLU/Swish for modern vision, ReLU everywhere else unless you have a reason.

```python
# PyTorch
nn.ReLU()           # max(0, x)
nn.GELU()           # x * Φ(x), default approximate='none'
nn.SiLU()           # x * sigmoid(x), same as Swish with β=1

# Keras
layers.Activation('relu')
layers.Activation('gelu')
layers.Activation('swish')  # identical to silu
```

---

## Weight Initialization

Poor initialization → vanishing/exploding activations from layer 1.

| Method | Formula (fan_in=n_in, fan_out=n_out) | Use With |
|--------|--------------------------------------|----------|
| **Xavier/Glorot Uniform** | `U(-√(6/(n_in+n_out)), √(6/(n_in+n_out)))` | Sigmoid, Tanh, Linear |
| **Xavier/Glorot Normal** | `N(0, 2/(n_in+n_out))` | Same |
| **He/Kaiming Uniform** | `U(-√(6/n_in), √(6/n_in))` | ReLU, Leaky ReLU |
| **He/Kaiming Normal** | `N(0, 2/n_in)` | ReLU (default PyTorch for Linear) |

**Intuition**: Maintain variance of activations ≈ 1 across layers. ReLU zeros half the distribution → need 2× variance → He init.

```python
# PyTorch — nn.Linear uses Kaiming Uniform by default
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.xavier_normal_(layer.weight)  # for tanh/sigmoid
nn.init.zeros_(layer.bias)

# Keras — Dense uses Glorot Uniform by default
layers.Dense(256, kernel_initializer='he_normal')  # for ReLU
layers.Dense(256, kernel_initializer='glorot_normal')  # for tanh
```

---

## Backpropagation Intuition

Backprop = chain rule applied recursively through computation graph.

1. **Forward pass**: Compute output layer by layer, cache intermediate activations
2. **Loss**: Scalar measuring prediction error
3. **Backward pass**: Compute ∂Loss/∂w for every weight via chain rule:
   - At output: `∂L/∂w_L = ∂L/∂a_L · ∂a_L/∂z_L · ∂z_L/∂w_L`
   - Propagate: `∂L/∂a_{l} = ∂L/∂z_{l+1} · W_{l+1}ᵀ`
4. **Update**: `w ← w - lr · ∂L/∂w`

**Key insight**: Gradients flow backward as matrix multiplications. If any layer's Jacobian has singular values consistently >1 or <1, gradients explode or vanish over many layers.

---

## Batch Normalization vs Layer Normalization

| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| **Normalizes over** | Batch dimension (per feature) | Feature dimension (per sample) |
| **Stats shape** | `[C]` (mean/var per channel) | `[C]` or `[H, W, C]` per sample |
| **Batch dependency** | Yes (needs batch stats) | No (per-sample) |
| **Inference mode** | Uses running mean/var | Same as training |
| **Best for** | CNNs, large-batch training | Transformers, RNNs, small-batch |
| **Fails when** | Batch size = 1, variable-length sequences | Rarely |

```python
# PyTorch
nn.BatchNorm1d(num_features)  # input: [B, C] → normalize over B
nn.LayerNorm(normalized_shape)  # input: [B, *, C] → normalize over last dims

# Example in transformer block
x = x + self.attn(self.ln1(x))  # Pre-norm: LayerNorm before attention
x = x + self.ffn(self.ln2(x))   # Pre-norm: LayerNorm before FFN

# Keras
layers.BatchNormalization()  # auto-infers axis
layers.LayerNormalization()
```

---

## Dropout Patterns

**Mechanism**: During training, randomly zero out activations with probability p. Scale remaining by 1/(1-p) to maintain expected value (inverted dropout).

**Rules of thumb**:
- `p=0.1–0.3` for hidden layers in most networks
- `p=0.5` was classic (AlexNet era), now considered aggressive
- **Never** after BatchNorm (they conflict — BN assumes stable statistics)
- Place **after** activation, **before** next linear layer
- Disable during inference (`model.eval()` in PyTorch)

```python
# PyTorch — standard pattern
nn.Sequential(
    nn.Linear(512, 256),   # [B, 512] → [B, 256]
    nn.ReLU(),
    nn.Dropout(0.1),       # Zero 10% of activations during training
    nn.Linear(256, 128),   # [B, 256] → [B, 128]
)

# DropPath (Stochastic Depth) — for residual networks
class DropPath(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) > self.p  # [B,1,1]
        return x * keep / (1 - self.p)
```

---

## Learning Rate Scheduling

### Warmup + Cosine Decay (most common for transformers)

```python
# PyTorch — warmup + cosine
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=1000)  # 0.01*lr → lr
cosine = CosineAnnealingLR(optimizer, T_max=50000, eta_min=1e-5)   # lr → 1e-5
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[1000])
```

### OneCycleLR (super-convergence, best for CNNs)

```python
# Ramps lr up then down in one cycle, also anneals momentum
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-2, total_steps=num_batches * num_epochs,
    pct_start=0.3,  # 30% warmup
    anneal_strategy='cos',
    div_factor=25,   # initial_lr = max_lr / 25
    final_div_factor=1e4  # final_lr = initial_lr / 10000
)
# Call scheduler.step() after EVERY batch, not every epoch
```

### Keras Equivalents

```python
# Warmup + cosine (custom)
class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps):
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        warmup = self.peak_lr * step / self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps
        cosine = self.peak_lr * 0.5 * (1 + tf.cos(
            3.14159 * (step - self.warmup_steps) / decay_steps))
        return tf.where(step < self.warmup_steps, warmup, cosine)

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=WarmupCosineSchedule(1e-3, 1000, 50000), weight_decay=0.01)
```

**Choosing a schedule**:
- **Transformers/LLMs**: Warmup + cosine decay (standard)
- **CNNs (fast training)**: OneCycleLR (super-convergence)
- **Fine-tuning**: Constant or linear decay with small lr (1e-5 to 5e-5)

---

## Vanishing and Exploding Gradients

### Causes

| Problem | Cause | Symptom |
|---------|-------|---------|
| **Vanishing** | Sigmoid/tanh saturation (grad → 0), too many layers, small init | Early layers don't learn, loss plateaus |
| **Exploding** | Large weights, no normalization, high lr | NaN loss, wild parameter updates |

### Fixes

| Fix | Targets | Mechanism |
|-----|---------|-----------|
| **ReLU/GELU** | Vanishing | Gradient = 1 for positive inputs (no saturation) |
| **He initialization** | Vanishing/Exploding | Maintains activation variance = 1 |
| **Residual connections** | Vanishing | Gradient shortcut: ∂L/∂x = ∂L/∂(x+F(x)) = 1 + ∂F/∂x |
| **Gradient clipping** | Exploding | `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` |
| **BatchNorm/LayerNorm** | Both | Normalizes pre-activations to unit variance |
| **Lower learning rate** | Exploding | Smaller updates prevent divergence |
| **Gradient scaling (AMP)** | Vanishing in fp16 | Scale loss up before backward, unscale grads before step |

```python
# Gradient clipping in training loop
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# Residual connection pattern
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):  # x: [B, dim]
        return x + self.net(self.norm(x))  # skip connection preserves gradient flow
```

---

## Universal Approximation Theorem — Practical Implications

**Theorem**: A single hidden layer with sufficient width can approximate any continuous function on a compact set to arbitrary precision.

**What it means practically**:
1. **Existence, not construction** — guarantees a solution exists, says nothing about finding it via SGD
2. **Width vs depth tradeoff** — single wide layer needs exponentially many neurons for complex functions; depth gives exponential expressiveness gains (deep > wide for fixed parameter budget)
3. **Doesn't guarantee generalization** — memorization satisfies the theorem but fails on test data
4. **Motivates architecture search** — since MLPs CAN approximate anything, architecture choices (CNNs, Transformers) encode useful inductive biases that make learning efficient, not possible

**Practical rule**: For tabular data, 2–4 hidden layers with width 2–8× input features is usually sufficient. For complex functions (images, language), use architectures with domain-specific inductive biases.

---

## Complete Training Template

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Model
model = MLP(in_features=784, hidden=[512, 256], out_features=10, dropout=0.1)

# Optimizer + Scheduler
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = OneCycleLR(optimizer, max_lr=1e-2, total_steps=len(train_loader)*epochs)

# Training loop
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:            # x: [B, 784], y: [B]
        logits = model(x)                 # [B, 10]
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = sum(nn.functional.cross_entropy(model(x), y) for x, y in val_loader)
```

```python
# Keras equivalent
model = build_mlp(784, [512, 256], 10, dropout=0.1)
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-2),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.fit(train_ds, validation_data=val_ds, epochs=epochs,
          callbacks=[tf.keras.callbacks.ReduceLROnPlateau(patience=3)])
```

---

## Quick Reference: Shape Flow

```
Input:       [B, in_features]
Linear(h):   [B, in_features] @ [in_features, h] + [h] → [B, h]
Activation:  [B, h] → [B, h]  (element-wise)
BatchNorm:   [B, h] → [B, h]  (normalize over dim=0)
LayerNorm:   [B, h] → [B, h]  (normalize over dim=-1)
Dropout:     [B, h] → [B, h]  (zero mask, training only)
Output:      [B, out_features]
```

## When to Use

| ✅ Use ANN/MLP | ❌ Don't Use |
|---|---|
| Tabular/structured data with many features | Images (use CNN) |
| Simple classification or regression baseline | Sequences with temporal order (use RNN/Transformer) |
| Feature-rich inputs after manual engineering | When data has spatial/local structure |
| Small-medium datasets (<100K rows) | When you need interpretability (use linear/tree models) |
| Final layers of larger architectures | As standalone on raw unstructured data |

**Typical domains**: Fraud detection, credit scoring, recommendation feature layers, sensor fusion, click prediction.

**Decision rule**: If your data is a flat feature vector with no spatial/temporal structure, start with MLP. If XGBoost beats it, stay with XGBoost.

---

## References

- [PyTorch nn Module](https://pytorch.org/docs/stable/nn.html) — Complete neural network building blocks
- [Batch Normalization (Ioffe & Szegedy, 2015)](https://arxiv.org/abs/1502.03167) — Accelerating deep network training
