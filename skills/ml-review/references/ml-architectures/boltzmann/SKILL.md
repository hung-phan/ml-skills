---
name: boltzmann
description: Restricted Boltzmann machines (RBMs), energy-based models, contrastive divergence, and Deep Belief Network (DBN) pretraining. Use when building probabilistic generative models with principled energy functions or studying foundational unsupervised learning.
---

## Why This Exists

**Problem**: You need a probabilistic generative model that can learn the underlying distribution of data and generate samples, but you also want principled uncertainty quantification rooted in statistical physics rather than ad-hoc objectives.

**Key insight**: By defining an energy function over configurations and using Gibbs sampling, the network learns to assign low energy (high probability) to data-like configurations — the restricted bipartite structure makes inference tractable via conditional independence.

**Reach for this when**: You need unsupervised feature learning for pretraining (Deep Belief Networks), collaborative filtering, or a theoretically grounded energy-based model. Mostly historical now — prefer VAEs or diffusion models for generation, but RBMs remain useful for understanding energy-based learning.

> **⚠ Historical note (2024+)**: RBMs and DBNs were superseded ~2012–2014 by batch normalization, ReLU, dropout, and Adam — which made random initialization viable and rendered greedy pretraining unnecessary. For modern generative tasks use **VAEs** (tractable ELBO), **diffusion models** (state-of-the-art quality), or **contrastive learning** (self-supervised representations). The primary current value of this skill is educational: RBMs build intuition for partition functions, MCMC, and energy-based reasoning that transfers to modern score-based models and EBMs.


# Boltzmann Machines

## Energy-Based Models

Boltzmann Machines are stochastic generative models rooted in statistical mechanics. Every configuration of units has a scalar **energy**; lower energy = higher probability.

**Joint probability** (Gibbs distribution):

```
P(v, h) = exp(-E(v, h)) / Z
```

Where:
- `v` = visible units (observed data)
- `h` = hidden units (latent features)
- `E(v, h)` = energy function
- `Z = Σ exp(-E(v, h))` = partition function (intractable normalization constant)

The **free energy** marginalizes over hidden units:

```
F(v) = -log Σ_h exp(-E(v, h))
P(v) = exp(-F(v)) / Z
```

For binary RBMs, free energy has closed form: `F(v) = -b'v - Σ_j log(1 + exp(c_j + W_j v))`

## Restricted Boltzmann Machines (RBMs)

RBMs restrict connectivity: no intra-layer connections. Only visible↔hidden edges exist.

**Energy function:**

```
E(v, h) = -b'v - c'h - v'Wh
```

- `W` ∈ ℝ^(n_visible × n_hidden) = weight matrix
- `b` = visible biases, `c` = hidden biases

**Key property:** Given the restriction, units within a layer are conditionally independent:

```
P(h_j=1 | v) = σ(c_j + W_j · v)
P(v_i=1 | h) = σ(b_i + W_i · h)    # binary visible
P(v_i | h) = N(b_i + W_i · h, σ²)  # Gaussian visible
```

This enables efficient block Gibbs sampling.

## Contrastive Divergence (CD-k)

Exact gradient of log-likelihood requires sampling from the model distribution (intractable). Hinton (2002) proposed CD-k:

**Algorithm:**

```
1. v₀ = training sample
2. For k steps:
     h_k ~ P(h | v_k)        # sample hidden given visible
     v_{k+1} ~ P(v | h_k)    # reconstruct visible given hidden
3. ΔW = η (v₀ h₀' - v_k h_k')   # positive phase - negative phase
   Δb = η (v₀ - v_k)
   Δc = η (h₀ - h_k)
```

**Practical notes:**
- CD-1 works surprisingly well for most tasks
- CD-k (k=5-25) gives better generative models but slower training
- **PCD (Persistent CD):** maintain persistent Markov chains across updates instead of restarting from data — better gradient estimates, standard choice for serious training
- Use mean-field (probabilities) for `h₀` in positive phase, sample for negative phase

## Gibbs Sampling

Block Gibbs sampling alternates between sampling all hidden units (given visible) and all visible units (given hidden). The RBM's bipartite structure makes this exact and parallel:

```
h ~ P(h|v)  →  v ~ P(v|h)  →  h ~ P(h|v)  →  ...
```

Each full pass = one Gibbs step. After many steps, samples approximate the model distribution.

## Deep Belief Networks (DBNs)

Stack RBMs greedily (Hinton et al., 2006):

1. Train RBM₁ on raw data → learn h₁
2. Train RBM₂ on h₁ activations → learn h₂
3. Train RBM₃ on h₂ activations → learn h₃
4. Fine-tune entire stack with backprop (supervised) or wake-sleep (generative)

**Why it worked (historically):** Greedy layerwise pretraining provided good weight initialization before batch norm, ReLU, and Adam made random init viable. Each RBM layer provably improves a variational bound on log P(data).

## Practical Training Tips

| Aspect | Recommendation |
|--------|---------------|
| Learning rate | 0.01 for binary, 0.001 for Gaussian visible |
| Mini-batch | 10-100 samples |
| Momentum | Start 0.5, increase to 0.9 after 5 epochs |
| Weight decay | L2 = 0.0001 (prevents large weights) |
| Sparsity | Target hidden activation ~0.05, penalty on deviation |
| Weight init | N(0, 0.01) |
| Monitoring | Track reconstruction error AND free energy gap |
| Gaussian visible | Fix visible unit variance=1 initially, learn later |
| Number of hidden units | Start with n_hidden = n_visible, tune down |
| CD-k | Start with k=1, increase if generative quality matters |

**Common failure modes:**
- Weights grow too large → reconstructions become deterministic → dead units
- Too few hidden units → underfitting (high recon error)
- Too many hidden units → slow mixing, poor generalization

## Applications

### Collaborative Filtering
Netflix Prize era — RBMs for rating prediction (Salakhutdinov et al., 2007). Each movie = one visible unit with softmax over ratings. Conditional RBMs handle missing data naturally.

### Feature Learning / Pretraining
DBN pretraining was the breakthrough that launched deep learning (2006-2012). Superseded by:
- Batch normalization
- ReLU activations  
- Dropout
- Better optimizers (Adam)

### Other Applications
- **Topic modeling** — Replicated Softmax (word count modeling)
- **Dimensionality reduction** — nonlinear alternative to PCA
- **Anomaly detection** — high free energy = unlikely data
- **Physics simulations** — natural fit for lattice models

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, k=1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.k = k

    def sample_h(self, v):
        """P(h=1|v) and sampled h"""
        p_h = torch.sigmoid(F.linear(v, self.W.t(), self.h_bias))
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h):
        """P(v=1|h) and sampled v"""
        p_v = torch.sigmoid(F.linear(h, self.W, self.v_bias))
        return p_v, torch.bernoulli(p_v)

    def free_energy(self, v):
        vbias_term = v @ self.v_bias
        wx_b = F.linear(v, self.W.t(), self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(dim=1)
        return -vbias_term - hidden_term

    def forward(self, v):
        """CD-k: returns positive hidden probs and negative visible probs"""
        p_h0, h0 = self.sample_h(v)
        h_k = h0
        for _ in range(self.k):
            p_v_k, v_k = self.sample_v(h_k)
            p_h_k, h_k = self.sample_h(v_k)
        return v, p_h0, v_k, p_h_k

    def loss(self, v):
        """Free energy difference (proxy for log-likelihood)"""
        _, _, v_k, _ = self.forward(v)
        return self.free_energy(v).mean() - self.free_energy(v_k.detach()).mean()


# Training loop
rbm = RBM(784, 256, k=1)
optimizer = torch.optim.SGD(rbm.parameters(), lr=0.01, momentum=0.5, weight_decay=1e-4)

for epoch in range(50):
    for batch in dataloader:
        v = batch.view(-1, 784)  # flatten
        loss = rbm.loss(v)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Monitor: reconstruction error
    with torch.no_grad():
        v0, _, v_k, _ = rbm(v)
        recon_err = F.mse_loss(v0, v_k)
```

## Historical Context

**Timeline:**
- 1985: Boltzmann Machine (Hinton & Sejnowski) — fully connected, intractable
- 2002: CD training (Hinton) — made RBMs practical
- 2006: DBN pretraining (Hinton, Salakhutdinov) — launched deep learning revival
- 2007: RBM for Netflix Prize — practical large-scale application
- 2012-2014: Superseded by supervised techniques (dropout, batch norm, ReLU)
- 2014+: VAEs and GANs provide better generative models with tractable training

**Why superseded:**
1. **Training instability** — CD is biased; exact gradients intractable
2. **Slow mixing** — Gibbs chains mix poorly for complex distributions
3. **Limited expressiveness** — binary units constrain representation capacity
4. **Better alternatives** — VAEs (tractable ELBO), GANs (adversarial training), normalizing flows (exact likelihood)
5. **No backprop** — can't easily integrate into end-to-end differentiable systems

## When Still Useful

- **Small structured data** — tabular data with missing values, RBMs handle missingness naturally
- **Discrete latent variables** — when you specifically need binary stochastic features
- **Anomaly detection** — free energy provides a principled density score without expensive sampling
- **Pretraining for very small labeled sets** — when you have much unlabeled data and tiny labeled set, greedy pretraining can still help
- **Physics/Ising model connections** — energy-based reasoning maps directly to statistical physics problems
- **Educational value** — understanding RBMs builds intuition for partition functions, MCMC, and energy-based thinking that transfers to modern EBMs and diffusion models

---

## References

- [A Practical Guide to Training RBMs (Hinton, 2010)](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf) — Comprehensive hyperparameter and training guide
- [A Fast Learning Algorithm for Deep Belief Nets (Hinton et al., 2006)](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf) — Greedy layerwise pretraining
- [Training Products of Experts by Minimizing Contrastive Divergence (Hinton, 2002)](https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf) — CD algorithm
- Fischer & Igel, "Training Restricted Boltzmann Machines: An Introduction" (2014) — best practical tutorial

## When to Use

| ✅ Use Boltzmann/RBM | ❌ Don't Use |
|---|---|
| Generative modeling of binary/categorical data | Large-scale image generation (use GAN/diffusion) |
| Collaborative filtering (Netflix-style) | Discriminative tasks with labels available |
| Feature learning/pretraining (historical, DBN) | When modern alternatives (VAE, contrastive) exist |
| Energy-based modeling, physics-inspired systems | Real-time inference (slow sampling) |
| Understanding statistical mechanics of learning | Production systems needing fast training |

**Typical domains**: Recommender systems, statistical physics, drug interaction modeling, unsupervised feature extraction.

**Decision rule**: Mostly historical/educational now. Use VAE or contrastive learning for modern pretraining. RBMs remain relevant for energy-based reasoning and small binary-data generative tasks.

---


