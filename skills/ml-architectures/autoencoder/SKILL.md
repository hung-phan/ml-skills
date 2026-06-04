---
name: Autoencoder
description: Use when building autoencoders, VAEs, anomaly detection, dimensionality reduction, or generative models with latent spaces
---

## Why This Exists

**Problem**: High-dimensional data (images, genomes, sensor streams) is expensive to store and process, and often contains redundant information. You need to find a compact representation that preserves the essential structure while enabling anomaly detection, denoising, or generation.

**Key insight**: By forcing data through a bottleneck and requiring accurate reconstruction, the network must discover the most important latent factors — anything it can't reconstruct through the bottleneck is noise or redundancy.

**Reach for this when**: You need unsupervised dimensionality reduction with non-linear structure (PCA fails), anomaly detection (high reconstruction error = anomaly), data denoising, or a structured latent space for generation (VAE).


# Autoencoders

## Core Concept

An autoencoder learns a compressed representation by training encoder `f(x) → z` and decoder `g(z) → x̂` to minimize reconstruction error. The bottleneck forces the network to learn meaningful features.

```
Input x → [Encoder] → Latent z → [Decoder] → Reconstruction x̂
(N,D)      (N,d)        d << D       (N,D)
```

## Taxonomy

### Vanilla (Undercomplete) Autoencoder

Bottleneck dimension `d < D` forces compression. Simplest form.

```python
# PyTorch — shape annotations included
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 784, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),  # (B, 784) → (B, 128)
            nn.ReLU(),
            nn.Linear(128, latent_dim), # (B, 128) → (B, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), # (B, 32) → (B, 128)
            nn.ReLU(),
            nn.Linear(128, input_dim),  # (B, 128) → (B, 784)
            nn.Sigmoid(),
        )

    def forward(self, x):          # x: (B, 784)
        z = self.encoder(x)        # z: (B, 32)
        return self.decoder(z)     # x̂: (B, 784)
```

```python
# Keras equivalent
from tensorflow import keras
from keras import layers

encoder_input = keras.Input(shape=(784,))               # (B, 784)
x = layers.Dense(128, activation='relu')(encoder_input) # (B, 128)
z = layers.Dense(32, activation='relu')(x)              # (B, 32)

x = layers.Dense(128, activation='relu')(z)             # (B, 128)
decoder_output = layers.Dense(784, activation='sigmoid')(x)  # (B, 784)

autoencoder = keras.Model(encoder_input, decoder_output)
autoencoder.compile(optimizer='adam', loss='mse')
```

### Overcomplete Autoencoder

`d ≥ D`. Without regularization, learns identity. Useful only with constraints (sparsity, denoising, contractive penalty).

### Denoising Autoencoder (DAE)

Input corrupted with noise; network learns to reconstruct clean version. Prevents identity mapping and learns robust features.

```python
# PyTorch DAE training step
def train_step(model, x, noise_factor=0.3):
    x_noisy = x + noise_factor * torch.randn_like(x)  # (B, D)
    x_noisy = x_noisy.clamp(0, 1)
    x_hat = model(x_noisy)       # reconstruct from noisy input
    loss = F.mse_loss(x_hat, x)  # compare to CLEAN target
    return loss
```

Corruption types: Gaussian noise, masking (dropout), salt-and-pepper.

### Sparse Autoencoder

Adds sparsity penalty on latent activations. Encourages most neurons to be inactive, learning parts-based representations.

```python
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=256):  # overcomplete ok
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Sigmoid(),  # bounded [0,1] for KL sparsity
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def sparsity_loss(z, target_sparsity=0.05):
    """KL divergence sparsity penalty (Andrew Ng formulation)."""
    rho = target_sparsity
    rho_hat = z.mean(dim=0)  # average activation per neuron
    kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    return kl.sum()

# Alternative: simple L1 penalty
# sparsity_loss = lambda z: z.abs().mean()
```

### Contractive Autoencoder (CAE)

Penalizes the Frobenius norm of the encoder Jacobian, making the representation insensitive to small input perturbations.

```python
def contractive_loss(model, x, x_hat, lam=1e-3):
    mse = F.mse_loss(x_hat, x)
    # Jacobian penalty: ‖∂h/∂x‖²_F
    x.requires_grad_(True)
    h = model.encoder(x)                          # (B, d)
    jac = torch.autograd.functional.jacobian(model.encoder, x)  # expensive
    # Practical approximation for single hidden layer:
    W = model.encoder[0].weight                   # (d, D)
    h_deriv = h * (1 - h)                         # sigmoid derivative
    penalty = (h_deriv ** 2 @ (W ** 2)).sum()     # (B,)
    return mse + lam * penalty / x.size(0)
```

### Variational Autoencoder (VAE)

Generative model. Encoder outputs distribution parameters; decoder samples from latent space. Trained with ELBO (Evidence Lower Bound).

**Key insight — Reparameterization Trick:**
Instead of sampling `z ~ N(μ, σ²)` (non-differentiable), compute `z = μ + σ * ε` where `ε ~ N(0, I)`. Gradients flow through μ and σ.

```python
class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        # Encoder outputs μ and log(σ²)
        self.encoder = nn.Sequential(nn.Linear(input_dim, 400), nn.ReLU())
        self.fc_mu = nn.Linear(400, latent_dim)      # (B, 400) → (B, 20)
        self.fc_logvar = nn.Linear(400, latent_dim)  # (B, 400) → (B, 20)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400), nn.ReLU(),
            nn.Linear(400, input_dim), nn.Sigmoid(),
        )

    def encode(self, x):              # x: (B, 784)
        h = self.encoder(x)           # h: (B, 400)
        return self.fc_mu(h), self.fc_logvar(h)  # both (B, 20)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)           # σ: (B, 20)
        eps = torch.randn_like(std)             # ε ~ N(0,I): (B, 20)
        return mu + std * eps                   # z: (B, 20)

    def decode(self, z):              # z: (B, 20)
        return self.decoder(z)        # x̂: (B, 784)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(x, x_hat, mu, logvar):
    """ELBO = Reconstruction + KL divergence."""
    # Reconstruction: E[log p(x|z)]
    recon = F.binary_cross_entropy(x_hat, x, reduction='sum')
    # KL: D_KL(q(z|x) || p(z)) where p(z) = N(0,I)
    # Closed form: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl
```

```python
# Keras VAE
class Sampling(layers.Layer):
    def call(self, inputs):
        mu, logvar = inputs
        eps = keras.backend.random_normal(shape=keras.backend.shape(mu))
        return mu + keras.backend.exp(0.5 * logvar) * eps

encoder_input = keras.Input(shape=(784,))
x = layers.Dense(400, activation='relu')(encoder_input)
mu = layers.Dense(20)(x)
logvar = layers.Dense(20)(x)
z = Sampling()([mu, logvar])
encoder = keras.Model(encoder_input, [mu, logvar, z])

decoder_input = keras.Input(shape=(20,))
x = layers.Dense(400, activation='relu')(decoder_input)
decoder_output = layers.Dense(784, activation='sigmoid')(x)
decoder = keras.Model(decoder_input, decoder_output)
```

### β-VAE

Scales KL term by β > 1 to encourage disentangled representations at cost of reconstruction quality.

```python
def beta_vae_loss(x, x_hat, mu, logvar, beta=4.0):
    recon = F.binary_cross_entropy(x_hat, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl
```

- β = 1: standard VAE
- β > 1: stronger disentanglement, worse reconstruction
- β < 1: better reconstruction, entangled latent space

### Conditional VAE (CVAE)

Conditions on label `y` by concatenating it to both encoder input and decoder input.

```python
class CVAE(nn.Module):
    def __init__(self, input_dim=784, label_dim=10, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + label_dim, 400), nn.ReLU())
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 400), nn.ReLU(),
            nn.Linear(400, input_dim), nn.Sigmoid())

    def forward(self, x, y_onehot):
        # Encode: concat x and y
        h = self.encoder(torch.cat([x, y_onehot], dim=1))  # (B, 794) → (B, 400)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        # Decode: concat z and y
        x_hat = self.decoder(torch.cat([z, y_onehot], dim=1))  # (B, 30) → (B, 784)
        return x_hat, mu, logvar
```

### VQ-VAE (Vector Quantized VAE)

Discrete latent space. Encoder output is quantized to nearest codebook vector. No KL term — uses codebook + commitment loss.

```python
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)  # (K, D)
        self.commitment_cost = commitment_cost

    def forward(self, z_e):
        # z_e: (B, D, H, W) for conv encoder
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, z_e.size(1))  # (B*H*W, D)

        # Find nearest codebook vector
        distances = torch.cdist(z_e_flat, self.codebook.weight)  # (B*H*W, K)
        indices = distances.argmin(dim=1)                         # (B*H*W,)
        z_q_flat = self.codebook(indices)                         # (B*H*W, D)

        # Reshape back
        z_q = z_q_flat.view_as(z_e.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Losses
        codebook_loss = F.mse_loss(z_q.detach(), z_e)     # move codebook to encoder
        commitment_loss = F.mse_loss(z_q, z_e.detach())   # move encoder to codebook

        # Straight-through estimator: copy gradients from z_q to z_e
        z_q_st = z_e + (z_q - z_e).detach()

        loss = codebook_loss + self.commitment_cost * commitment_loss
        return z_q_st, loss, indices
```

VQ-VAE-2 adds hierarchical quantization (top + bottom codebooks) for high-fidelity image generation.

## Reconstruction Loss Choices

| Loss | When to Use | Formula |
|------|-------------|---------|
| **MSE** | Continuous data, Gaussian decoder assumption | `‖x - x̂‖²` |
| **BCE** | Binary/normalized [0,1] data (e.g., MNIST pixels) | `-Σ[x·log(x̂) + (1-x)·log(1-x̂)]` |
| **L1** | Sharper reconstructions, robust to outliers | `‖x - x̂‖₁` |
| **Perceptual** | Images — uses VGG feature distances | `‖φ(x) - φ(x̂)‖²` |

```python
# MSE — use with linear output activation
loss = F.mse_loss(x_hat, x)

# BCE — use with sigmoid output activation, data in [0,1]
loss = F.binary_cross_entropy(x_hat, x, reduction='sum')

# For VAE: BCE gives per-pixel log-likelihood, sum over pixels, mean over batch
loss = F.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(0)
```

## Latent Space Interpolation

```python
def interpolate(model, x1, x2, steps=10):
    """Linear interpolation in latent space (for VAE, use mu)."""
    with torch.no_grad():
        mu1, _ = model.encode(x1.unsqueeze(0))
        mu2, _ = model.encode(x2.unsqueeze(0))
        alphas = torch.linspace(0, 1, steps)
        interpolations = []
        for a in alphas:
            z = (1 - a) * mu1 + a * mu2          # linear interp in latent
            interpolations.append(model.decode(z))
    return torch.cat(interpolations, dim=0)       # (steps, D)

# Spherical interpolation (slerp) — better for high-dim normed spaces
def slerp(z1, z2, alpha):
    omega = torch.acos((z1 / z1.norm() * z2 / z2.norm()).sum())
    return (torch.sin((1 - alpha) * omega) * z1 + torch.sin(alpha * omega) * z2) / torch.sin(omega)
```

## Applications

### Anomaly Detection

Train on normal data only. High reconstruction error → anomaly.

```python
def detect_anomalies(model, data, threshold):
    """Reconstruction-error based anomaly detection."""
    with torch.no_grad():
        x_hat = model(data)
        errors = ((data - x_hat) ** 2).mean(dim=1)  # per-sample MSE
    return errors > threshold  # boolean mask of anomalies
```

For VAE: use reconstruction probability or ELBO as anomaly score (lower ELBO → more anomalous).

### Generation (VAE)

```python
# Sample from prior p(z) = N(0, I) and decode
z = torch.randn(num_samples, latent_dim)  # (N, 20)
samples = model.decode(z)                  # (N, 784)
```

### Representation Learning

Use encoder output as feature extractor for downstream tasks. Freeze encoder, train linear classifier on `z`.

### Denoising

Train DAE on domain-specific noise (sensor noise, compression artifacts, missing data imputation).

## Convolutional Autoencoder (Images)

```python
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # (B,1,28,28)→(B,32,14,14)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # →(B,64,7,7)
            nn.ReLU(),
            nn.Flatten(),                                # →(B, 3136)
            nn.Linear(3136, latent_dim),                 # →(B, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 3136),                 # →(B, 3136)
            nn.Unflatten(1, (64, 7, 7)),                 # →(B,64,7,7)
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # →(B,32,14,14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),   # →(B,1,28,28)
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
```

## Training Tips

- **Learning rate**: 1e-3 (Adam) is standard starting point
- **Latent dim**: Start small (2-20 for MNIST, 128-512 for complex images)
- **KL annealing** (VAE): Linearly increase β from 0→1 over warmup epochs to prevent posterior collapse
- **Batch norm**: Helps training stability in deeper autoencoders
- **Skip connections**: U-Net style for image autoencoders (better detail preservation)
- **Posterior collapse** (VAE): Decoder too powerful ignores z. Fix: reduce decoder capacity, free bits, or KL annealing

```python
# KL annealing schedule
def kl_weight(epoch, warmup_epochs=10):
    return min(1.0, epoch / warmup_epochs)
```

## Quick Reference

| Variant | Regularization | Latent | Generative? |
|---------|---------------|--------|-------------|
| Vanilla | Bottleneck only | Continuous | No |
| DAE | Input noise | Continuous | No |
| Sparse | L1/KL on activations | Continuous | No |
| Contractive | Jacobian penalty | Continuous | No |
| VAE | KL to N(0,I) | Continuous | Yes |
| β-VAE | Scaled KL (β>1) | Continuous | Yes (disentangled) |
| CVAE | KL + label conditioning | Continuous | Yes (conditional) |
| VQ-VAE | Codebook quantization | Discrete | Yes (with prior) |

## When to Use

| ✅ Use Autoencoder | ❌ Don't Use |
|---|---|
| Anomaly detection (reconstruction error as score) | When labeled data is plentiful (use supervised) |
| Dimensionality reduction (nonlinear, better than PCA) | Simple tabular data (PCA or UMAP suffices) |
| Denoising (corrupted → clean signal) | When you need exact generation control (use diffusion) |
| Generative modeling via VAE (smooth latent space) | Photorealistic generation (use GAN/diffusion) |
| Pretraining representations (MAE, BERT-style) | When downstream task is simple enough without pretraining |

**Typical domains**: Manufacturing defect detection, medical imaging anomalies, data compression, drug discovery (molecular VAE), representation learning.

**Decision rule**: Need to detect "what's abnormal" without labeled anomalies? Autoencoder. Need to generate realistic samples? VAE for diversity, GAN/diffusion for quality.

---

## References

- [VAE (Kingma & Welling, 2013)](https://arxiv.org/abs/1312.6114) — Variational Autoencoders
- [Ladder VAE (Sønderby et al., 2016)](https://arxiv.org/abs/1511.05644) — Hierarchical latent variables
