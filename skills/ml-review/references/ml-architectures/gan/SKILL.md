---
name: gan
description: Generative adversarial networks for image and data synthesis — DCGAN, WGAN-GP, StyleGAN, Pix2Pix, CycleGAN, and adversarial training patterns. Use when building one-shot generators (faster than diffusion), doing image-to-image translation, or generating synthetic data for augmentation.
---

## Why This Exists

**Problem**: You need to generate realistic data (images, audio, tabular) in a single fast forward pass — diffusion models produce higher quality but require hundreds of iterative steps, making them impractical for real-time synthesis.

**Key insight**: Pit two networks against each other — the generator tries to fool the discriminator, the discriminator tries to catch fakes. This adversarial game drives the generator toward producing samples indistinguishable from real data without needing an explicit density model.

**Reach for this when**: You need real-time generation (single forward pass, ~ms), image-to-image translation (Pix2Pix/CycleGAN), super-resolution, or style transfer where inference speed matters. If quality is paramount and latency is acceptable, prefer diffusion models instead.


# GANs (Generative Adversarial Networks)

> Skill for Claude Code: PyTorch GAN implementation patterns, architecture selection, and training techniques.

## Core Concept

Two networks play a minimax game:
- **Generator (G)**: Maps latent noise z ~ N(0,I) → fake data
- **Discriminator (D)**: Classifies real vs fake

Objective: `min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]`

---

## Architecture Variants

| Variant | Key Innovation | Use Case | Paired Data? |
|---------|---------------|----------|--------------|
| **Vanilla GAN** | MLP G + D | Simple distributions, tabular | N/A |
| **DCGAN** | ConvTranspose2d G, strided Conv D, BatchNorm | Image generation baseline | N/A |
| **WGAN-GP** | Wasserstein distance + gradient penalty | Stable training, no mode collapse | N/A |
| **Conditional GAN** | Class label concat to z and input | Class-conditional generation | N/A |
| **Pix2Pix** | U-Net G + PatchGAN D + L1 loss | Paired image translation | Yes |
| **CycleGAN** | Cycle consistency loss, 2 G + 2 D | Unpaired domain transfer | No |
| **StyleGAN** | Mapping network + AdaIN + progressive | High-res faces, controllable style | N/A |
| **ProGAN** | Progressive resolution growing 4→8→...→1024 | High-res from scratch | N/A |

---

## GAN vs VAE vs Diffusion — When to Use What

| Criterion | GAN | VAE | Diffusion |
|-----------|-----|-----|-----------|
| **Sample quality** | Sharp, high-fidelity | Blurry (mean-seeking) | Highest quality |
| **Diversity** | Risk of mode collapse | High (covers modes) | High |
| **Training stability** | Fragile, requires tuning | Stable (ELBO) | Stable |
| **Speed (inference)** | Single forward pass (~ms) | Single forward pass | Slow (100-1000 steps) |
| **Latent space** | Entangled (except StyleGAN) | Structured, smooth | Implicit |
| **Likelihood** | No tractable density | Tractable ELBO | Tractable |
| **Best for** | Real-time synthesis, super-resolution, style transfer | Representation learning, anomaly detection | Quality-critical image/video, text-to-image |

**Decision heuristic:**
- Need real-time inference? → GAN
- Need structured latent space for interpolation? → VAE or StyleGAN
- Need highest quality regardless of speed? → Diffusion
- Need unpaired domain transfer? → CycleGAN
- Small dataset + high fidelity? → GAN with data augmentation (DiffAugment)

---

## PyTorch Implementation Patterns

### 1. DCGAN Generator

```python
import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    """Fractionally-strided convolutions: z (nz,1,1) → image (nc,64,64)"""
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)
```

### 2. DCGAN Discriminator

```python
class DCGANDiscriminator(nn.Module):
    """Strided convolutions: image (nc,64,64) → scalar"""
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)
```

### 3. Weight Initialization (Critical for GANs)

```python
def weights_init(m):
    """DCGAN paper: N(0, 0.02) for conv/deconv, N(1, 0.02) for BN"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator.apply(weights_init)
discriminator.apply(weights_init)
```

### 4. Standard Training Loop (BCE Loss)

```python
criterion = nn.BCELoss()
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(num_epochs):
    for real_batch, _ in dataloader:
        batch_size = real_batch.size(0)
        real = real_batch.to(device)

        # --- Train Discriminator ---
        opt_d.zero_grad()
        output_real = D(real)
        loss_d_real = criterion(output_real, torch.ones(batch_size, device=device))
        z = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = G(z)
        output_fake = D(fake.detach())
        loss_d_fake = criterion(output_fake, torch.zeros(batch_size, device=device))
        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        opt_d.step()

        # --- Train Generator ---
        opt_g.zero_grad()
        output = D(fake)
        loss_g = criterion(output, torch.ones(batch_size, device=device))
        loss_g.backward()
        opt_g.step()
```

---

## Loss Functions

### WGAN-GP (Gradient Penalty)
```python
def gradient_penalty(D, real, fake, device, lambda_gp=10):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolated = D(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True, retain_graph=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * gp

loss_d = -(torch.mean(D(real)) - torch.mean(D(fake.detach()))) + gradient_penalty(D, real, fake, device)
```

### Hinge Loss (used in BigGAN, SAGAN)
```python
loss_d = F.relu(1.0 - D(real)).mean() + F.relu(1.0 + D(fake.detach())).mean()
loss_g = -D(fake).mean()
```

---

## Spectral Normalization

```python
class SNDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, 1, 4, 1, 0)),
        )

    def forward(self, x):
        return self.main(x).view(-1)
```

---

## Pix2Pix (Paired Image-to-Image)

Key components:
- **Generator**: U-Net with skip connections
- **Discriminator**: PatchGAN (classifies 70×70 patches)
- **Loss**: Adversarial + L1 reconstruction (λ=100)

```python
# Pix2Pix loss
loss_g = criterion_gan(D(torch.cat([input_img, fake], 1)), ones) + 100 * F.l1_loss(fake, target)
```

---

## CycleGAN (Unpaired Image-to-Image)

Two generators (G_AB, G_BA) + two discriminators (D_A, D_B).

```python
loss_identity = F.l1_loss(G_BA(real_A), real_A) + F.l1_loss(G_AB(real_B), real_B)  # λ_id=5
loss_gan = criterion(D_B(fake_B), ones) + criterion(D_A(fake_A), ones)
loss_cycle = F.l1_loss(G_BA(fake_B), real_A) + F.l1_loss(G_AB(fake_A), real_B)  # λ_cyc=10
loss_G = loss_gan + 10.0 * loss_cycle + 5.0 * loss_identity
```

---

## Mode Collapse Prevention

| Technique | How It Works |
|-----------|-------------|
| **WGAN-GP** | Wasserstein distance + gradient penalty |
| **Spectral normalization** | Normalize D weights by spectral norm |
| **Label smoothing** | Real labels = 0.9 instead of 1.0 |
| **Feature matching** | G minimizes difference in D's intermediate features |
| **DiffAugment** | Apply augmentation to real AND fake in D |
| **Two-timescale (TTUR)** | Higher lr for D than G |
| **R1 regularization** | Penalize D gradient norm on real data only |

```python
# R1 regularization (StyleGAN)
def r1_penalty(D, real, gamma=10.0):
    real.requires_grad_(True)
    d_out = D(real)
    grads = torch.autograd.grad(d_out.sum(), real, create_graph=True)[0]
    return (gamma / 2) * grads.view(grads.size(0), -1).pow(2).sum(1).mean()
```

---

## Training Tips & Hyperparameters

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| Optimizer | Adam β=(0.0, 0.9) for WGAN-GP; (0.5, 0.999) for DCGAN | β1=0 helps WGAN stability |
| LR (D) | 1e-4 to 4e-4 | TTUR: D lr = 2-4× G lr |
| LR (G) | 1e-4 to 2e-4 | |
| Batch size | 32-64 (small), 256+ (StyleGAN) | Larger = more stable |
| D steps per G step | 1 (DCGAN/SN), 5 (WGAN-GP) | More D steps for Wasserstein |
| Latent dim (nz) | 100-512 | 512 for StyleGAN |
| LeakyReLU slope | 0.2 | Standard for D |
| Tanh output (G) | [-1, 1] range | Match input normalization |

---

## Quick Architecture Selection

```
Need: Generate images from noise
├── Resolution ≤ 64×64 → DCGAN + spectral norm
├── Resolution 256-1024 → StyleGAN2/3
├── Controllable attributes → Conditional GAN or StyleGAN (w-space)
└── Need training stability → WGAN-GP or hinge + SN

Need: Image-to-image translation
├── Have paired data → Pix2Pix (U-Net + PatchGAN + L1)
├── Unpaired domains → CycleGAN
└── Multi-domain → StarGAN

Need: Super-resolution
├── Perceptual quality → SRGAN / ESRGAN
└── Real-time → lightweight GAN with distillation
```

---

## References

- [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661) — Original GAN paper
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) — Stable training with Earth Mover distance
