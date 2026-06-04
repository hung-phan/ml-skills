---
name: diffusion
description: Diffusion models for high-quality image, audio, and video generation — DDPM, DDIM, Stable Diffusion, LoRA fine-tuning, and classifier-free guidance. Use when generating high-fidelity content where mode coverage matters more than sampling speed, or fine-tuning Stable Diffusion with LoRA/DreamBooth.
---

## Why This Exists

**Problem**: Generating high-quality, diverse images (or audio/video/3D) requires sampling from incredibly complex distributions. GANs produce sharp images but suffer mode collapse and training instability; VAEs are stable but produce blurry outputs.

**Key insight**: Instead of learning to generate in one shot, learn to gradually denoise — the reverse of a simple noise-adding process. Each denoising step is a small, easy-to-learn Gaussian, and the chain of steps composes into arbitrarily complex distributions.

**Reach for this when**: You need state-of-the-art generation quality (text-to-image, inpainting, super-resolution, video), diversity matters more than inference speed, or you need controllable generation via guidance. Trade-off: slower inference than GANs (100-1000 steps vs 1).


# Diffusion Models

## Core Concept

Diffusion models learn to reverse a gradual noising process. Forward process adds Gaussian noise over T steps; reverse process (learned neural network) denoises step-by-step to generate samples.

## Mathematical Foundation

### Forward Process (q)
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) * x_{t-1}, β_t * I)
```
Closed-form sampling at any timestep t:
```
x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε,  ε ~ N(0, I)
```
Where `α_t = 1 - β_t`, `ᾱ_t = ∏_{s=1}^{t} α_s`

### Reverse Process (p_θ)
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```
Model predicts noise ε_θ(x_t, t), then:
```
μ_θ = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t))
```

## Noise Schedules

```python
import torch

# Linear schedule (DDPM original)
def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

# Cosine schedule (improved, less info loss at high t)
def cosine_schedule(T, s=0.008):
    steps = torch.arange(T + 1)
    f = torch.cos((steps / T + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, 0.0001, 0.9999)

# Scaled linear (Stable Diffusion default)
def scaled_linear_schedule(T, beta_start=0.00085, beta_end=0.012):
    return torch.linspace(beta_start**0.5, beta_end**0.5, T) ** 2
```

## DDPM — Denoising Diffusion Probabilistic Models

```python
import torch
import torch.nn as nn

class DDPM:
    def __init__(self, model, T=1000, schedule='linear'):
        self.model = model
        self.T = T
        betas = linear_schedule(T) if schedule == 'linear' else cosine_schedule(T)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """Forward process: sample x_t from x_0"""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def training_loss(self, x_0):
        """Simple loss: MSE between predicted and actual noise"""
        t = torch.randint(0, self.T, (x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t)
        return nn.functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, shape, device):
        """DDPM sampling (T steps, stochastic)"""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = self.model(x, t_batch)
            alpha = self.alphas[t]
            alpha_cum = self.alphas_cumprod[t]
            beta = self.betas[t]
            x = (1 / alpha.sqrt()) * (x - (beta / (1 - alpha_cum).sqrt()) * predicted_noise)
            if t > 0:
                x += beta.sqrt() * torch.randn_like(x)
        return x
```

## DDIM — Denoising Diffusion Implicit Models

Deterministic sampling; allows skipping steps (e.g., 1000 → 50 steps).

```python
@torch.no_grad()
def ddim_sample(model, shape, device, alphas_cumprod, timesteps, eta=0.0):
    """
    DDIM sampling. eta=0 is deterministic, eta=1 recovers DDPM.
    timesteps: subsequence of [0, T), e.g. [0, 20, 40, ..., 980]
    """
    x = torch.randn(shape, device=device)
    timesteps = list(reversed(timesteps))

    for i, t in enumerate(timesteps):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_batch)

        alpha_t = alphas_cumprod[t]
        alpha_prev = alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)

        pred_x0 = (x - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
        pred_x0 = pred_x0.clamp(-1, 1)

        sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
        dir_xt = (1 - alpha_prev - sigma**2).sqrt() * predicted_noise

        x = alpha_prev.sqrt() * pred_x0 + dir_xt
        if sigma > 0:
            x += sigma * torch.randn_like(x)
    return x
```

## Classifier-Free Guidance (CFG)

Train one model for both conditional and unconditional generation (randomly drop conditioning during training).

```python
def training_step(model, x_0, condition, p_uncond=0.1):
    mask = torch.rand(x_0.shape[0]) < p_uncond
    condition[mask] = null_token

    t = torch.randint(0, T, (x_0.shape[0],))
    noise = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, noise)
    pred = model(x_t, t, condition)
    return F.mse_loss(pred, noise)

@torch.no_grad()
def cfg_sample(model, x_t, t, condition, guidance_scale=7.5):
    """guidance_scale (w): 1.0=no guidance, 7.5=typical, 15+=very strong"""
    eps_uncond = model(x_t, t, null_token)
    eps_cond = model(x_t, t, condition)
    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    return eps
```

## Latent Diffusion (Stable Diffusion Architecture)

Runs diffusion in compressed latent space (4-8x spatial compression via VAE).

```
Architecture:
┌─────────────┐     ┌────────────┐     ┌─────────────┐
│ Text Encoder │────▶│  U-Net     │◀────│  Scheduler  │
│ (CLIP)       │     │ (Latent)   │     │ (DDIM/etc)  │
└─────────────┘     └─────┬──────┘     └─────────────┘
                          │
                    ┌─────▼──────┐
                    │ VAE Decoder │
                    └────────────┘
                          │
                      [Image]

Pipeline: text → CLIP encode → U-Net denoise in latent → VAE decode → pixel image
Latent shape: (B, 4, H/8, W/8) for 512x512 images
```

```python
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

image = pipe(
    "a photo of an astronaut riding a horse",
    num_inference_steps=50,
    guidance_scale=7.5,
).images[0]
```

### Using diffusers Components Directly

```python
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear")

with torch.no_grad():
    latent = vae.encode(image_tensor).latent_dist.sample() * 0.18215

tokens = tokenizer("a cat", padding="max_length", max_length=77, return_tensors="pt")
text_emb = text_encoder(tokens.input_ids).last_hidden_state  # (1, 77, 768)

noise_pred = unet(noisy_latent, timestep, encoder_hidden_states=text_emb).sample

with torch.no_grad():
    image = vae.decode(latent / 0.18215).sample
```

## LoRA Fine-Tuning

```python
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    target_modules=["to_q", "to_v", "to_k", "to_out.0"],
    lora_dropout=0.0,
)

unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()  # ~0.1% of total

optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
for batch in dataloader:
    latents = vae.encode(batch["images"]).latent_dist.sample() * 0.18215
    noise = torch.randn_like(latents)
    t = torch.randint(0, 1000, (latents.shape[0],), device=device)
    noisy = scheduler.add_noise(latents, noise, t)
    text_emb = text_encoder(batch["input_ids"]).last_hidden_state
    pred = unet(noisy, t, encoder_hidden_states=text_emb).sample
    loss = F.mse_loss(pred, noise)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

unet.save_pretrained("./lora_weights")
pipe.unet.load_attn_procs("./lora_weights")
```

## Training Template (Custom Dataset)

```python
import torch
from diffusers import UNet2DModel, DDPMScheduler

model = UNet2DModel(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 256, 256, 512),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
).to("cuda")

scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(100):
    for batch in dataloader:
        images = batch["images"].to("cuda")  # [-1, 1] normalized
        noise = torch.randn_like(images)
        t = torch.randint(0, 1000, (images.shape[0],), device="cuda")
        noisy = scheduler.add_noise(images, noise, t)

        pred = model(noisy, t).sample
        loss = torch.nn.functional.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

## When to Use: Diffusion vs GAN vs VAE

| Criterion | Diffusion | GAN | VAE |
|-----------|-----------|-----|-----|
| **Sample quality** | Best (FID) | Excellent | Good |
| **Mode coverage** | Full distribution | Mode collapse risk | Full but blurry |
| **Training stability** | Stable (simple MSE) | Unstable (adversarial) | Stable |
| **Sampling speed** | Slow (10-1000 steps) | Fast (1 forward pass) | Fast (1 decode) |
| **Controllability** | Excellent (CFG, inpainting) | Limited | Latent interpolation |
| **Likelihood** | Tractable (ELBO) | None | Tractable (ELBO) |

### Decision Guide

**Use Diffusion when:**
- Quality is paramount (art, photography, video)
- You need fine-grained control (text-to-image, inpainting, editing)
- Training stability matters more than inference speed
- You want to fine-tune with LoRA/DreamBooth on few images

**Use GAN when:**
- Real-time generation needed (game textures, video frames)
- Single-domain generation (faces, specific style)
- Inference latency is critical (<50ms)

**Use VAE when:**
- You need a structured latent space (interpolation, arithmetic)
- Anomaly detection (reconstruction error as signal)
- Part of a larger pipeline (VQ-VAE for discrete tokens)

## Key Hyperparameters

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| Timesteps (T) | 1000 | Training; sample with fewer via DDIM |
| Sampling steps | 20-50 (DDIM) | Quality/speed tradeoff |
| CFG scale | 5-15 | Higher=more prompt adherence, less diversity |
| Learning rate | 1e-5 to 1e-4 | Lower for fine-tuning |
| EMA decay | 0.9999 | Smooths model weights for better samples |
| Batch size | 16-256 | Larger=more stable gradients |
| LoRA rank | 4-64 | 4=minimal, 64=expressive |
| β schedule | scaled_linear | Default for latent diffusion |

## Common Pitfalls

1. **Not using EMA** — always maintain exponential moving average of weights for sampling
2. **Wrong normalization** — images must be [-1, 1] not [0, 1] for standard implementations
3. **Ignoring loss weighting** — uniform timestep sampling underweights important middle steps; use SNR weighting (`min_snr_gamma=5`)
4. **CFG scale too high** — causes saturation/artifacts; start at 7.5
5. **Insufficient steps** — DDPM needs ~1000, switch to DDIM/DPM-Solver for 20-50 steps
6. **LoRA overfitting** — use small rank (4-8), low LR (1e-4), few steps (500-1500)

## References

- [DDPM](https://arxiv.org/abs/2006.11239) — Ho et al. 2020
- [DDIM](https://arxiv.org/abs/2010.02502) — Song et al. 2020
- [Score SDE](https://arxiv.org/abs/2011.13456) — Song et al. 2021
- [Latent Diffusion](https://arxiv.org/abs/2112.10752) — Rombach et al. 2022
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) — Ho & Salimans 2022
- [huggingface/diffusers](https://github.com/huggingface/diffusers)
- [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
