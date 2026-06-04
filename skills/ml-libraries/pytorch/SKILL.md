---
name: pytorch
description: Use when user wants to build models with PyTorch, mentions nn.Module, tensors, autograd, DataLoader, DDP, or asks about custom training loops and GPU deep learning.
---

# PyTorch Skill Reference

## Why This Exists

1. **Problem solved**: Building and training neural networks with GPU acceleration and automatic differentiation. Without PyTorch, you'd need to manually derive gradients, implement backpropagation, manage GPU memory transfers, and write CUDA kernels — making deep learning research and production impractical.

2. **When to pick this over alternatives**: Choose PyTorch over Keras when you need full control over the training loop, custom autograd functions, or access to the HuggingFace/timm ecosystem. Choose PyTorch over JAX when you prefer imperative (eager) execution and don't need TPU-first workflows. Choose Keras when rapid prototyping of standard architectures matters more than flexibility.

3. **Mental model**: pytorch = dynamic computation graph with eager tensor operations. Every operation on a Tensor builds a graph node on-the-fly. Calling `.backward()` walks this graph in reverse to compute gradients. `nn.Module` is a tree of parameters + a `forward()` method. The training loop is explicit: forward → loss → zero_grad → backward → step.

## nn.Module Patterns

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()  # Always call super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(128, num_classes)
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

# Functional vs Module style:
# - Use nn.Module layers for anything with learnable parameters
# - Use torch.nn.functional (F) for stateless ops in forward() (e.g., F.relu, F.interpolate)
# - NEVER use F.dropout in forward without checking self.training
```

### Module Composition Patterns

```python
# ModuleList -- for dynamic/indexed access (NOT auto-called in forward)
class MultiHead(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(64, 10) for _ in range(n_heads)])

    def forward(self, x):
        return [head(x) for head in self.heads]

# ModuleDict -- for named conditional branches
class ConditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleDict({
            'text': nn.Linear(768, 256),
            'image': nn.Linear(2048, 256),
        })

    def forward(self, x, modality: str):
        return self.branches[modality](x)

# CRITICAL: Always use ModuleList/ModuleDict, never plain list/dict
# Plain containers won't register parameters or move to device
```

## Forward / Backward

```python
model = MyModel(784, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training step
model.train()  # Enable dropout/batchnorm training behavior
logits = model(inputs)          # forward pass
loss = criterion(logits, targets)
optimizer.zero_grad()           # Clear old gradients (or set_to_none=True for perf)
loss.backward()                 # Compute gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
optimizer.step()                # Update weights

# Eval step
model.eval()  # Disable dropout, batchnorm uses running stats
with torch.no_grad():
    logits = model(inputs)
```

### Gradient Accumulation

```python
accumulation_steps = 4
for i, (inputs, targets) in enumerate(dataloader):
    logits = model(inputs)
    loss = criterion(logits, targets) / accumulation_steps  # Scale loss
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## DataLoader / Dataset

```python
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

class CustomDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.samples = self._load(data_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def _load(self, path):
        # Load your data here
        ...

# DataLoader best practices
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,              # Only for training
    num_workers=4,             # Parallel data loading (0 for debugging)
    pin_memory=True,           # Faster CPU->GPU transfer
    drop_last=True,            # Avoid small last batch (important for BatchNorm)
    persistent_workers=True,   # Keep workers alive between epochs (num_workers > 0)
    prefetch_factor=2,         # Batches prefetched per worker
)

# IterableDataset for streaming/large data
class StreamDataset(torch.utils.data.IterableDataset):
    def __init__(self, url):
        self.url = url

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # Shard data across workers if num_workers > 0
        for record in self._stream(self.url, worker_info):
            yield record

# Weighted sampling for imbalanced classes
class_counts = [1000, 100, 50]  # samples per class
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[targets]  # weight per sample
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))
loader = DataLoader(dataset, batch_size=64, sampler=sampler)  # Don't use shuffle with sampler
```

### Custom Collate

```python
def collate_variable_length(batch):
    """Pad sequences to max length in batch."""
    sequences, labels = zip(*batch)
    lengths = [len(s) for s in sequences]
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels), torch.tensor(lengths)

loader = DataLoader(dataset, collate_fn=collate_variable_length)
```

## Optimizers & Scheduling

```python
# Adam -- good default, fast convergence
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# AdamW -- decoupled weight decay (preferred for transformers)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# SGD with momentum -- often best final accuracy for CNNs
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Per-parameter groups (different LR for backbone vs head)
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.head.parameters(), 'lr': 1e-3},
], weight_decay=0.01)

# --- Schedulers ---
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau,
    LinearLR, SequentialLR, CosineAnnealingWarmRestarts
)

# Cosine annealing (smooth decay)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# OneCycleLR (warmup + cosine, best for super-convergence)
scheduler = OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=len(loader) * num_epochs,
    pct_start=0.1, anneal_strategy='cos'
)
# Call scheduler.step() AFTER optimizer.step() each BATCH (not epoch)

# Warmup + cosine (compose schedulers)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=500)
cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - 5)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[500])

# ReduceLROnPlateau (adaptive, based on val loss)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
# Call: scheduler.step(val_loss)

# RULE: CosineAnnealing/OneCycleLR -> step per batch
#        ReduceLROnPlateau -> step per epoch with metric
```

### Optimizer Selection Guide

| Scenario | Optimizer | LR Range |
|----------|-----------|----------|
| Transformers / LLMs | AdamW | 1e-5 to 5e-4 |
| CNNs (ResNet, etc.) | SGD+momentum or AdamW | 0.01-0.1 (SGD), 1e-4 (AdamW) |
| Fine-tuning pretrained | AdamW, low LR backbone | 1e-5 backbone, 1e-3 head |
| GANs | Adam (β1=0.0, β2=0.9) | 1e-4 to 2e-4 |
| Small datasets | AdamW + high weight_decay | 1e-3, wd=0.1 |

## Loss Functions Selection Guide

```python
# Classification (mutually exclusive classes)
nn.CrossEntropyLoss()          # Expects raw logits, integer targets
nn.CrossEntropyLoss(weight=class_weights)  # Imbalanced classes
nn.CrossEntropyLoss(label_smoothing=0.1)   # Regularization

# Multi-label classification
nn.BCEWithLogitsLoss()         # Each class independent, expects raw logits
nn.BCEWithLogitsLoss(pos_weight=pos_weights)  # Imbalanced multi-label

# Regression
nn.MSELoss()                   # L2, sensitive to outliers
nn.L1Loss()                    # MAE, robust to outliers
nn.SmoothL1Loss(beta=1.0)     # Huber loss, best of both
nn.HuberLoss(delta=1.0)       # Same as SmoothL1Loss

# Ranking / Contrastive
nn.TripletMarginLoss(margin=1.0)
nn.CosineEmbeddingLoss()
# InfoNCE / NT-Xent for self-supervised (implement manually or use a library)

# Segmentation
nn.CrossEntropyLoss(ignore_index=255)  # Ignore unlabeled pixels
# Dice loss (implement manually for better boundary handling)

# RULE: Always pass raw logits to CrossEntropyLoss/BCEWithLogitsLoss
#        They apply softmax/sigmoid internally (numerically stable)
#        NEVER apply softmax before CrossEntropyLoss
```

## Device Management

```python
# Automatic device selection
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

device = get_device()

# Move model and data
model = model.to(device)
inputs = inputs.to(device, non_blocking=True)  # non_blocking with pin_memory

# Multi-GPU: specify device index
device = torch.device('cuda:0')

# COMMON ERROR: Tensors on different devices
# Fix: ensure all tensors and model on same device before operations
# model.device doesn't exist -- use next(model.parameters()).device

# MPS (Apple Silicon) caveats:
# - Some ops fall back to CPU silently (slower)
# - No AMP support (use float32)
# - Some ops unsupported: torch.cdist, some indexing ops
```

## Autograd

```python
# torch.no_grad() -- disables gradient computation (saves memory)
# Use for: inference, metric computation, manual weight updates
with torch.no_grad():
    predictions = model(test_inputs)
    accuracy = (predictions.argmax(1) == targets).float().mean()

# torch.inference_mode() -- stricter, faster than no_grad (PyTorch 1.9+)
# Use for: pure inference (no autograd ops at all)
with torch.inference_mode():
    output = model(inputs)
# CANNOT use output in any gradient computation after this

# Detach -- stop gradient flow through a tensor
frozen_features = encoder(x).detach()  # Treat as constant
logits = classifier(frozen_features)

# Selective gradient freezing
for param in model.backbone.parameters():
    param.requires_grad = False
# Only classifier params will be updated

# Re-enable later:
for param in model.backbone.parameters():
    param.requires_grad = True

# Custom autograd function
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None
```

## Distributed Training (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # CRITICAL: shuffle differently each epoch
        for inputs, targets in loader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            loss = criterion(model(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    cleanup()

# Launch with torchrun (preferred):
# torchrun --nproc_per_node=4 train.py

# Or programmatic:
import torch.multiprocessing as mp
mp.spawn(train, args=(world_size,), nprocs=world_size)

# Access underlying model (DDP wraps it):
model.module.save_pretrained(...)  # not model.save_pretrained()
```

### DDP Key Rules
- Use `DistributedSampler` and call `sampler.set_epoch(epoch)`
- All ranks must call the same collective ops (no conditional forward paths per rank)
- Save/load checkpoints on rank 0 only, then broadcast
- Use `model.module` to access the original model

## Mixed Precision (torch.amp)

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')

for inputs, targets in loader:
    inputs, targets = inputs.to(device), targets.to(device)

    with autocast('cuda'):  # or 'cpu' for bfloat16 on CPU
        logits = model(inputs)
        loss = criterion(logits, targets)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    # Unscale before clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

# BFloat16 (A100+, no scaler needed):
with autocast('cuda', dtype=torch.bfloat16):
    output = model(inputs)

# RULES:
# - Don't cast model manually; autocast handles it
# - Loss computation should be inside autocast
# - Scaler is only needed for float16, NOT bfloat16
# - Some ops stay float32 (loss functions, softmax, layer norm)
```

## Checkpointing

```python
# Save checkpoint (include everything needed to resume)
def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'scaler_state_dict': scaler.state_dict(),  # If using AMP
    }, path)

# Load checkpoint
def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

# weights_only=True (PyTorch 2.0+) -- SECURITY: prevents arbitrary code execution
# map_location='cpu' -- load to CPU first, then move to device (avoids GPU OOM)

# Activation checkpointing (trade compute for memory)
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    def forward(self, x):
        # Recompute activations in backward (saves memory, costs ~30% compute)
        x = checkpoint(self.block1, x, use_reentrant=False)
        x = checkpoint(self.block2, x, use_reentrant=False)
        return self.head(x)
```

## Custom Layers

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)

# Lazy modules (infer input size from first forward)
class LazyClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.LazyLinear(num_classes)  # in_features inferred

    def forward(self, x):
        return self.fc(x.flatten(1))
```

## Hooks

```python
# Forward hook -- inspect/modify outputs
activations = {}
def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

handle = model.layer3.register_forward_hook(save_activation('layer3'))
output = model(x)
print(activations['layer3'].shape)
handle.remove()  # Always remove when done

# Backward hook -- inspect/modify gradients
def grad_hook(module, grad_input, grad_output):
    print(f"Grad norm: {grad_output[0].norm():.4f}")

handle = model.fc.register_full_backward_hook(grad_hook)

# Tensor hook -- modify gradient of a specific tensor
def clip_grad(grad):
    return grad.clamp(-1, 1)

x = torch.randn(3, requires_grad=True)
x.register_hook(clip_grad)

# Common use: feature extraction from pretrained model
class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.features = {}
        for name, module in model.named_modules():
            if name in layers:
                module.register_forward_hook(
                    lambda m, i, o, name=name: self.features.update({name: o})
                )

    def forward(self, x):
        self.features.clear()
        _ = self.model(x)
        return self.features
```

## Torchvision Transforms v2

```python
from torchvision.transforms import v2

# Training transforms
train_transform = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.08, 1.0)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    v2.RandAugment(num_ops=2, magnitude=9),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),  # Replaces ToTensor()
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.RandomErasing(p=0.25),
])

# Eval transforms
eval_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# v2 advantages over v1:
# - Works on PIL, Tensor, AND Datapoints (BoundingBox, Mask, Video)
# - Joint transforms (same random crop for image + mask in segmentation)
# - MixUp / CutMix built-in:
mixup = v2.MixUp(num_classes=1000)
cutmix = v2.CutMix(num_classes=1000)
mixup_cutmix = v2.RandomChoice([mixup, cutmix])
# Apply after DataLoader (on batch):
for images, targets in loader:
    images, targets = mixup_cutmix(images, targets)
```

## Debugging

### Anomaly Detection

```python
# Detects the exact operation that produced NaN/Inf
torch.autograd.set_detect_anomaly(True)
# Use in training loop to find where NaN gradients originate
# WARNING: Very slow, use only for debugging

# Context manager version (scoped)
with torch.autograd.detect_anomaly():
    output = model(input)
    loss = criterion(output, target)
    loss.backward()  # Will show traceback to the op that created NaN
```

### Gradient Checking

```python
from torch.autograd import gradcheck, gradgradcheck

# Verify custom autograd function correctness
input = torch.randn(3, 4, requires_grad=True, dtype=torch.float64)
# Must use float64 for numerical gradient checking
assert gradcheck(my_custom_function, input, eps=1e-6, atol=1e-4)
```

### Shape Debugging

```python
# Print shapes through a model
from torchinfo import summary  # pip install torchinfo
summary(model, input_size=(1, 3, 224, 224), col_names=["output_size", "num_params"])

# Manual shape tracing with hooks
def shape_hook(name):
    def hook(module, input, output):
        in_shape = [i.shape for i in input if isinstance(i, torch.Tensor)]
        out_shape = output.shape if isinstance(output, torch.Tensor) else "non-tensor"
        print(f"{name}: {in_shape} -> {out_shape}")
    return hook

for name, module in model.named_modules():
    module.register_forward_hook(shape_hook(name))
```

### Profiling

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    with record_function("model_forward"):
        output = model(inputs)
    with record_function("loss_backward"):
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
# Export for Chrome trace viewer: prof.export_chrome_trace("trace.json")
```

## Common Errors & Fixes

### Shape Mismatches

```python
# ERROR: RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x512 and 256x10)
# FIX: Check tensor shapes at each step. Print x.shape in forward().
# Common causes:
#   - Forgot to flatten CNN output before linear layer
#   - Wrong in_features after pooling
#   - Batch dimension confusion (missing unsqueeze/squeeze)

# Pattern: adaptive pooling before classifier ensures fixed size
self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Always outputs (B, C, 1, 1)
x = self.pool(x).flatten(1)  # (B, C)
```

### In-place Operations Breaking Autograd

```python
# ERROR: RuntimeError: one of the variables needed for gradient computation
#        has been modified by an inplace operation

# BAD -- in-place modification
x += residual        # In-place add
x.relu_()           # In-place ReLU
x[:, 0] = 0        # In-place indexing assignment

# GOOD -- out-of-place
x = x + residual
x = torch.relu(x)   # or F.relu(x)
x = x.clone()
x[:, 0] = 0        # Clone first, then assign
```

### CUDA Out of Memory

```python
# Strategies (in order of preference):
# 1. Reduce batch size
# 2. Use mixed precision (amp)
# 3. Use gradient accumulation
# 4. Use activation checkpointing
# 5. Use gradient_checkpointing_enable() for HuggingFace models

# Find memory leak:
torch.cuda.memory_summary()  # Detailed allocation stats
torch.cuda.empty_cache()     # Release cached memory (doesn't fix leaks)

# Common leak: storing tensors with grad history
losses.append(loss.item())      # GOOD: .item() detaches
losses.append(loss)             # BAD: keeps entire computation graph alive
```

### DataLoader Worker Errors

```python
# ERROR: RuntimeError: DataLoader worker is killed by signal: Killed (OOM)
# FIX: Reduce num_workers or batch_size

# ERROR: Broken pipe / EOF when using num_workers > 0
# FIX: Wrap main code in if __name__ == '__main__': (Windows/macOS)
# Or set multiprocessing start method:
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)  # Before any CUDA calls

# Debugging: set num_workers=0 to get full tracebacks
```

### State Dict Mismatches

```python
# ERROR: Missing key(s) / Unexpected key(s) in state_dict
# Cause: model architecture changed, or DDP prefix ('module.')

# Fix missing 'module.' prefix (saved with DDP, loading without):
state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.load_state_dict(state_dict)

# Partial loading (transfer learning):
model.load_state_dict(checkpoint, strict=False)  # Ignores missing/unexpected keys
```

### Reproducibility

```python
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable for reproducibility
    # Note: deterministic=True may slow training significantly

# For DataLoader reproducibility:
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    import numpy as np, random
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)
loader = DataLoader(dataset, worker_init_fn=seed_worker, generator=g)
```

## torch.compile (PyTorch 2.0+)

```python
# JIT compile for 10-40% speedup (no code changes needed)
model = torch.compile(model)  # Default mode='default'

# Modes:
model = torch.compile(model, mode='reduce-overhead')  # Best for small models
model = torch.compile(model, mode='max-autotune')     # Slower compile, fastest runtime

# CAVEATS:
# - First forward is slow (compilation)
# - Dynamic shapes cause recompilation (use dynamic=True if shapes vary)
# - Not all ops supported (graph breaks)
# - Use TORCH_LOGS="+dynamo" for debugging compilation issues

# Works with DDP and AMP:
model = DDP(torch.compile(model), device_ids=[rank])
```

## When to Use

| ✅ Use PyTorch | ❌ Don't Use |
|---|---|
| Research, custom architectures | Quick prototype (Keras is faster to write) |
| Full control over training loop | TPU-first workflows (use JAX) |
| Dynamic computation graphs | When team only knows TensorFlow |
| Ecosystem (HuggingFace, timm, torchvision) | Mobile deployment (use TFLite or ONNX) |
| Distributed training (DDP, FSDP) | Simple inference-only (use ONNX Runtime) |

**Decision rule**: Default for ML research and production. Use Keras for rapid prototyping, JAX for TPU/XLA workloads.

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch GitHub](https://github.com/pytorch/pytorch)