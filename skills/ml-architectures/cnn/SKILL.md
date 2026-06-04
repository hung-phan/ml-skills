---
name: CNN
description: Use when building convolutional networks, image classification, detection, segmentation, or transfer learning
---

## Why This Exists

**Problem**: Images have spatial structure — neighboring pixels are correlated, objects can appear anywhere in the frame, and a cat in the top-left should be recognized the same as one in the bottom-right. Fully-connected networks ignore this structure and scale impossibly (a 224×224 RGB image = 150K input neurons).

**Key insight**: Weight sharing (same filter applied everywhere) + local receptive fields (only look at patches) + hierarchical composition (edges → textures → parts → objects) gives translation equivariance and reduces parameters by orders of magnitude.

**Reach for this when**: Your input has spatial/grid structure (images, spectrograms, 2D signals), you need translation invariance, or you're doing classification/detection/segmentation on visual data. CNNs are the default before considering ViT (which needs more data).


# CNN Architectures Skill

Reference for convolutional neural network design, implementation patterns, and PyTorch usage.

---

## Convolution Arithmetic

### Output Size Formula

```
O = floor((I - K + 2P) / S) + 1
```

| Symbol | Meaning |
|--------|---------|
| I | Input spatial dimension |
| K | Kernel size |
| P | Padding |
| S | Stride |

**Transposed convolution (deconv):**
```
O = (I - 1) * S - 2P + K + output_padding
```

**Dilated convolution effective kernel:**
```
K_eff = K + (K - 1) * (dilation - 1)
O = floor((I - K_eff + 2P) / S) + 1
```

### Quick Reference

| Input | Kernel | Stride | Padding | Output |
|-------|--------|--------|---------|--------|
| 32 | 3 | 1 | 1 | 32 (same) |
| 32 | 3 | 2 | 1 | 16 (halve) |
| 32 | 5 | 1 | 2 | 32 (same) |
| 32 | 7 | 2 | 3 | 16 (halve) |

**"Same" padding formula:** `P = (K - 1) // 2` (only exact for odd K, stride=1)

---

## Classic Architectures

### LeNet-5 (1998)
```python
# Input: (1, 32, 32)
nn.Sequential(
    nn.Conv2d(1, 6, 5),          # -> (6, 28, 28)
    nn.Tanh(),
    nn.AvgPool2d(2, 2),          # -> (6, 14, 14)
    nn.Conv2d(6, 16, 5),         # -> (16, 10, 10)
    nn.Tanh(),
    nn.AvgPool2d(2, 2),          # -> (16, 5, 5)
    nn.Flatten(),                 # -> (400,)
    nn.Linear(400, 120),
    nn.Linear(120, 84),
    nn.Linear(84, 10),
)
```

### AlexNet (2012)
Key innovations: ReLU, dropout, GPU parallelism, LRN.
```python
# Input: (3, 224, 224)
nn.Sequential(
    nn.Conv2d(3, 64, 11, stride=4, padding=2),   # -> (64, 55, 55)
    nn.ReLU(inplace=True),
    nn.MaxPool2d(3, stride=2),                     # -> (64, 27, 27)
    nn.Conv2d(64, 192, 5, padding=2),              # -> (192, 27, 27)
    nn.ReLU(inplace=True),
    nn.MaxPool2d(3, stride=2),                     # -> (192, 13, 13)
    nn.Conv2d(192, 384, 3, padding=1),             # -> (384, 13, 13)
    nn.ReLU(inplace=True),
    nn.Conv2d(384, 256, 3, padding=1),             # -> (256, 13, 13)
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, 3, padding=1),             # -> (256, 13, 13)
    nn.ReLU(inplace=True),
    nn.MaxPool2d(3, stride=2),                     # -> (256, 6, 6)
)
# Classifier: Flatten -> 4096 -> 4096 -> num_classes
```

### VGG-16 (2014)
Key insight: stack 3×3 convs (two 3×3 = one 5×5 receptive field, fewer params).
```python
# Pattern: Conv blocks with increasing channels, MaxPool to halve
# Input: (3, 224, 224)
# Block 1: 2x Conv(3,64,3,p=1) + MaxPool  -> (64, 112, 112)
# Block 2: 2x Conv(64,128,3,p=1) + MaxPool -> (128, 56, 56)
# Block 3: 3x Conv(128,256,3,p=1) + MaxPool -> (256, 28, 28)
# Block 4: 3x Conv(256,512,3,p=1) + MaxPool -> (512, 14, 14)
# Block 5: 3x Conv(512,512,3,p=1) + MaxPool -> (512, 7, 7)
# Classifier: 25088 -> 4096 -> 4096 -> 1000
```
Total params: ~138M (most in FC layers).

### ResNet (2015)
Key innovation: skip/residual connections enable training 100+ layer networks.

```python
class BasicBlock(nn.Module):
    """ResNet-18/34 block. Input: (C, H, W) -> Output: (C, H, W)"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):          # x: (B, in_ch, H, W)
        out = F.relu(self.bn1(self.conv1(x)))  # (B, out_ch, H', W')
        out = self.bn2(self.conv2(out))         # (B, out_ch, H', W')
        out += self.shortcut(x)                 # residual connection
        return F.relu(out)                      # (B, out_ch, H', W')


class Bottleneck(nn.Module):
    """ResNet-50/101/152 block. Reduces computation with 1x1 squeeze."""
    expansion = 4
    def __init__(self, in_ch, mid_ch, stride=1):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)       # squeeze
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)      # expand
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)
```

ResNet variants: 18 (BasicBlock×[2,2,2,2]), 34 ([3,4,6,3]), 50 (Bottleneck×[3,4,6,3]), 101 ([3,4,23,3]), 152 ([3,8,36,3]).

### DenseNet (2017)
Key innovation: each layer receives feature maps from ALL preceding layers (dense connectivity).

```python
class DenseLayer(nn.Module):
    """BN-ReLU-1x1Conv-BN-ReLU-3x3Conv. Growth rate k = new channels per layer."""
    def __init__(self, in_ch, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, 4 * growth_rate, 1, bias=False)  # bottleneck
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)

    def forward(self, x):  # x: (B, in_ch, H, W)
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], dim=1)  # (B, in_ch + growth_rate, H, W)
```

Transition layer between blocks: BN → 1×1 Conv (halve channels) → AvgPool2d(2).

### EfficientNet (2019)
Key innovation: compound scaling (depth α, width β, resolution γ) with constraint α·β²·γ² ≈ 2.

Base building block: **MBConv** (Mobile Inverted Bottleneck + Squeeze-and-Excitation):
```python
class MBConv(nn.Module):
    """expand(1x1) -> depthwise(3x3/5x5) -> SE -> project(1x1) + skip"""
    def __init__(self, in_ch, out_ch, expand_ratio, kernel_size, stride, se_ratio=0.25):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_skip = (stride == 1 and in_ch == out_ch)
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.SiLU(),
        ) if expand_ratio != 1 else nn.Identity()
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size, stride=stride,
                      padding=kernel_size//2, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch), nn.SiLU(),
        )
        se_ch = max(1, int(in_ch * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_ch, se_ch, 1), nn.SiLU(),
            nn.Conv2d(se_ch, mid_ch, 1), nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = out * self.se(out)
        out = self.project(out)
        return x + out if self.use_skip else out
```

---

## Key Concepts

### Residual Connections

```
y = F(x) + x          # identity shortcut (same dimensions)
y = F(x) + W_s(x)     # projection shortcut (dimension mismatch)
```

Why they work:
- Gradient flows directly through addition (no vanishing)
- Network only needs to learn the residual F(x) = H(x) - x
- Enables training 1000+ layer networks

Variants: Pre-activation ResNet (BN→ReLU→Conv), ResNeXt (grouped convolutions in residual path).

### 1×1 Convolutions

Pointwise operations across channels only (no spatial mixing):
- **Channel reduction** (bottleneck): 256→64 before expensive 3×3
- **Channel expansion**: 64→256 after bottleneck
- **Cross-channel interaction**: learns channel combinations
- **Adds non-linearity**: when followed by activation

```python
# Reduce 512 channels to 64, apply 3x3, expand back
nn.Conv2d(512, 64, 1),   # (512,H,W) -> (64,H,W) -- 32K params
nn.Conv2d(64, 64, 3, padding=1),  # (64,H,W) -> (64,H,W) -- 36K params
nn.Conv2d(64, 512, 1),   # (64,H,W) -> (512,H,W) -- 32K params
# Total: 100K vs 2.4M for direct Conv2d(512, 512, 3)
```

### Pooling Strategies

| Type | Use Case | PyTorch |
|------|----------|---------|
| MaxPool2d(2,2) | Downsample, translation invariance | `nn.MaxPool2d(2, 2)` |
| AvgPool2d(2,2) | Smoother downsample | `nn.AvgPool2d(2, 2)` |
| AdaptiveAvgPool2d(1) | Global average pool (any input→1×1) | `nn.AdaptiveAvgPool2d(1)` |
| Strided conv | Learnable downsample (modern preference) | `nn.Conv2d(..., stride=2)` |

**Global Average Pooling** replaces FC layers at the end of modern CNNs:
```python
# Instead of: Flatten() -> Linear(512*7*7, 1000)
nn.AdaptiveAvgPool2d(1)  # (B, 512, H, W) -> (B, 512, 1, 1)
nn.Flatten()             # -> (B, 512)
nn.Linear(512, num_classes)
```

### Receptive Field Calculation

For a stack of layers (from input to deeper layers):
```
RF_k = RF_{k-1} + (K_k - 1) * jump_{k-1}
jump_k = jump_{k-1} * stride_k
```

Starting: RF_0 = 1, jump_0 = 1.

| Architecture | Approx. Receptive Field |
|-------------|------------------------|
| VGG-16 | 212×212 |
| ResNet-50 | 483×483 |
| 3 stacked 3×3 convs | 7×7 equivalent |

### Depthwise Separable Convolutions

Factorize standard conv into spatial + channel operations:

```python
# Standard conv: K×K×C_in×C_out params
nn.Conv2d(64, 128, 3, padding=1)  # 64*128*3*3 = 73,728 params

# Depthwise separable: K×K×C_in + C_in×C_out params
nn.Sequential(
    # Depthwise: one K×K filter per input channel
    nn.Conv2d(64, 64, 3, padding=1, groups=64),  # 64*3*3 = 576 params
    nn.BatchNorm2d(64), nn.ReLU6(),
    # Pointwise: 1×1 conv for channel mixing
    nn.Conv2d(64, 128, 1),                        # 64*128 = 8,192 params
    nn.BatchNorm2d(128), nn.ReLU6(),
)
# Total: 8,768 params — ~8.4x reduction
```

Used in: MobileNet, EfficientNet, Xception.

---

## Modern Architectures

### ConvNeXt (2022)
"A ConvNet for the 2020s" — modernizes ResNet with Transformer-era design choices:

```python
class ConvNeXtBlock(nn.Module):
    """
    Design choices from Transformers applied to ConvNets:
    - 7×7 depthwise conv (large kernel, like ViT attention span)
    - Inverted bottleneck (expand 4x in MLP, like Transformer FFN)
    - LayerNorm instead of BatchNorm
    - GELU activation
    - Fewer activation/norm layers
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)  # depthwise
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)   # inverted bottleneck expand
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)   # project back
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))  # layer scale

    def forward(self, x):              # x: (B, C, H, W)
        residual = x
        x = self.dwconv(x)            # (B, C, H, W) spatial mixing
        x = x.permute(0, 2, 3, 1)     # (B, H, W, C) for LayerNorm
        x = self.norm(x)
        x = self.pwconv1(x)            # (B, H, W, 4C) expand
        x = self.act(x)
        x = self.pwconv2(x)            # (B, H, W, C) project
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)     # (B, C, H, W)
        return residual + x
```

ConvNeXt-T/S/B/L/XL scale by depth [3,3,9,3] → [3,3,27,3] and width 96→192.

---

## Feature Pyramid Network (FPN)

Multi-scale feature extraction for detection/segmentation:

```python
class SimpleFPN(nn.Module):
    """Top-down pathway with lateral connections."""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        self.smooths = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features):
        laterals = [l(f) for l, f in zip(self.laterals, features)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] += F.interpolate(laterals[i+1], scale_factor=2)
        return [s(l) for s, l in zip(self.smooths, laterals)]
```

---

## Transfer Learning

### Frozen Feature Extractor
```python
model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, num_classes)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

### Fine-Tuning (gradual unfreeze)
```python
model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
model.fc = nn.Linear(2048, num_classes)

# Phase 1: Train head only
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Phase 2: Unfreeze later layers with lower LR
for param in model.layer4.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])
```

### When to Use What

| Dataset Size | Similarity to ImageNet | Strategy |
|-------------|----------------------|----------|
| Small | High | Frozen features + linear head |
| Small | Low | Frozen features + small MLP head |
| Large | High | Fine-tune all layers, low LR |
| Large | Low | Fine-tune all, or train from scratch |

---

## Data Augmentation

### Standard (torchvision.transforms v2)
```python
from torchvision.transforms import v2

train_transform = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.08, 1.0)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Advanced Techniques

| Technique | Effect | When |
|-----------|--------|------|
| **CutMix** | Patches from other images + mixed labels | Always helps classification |
| **MixUp** | Linear interpolation of images + labels | Regularization |
| **RandAugment** | N random transforms at magnitude M | Simple, effective |
| **TrivialAugmentWide** | Single random transform per image | Best default |
| **Cutout/RandomErasing** | Black-out random rectangles | Occlusion robustness |

```python
# RandAugment + CutMix (modern recipe)
train_transform = v2.Compose([
    v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(),
    v2.RandAugment(num_ops=2, magnitude=9),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    v2.RandomErasing(p=0.25),
])
cutmix = v2.CutMix(num_classes=1000)
mixup = v2.MixUp(num_classes=1000)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
```

---

## PyTorch Conv2d Patterns

### Complete Classification Model
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """Shape-annotated CNN for CIFAR-10 (3, 32, 32) -> 10 classes."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),     # (3,32,32) -> (32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),    # (32,32,32) -> (32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                  # (32,32,32) -> (32,16,16)
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1),    # (32,16,16) -> (64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),    # (64,16,16) -> (64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                  # (64,16,16) -> (64,8,8)
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1),   # (64,8,8) -> (128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),             # (128,8,8) -> (128,1,1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # (128,1,1) -> (128,)
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),         # (128,) -> (10,)
        )

    def forward(self, x):  # x: (B, 3, 32, 32)
        x = self.features(x)     # (B, 128, 1, 1)
        return self.classifier(x) # (B, 10)
```

### Conv2d Parameter Reference
```python
nn.Conv2d(
    in_channels,    # int: input feature maps
    out_channels,   # int: output feature maps (num filters)
    kernel_size,    # int or (H,W): spatial extent of filter
    stride=1,       # int or (H,W): step size
    padding=0,      # int, (H,W), or 'same'/'valid'
    dilation=1,     # int or (H,W): spacing between kernel elements
    groups=1,       # int: 1=standard, in_ch=depthwise, other=grouped
    bias=True,      # bool: False when using BatchNorm (BN has its own bias)
    padding_mode='zeros',  # 'zeros', 'reflect', 'replicate', 'circular'
)
# Weight shape: (out_channels, in_channels/groups, kH, kW)
# Output shape: (B, out_channels, H_out, W_out)
```

### Modern Training Recipe (torchvision reference)

```python
# From torchvision's training script for ResNet-50 -> 80.9% top-1
config = {
    'epochs': 600,
    'optimizer': 'SGD',
    'lr': 0.5,
    'lr_scheduler': 'cosineannealinglr',
    'lr_warmup_epochs': 5,
    'lr_warmup_method': 'linear',
    'batch_size': 128,
    'weight_decay': 2e-5,
    'norm_weight_decay': 0.0,
    'label_smoothing': 0.1,
    'mixup_alpha': 0.2,
    'cutmix_alpha': 1.0,
    'augmentation': 'TrivialAugmentWide',
    'random_erase': 0.1,
    'ema_decay': 0.99998,
}
```

---

## Architecture Selection Guide

| Task | Recommended | Why |
|------|-------------|-----|
| Classification (accuracy) | ConvNeXt-B, EfficientNet-V2 | Best accuracy/compute |
| Classification (speed) | MobileNetV3, EfficientNet-B0 | Optimized for inference |
| Detection backbone | ResNet-50 + FPN, ConvNeXt | Multi-scale features |
| Segmentation | ResNet + DeepLabv3, UNet | Dense prediction |
| Edge deployment | MobileNetV3-Small | <5M params, fast |
| Transfer learning | ResNet-50, ConvNeXt-T | Well-studied, good features |
| From scratch (small data) | Simple custom CNN | Avoid overfitting |

---

## References

- [Deep Residual Learning (He et al., 2015)](https://arxiv.org/abs/1512.03385) — ResNet skip connections
- [EfficientNet (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946) — Compound scaling for CNNs
