---
name: Vision Models
description: Vision transformer architectures (ViT, Swin, DeiT), CLIP, SAM, DINOv2, MAE, DETR, YOLOv8, transfer learning patterns, timm/HuggingFace usage, and task selection guide for classification, detection, and segmentation.
---

## Why This Exists

**Problem**: Vision tasks (classification, detection, segmentation) benefit enormously from pretraining on large datasets, but the landscape of pretrained models has fragmented into dozens of architectures (ViT, Swin, CLIP, SAM, DINO, DETR, YOLO) each optimized for different tasks and trade-offs.

**Key insight**: Vision Transformers treat images as sequences of patches, enabling the same self-attention mechanism from NLP to learn visual features — combined with large-scale pretraining (supervised, contrastive, or self-supervised), these models transfer to virtually any vision task with minimal adaptation.

**Reach for this when**: You need a pretrained visual backbone (DINOv2, CLIP), zero-shot classification (CLIP), real-time detection (YOLOv8), interactive segmentation (SAM), or high-accuracy dense prediction (Swin). Choose by task requirements: speed → YOLO, versatility → CLIP, accuracy → Swin/DINOv2, segmentation → SAM.


# Vision Models Skill

## Architecture Overview

| Model | Type | Key Idea | Best For |
|-------|------|----------|----------|
| ViT | Classification | Patch embeddings + transformer encoder | Image classification, feature extraction |
| DeiT | Classification | ViT + distillation token + data-efficient training | Classification without massive datasets |
| Swin | Hierarchical | Shifted windows + multi-scale features | Dense prediction, detection, segmentation |
| CLIP | Vision-Language | Contrastive image-text pretraining | Zero-shot classification, retrieval, embeddings |
| SAM | Segmentation | Promptable segmentation (points/boxes/text) | Interactive/automatic segmentation |
| DINO/DINOv2 | Self-supervised | Self-distillation, no labels needed | Feature extraction, retrieval, zero-shot |
| MAE | Self-supervised | Masked patch reconstruction (75% masking) | Pretraining, representation learning |
| DETR | Detection | End-to-end detection with transformers, no NMS/anchors | Object detection without post-processing |
| YOLOv8 | Detection | Real-time CNN detection + segmentation | Real-time inference, edge deployment |

## Task Selection Guide

```
What's your task?
├── Classification (single label per image)
│   ├── Have lots of labeled data? → ViT-L/H, Swin-L
│   ├── Limited data? → DeiT (distillation), DINO fine-tune
│   └── Zero-shot (no training)? → CLIP
├── Object Detection (boxes around objects)
│   ├── Real-time needed? → YOLOv8 (ultralytics)
│   ├── High accuracy, latency OK? → DETR, DINO-DETR
│   └── Custom classes, few examples? → Grounding DINO + CLIP
├── Segmentation
│   ├── Semantic (pixel classes)? → Swin + UperNet, SegFormer
│   ├── Instance (separate objects)? → Mask R-CNN, YOLOv8-seg
│   ├── Interactive/promptable? → SAM
│   └── Panoptic? → Mask2Former
├── Feature Extraction / Embeddings
│   ├── General visual features? → DINOv2
│   ├── Image-text aligned? → CLIP
│   └── Self-supervised pretrain? → MAE → fine-tune
└── Zero-Shot / Open Vocabulary
    ├── Classification? → CLIP
    ├── Detection? → Grounding DINO, OWLv2
    └── Segmentation? → SAM + CLIP, Grounded SAM
```

## Core Architectures

### ViT (Vision Transformer)

```python
import timm

model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
model.eval()
output = model(images)  # (B, num_classes)

# Feature extraction (no classification head)
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
features = model(images)  # (B, 768)
```

### CLIP

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Zero-shot classification
inputs = processor(text=["a dog", "a cat", "a bird"], images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=-1)

# Image embeddings only
image_features = model.get_image_features(**processor(images=image, return_tensors="pt"))
image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # L2 normalize
```

### SAM (Segment Anything)

```python
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# Point prompt
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),  # 1=foreground, 0=background
    multimask_output=True
)

# Automatic mask generation (segment everything)
generator = SamAutomaticMaskGenerator(sam)
masks = generator.generate(image_rgb)
```

### DINOv2

```python
import torch

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
model.eval()

with torch.no_grad():
    features = model(images)  # (B, 768) CLS token

# Linear probe (freeze backbone, train linear head)
model.requires_grad_(False)
classifier = torch.nn.Linear(768, num_classes)
output = classifier(model(images))
```

### YOLOv8 (Ultralytics)

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("image.jpg")
boxes = results[0].boxes  # .xyxy, .conf, .cls

# Training on custom dataset
model = YOLO("yolov8n.pt")
model.train(data="dataset.yaml", epochs=100, imgsz=640, batch=16)

# Export
model.export(format="onnx")
```

## Transfer Learning Patterns

### Pattern 1: Linear Probe (Frozen Backbone)

```python
backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
backbone.requires_grad_(False)
model = nn.Sequential(backbone, nn.Linear(768, num_classes))
optimizer = torch.optim.Adam(model[1].parameters(), lr=1e-3)
```

### Pattern 2: Fine-tune with Lower LR on Backbone

```python
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
param_groups = [
    {"params": model.head.parameters(), "lr": 1e-3},
    {"params": [p for n, p in model.named_parameters() if "head" not in n], "lr": 1e-5},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
```

### Pattern 3: CLIP Zero-Shot → Few-Shot

```python
# Step 1: Zero-shot with text prompts
texts = [f"a photo of a {c}" for c in class_names]

# Step 2: If insufficient, use CLIP features + linear probe
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(np.vstack(features), labels)
```

## timm Library Patterns

```python
import timm

timm.list_models('vit*')          # List available models
timm.list_models(pretrained=True) # Only with pretrained weights

# Model with custom input size
model = timm.create_model('vit_base_patch16_224', pretrained=True, img_size=384)

# Get model-specific transforms
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# Feature extraction at multiple scales
model = timm.create_model('resnet50', pretrained=True, features_only=True)
features = model(img)  # List of feature maps at each stage
```

## HuggingFace Patterns

```python
from transformers import pipeline

# Generic image classification
clf = pipeline("image-classification", model="google/vit-base-patch16-224")
result = clf("image.jpg")

# Object detection
det = pipeline("object-detection", model="facebook/detr-resnet-50")
result = det("image.jpg")

# Zero-shot image classification
zs = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
result = zs("image.jpg", candidate_labels=["dog", "cat", "bird"])
```

## Key Design Decisions

| Decision | Recommendation |
|----------|---------------|
| Backbone for classification | ViT-B or Swin-B (timm) |
| Backbone for detection | Swin + DINO-DETR or YOLOv8 |
| Need real-time? | YOLOv8 (ultralytics) |
| No labeled data? | CLIP zero-shot or DINOv2 linear probe |
| Self-supervised pretrain? | MAE or DINOv2 |
| Interactive segmentation? | SAM |
| Open-vocabulary detection? | Grounding DINO |
| Production deployment? | ONNX export + TensorRT |
| Limited compute? | DeiT-S, EfficientNet, YOLOv8-n |
| Maximum accuracy? | Swin-L, ViT-H, BEiT-3 |

---

## References

- [Vision Transformer (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — ViT: images as patch sequences
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) — Pretrained vision model zoo
