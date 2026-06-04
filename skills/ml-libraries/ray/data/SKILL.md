---
name: ray-data
description: Streaming distributed dataset processing for ML pipelines. Covers read/write, map_batches with UDFs, tokenization, image transforms, and integration with Ray Train. Use when preprocessing data too large for memory, building ETL for training, or need distributed tokenization/transforms.
---

# Ray Data

- **Docs**: https://docs.ray.io/en/latest/data/data.html
- **API**: https://docs.ray.io/en/latest/data/api/dataset.html
- **Key design**: Streaming block-by-block execution, never materializes full dataset

## Why This Exists

**Problem**: ML datasets (tokenization corpora, image archives, parquet shards) are often larger than RAM; pandas and HuggingFace `datasets` load eagerly and choke on multi-TB data, while custom multiprocessing pipelines are brittle and don't integrate with distributed trainers.

**Key insight**: Ray Data streams data block-by-block through a pipeline of transforms, so memory usage stays bounded regardless of dataset size, and the same pipeline feeds directly into Ray Train workers without extra serialization.

**Reach for this when**: You need to preprocess or tokenize datasets too large for memory, apply GPU-accelerated transforms at scale (e.g., image augmentation), or pipe data into a distributed Ray Train job — prefer it over pandas/HuggingFace `datasets` once data exceeds a single node's RAM.

## Reading Data

```python
import ray

ds = ray.data.read_parquet("s3://bucket/data/")
ds = ray.data.read_json("data/*.json")
ds = ray.data.read_csv("data.csv")
ds = ray.data.read_images("s3://bucket/images/", mode="RGB")
ds = ray.data.from_huggingface("imdb")

ds.schema()   # Arrow schema
ds.count()
ds.show(5)
```

## Core Operations

```python
ds = ds.repartition(100)              # control parallelism
ds = ds.random_shuffle()              # expensive -- materializes
train, val = ds.train_test_split(test_size=0.2)
shards = ds.split(n=4, equal=True)    # for multi-worker training

# Write
ds.write_parquet("s3://bucket/output/")
```

## map_batches (The Main Tool)

```python
# Stateless function
def double(batch: dict) -> dict:
    batch["value"] = batch["value"] * 2
    return batch

ds = ds.map_batches(double, batch_size=256)

# Stateful class UDF (loads model once per worker)
class TokenizeUDF:
    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

    def __call__(self, batch: dict) -> dict:
        enc = self.tok(batch["text"], truncation=True, max_length=2048, padding="max_length")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

ds = ds.map_batches(TokenizeUDF, concurrency=4, batch_size=512)
```

### Key Parameters

| Param | Default | Purpose |
|-------|---------|---------|
| `batch_size` | 4096 | Rows per batch |
| `concurrency` | auto | Parallel workers |
| `num_gpus` | 0 | GPU per worker (for GPU transforms) |
| `batch_format` | "default" | `"numpy"`, `"pandas"`, `"pyarrow"` |

## Streaming into Training

```python
for batch in ds.iter_batches(batch_size=32, prefetch_batches=4):
    # batch is dict of numpy arrays
    model.train_step(batch)
```

## Integration with Ray Train

```python
from ray.train import get_dataset_shard

def train_func(config):
    # Each worker gets its shard automatically
    shard = get_dataset_shard("train")
    for batch in shard.iter_torch_batches(batch_size=32):
        ...

from ray.train.torch import TorchTrainer
trainer = TorchTrainer(
    train_func,
    datasets={"train": ds},  # pass dataset to trainer
    scaling_config=ScalingConfig(num_workers=4),
)
```

## Image Preprocessing

```python
class ImagePreprocess:
    def __init__(self):
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, batch: dict) -> dict:
        from PIL import Image
        import numpy as np
        imgs = [self.transform(Image.fromarray(i)).numpy() for i in batch["image"]]
        return {"image": np.stack(imgs), "label": batch["label"]}

ds = ray.data.read_images("s3://bucket/train/", mode="RGB")
ds = ds.map_batches(ImagePreprocess, concurrency=8, batch_size=64)
```

## Performance Tips

| Tip | Why |
|-----|-----|
| Use class UDFs for expensive init (model loading) | Init runs once per worker, not per batch |
| Set `concurrency` explicitly | Auto-detection can over/under-provision |
| Large `batch_size` for GPU transforms | Amortize kernel launch overhead |
| `prefetch_batches=4` in `iter_batches` | Overlap I/O with compute |
| Avoid `random_shuffle()` on large data | Materializes entire dataset; use per-epoch shuffle in trainer |

## References

- Official docs: https://docs.ray.io/en/latest/data/data.html
- Loading data guide: https://docs.ray.io/en/latest/data/loading-data.html
- GitHub: https://github.com/ray-project/ray
