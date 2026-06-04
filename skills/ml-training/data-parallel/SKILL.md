---
name: distributed-data-parallelism
description: Distributed training strategies for scaling deep learning across multiple GPUs
version: 1.0.0
triggers:
  - distributed training
  - multi-GPU training
  - DDP
  - FSDP
  - data parallelism
  - tensor parallelism
  - pipeline parallelism
  - model parallelism
  - scale training
  - OOM during training
  - gradient all-reduce
---

# Distributed Data Parallelism

Scale deep learning training across multiple GPUs when a single device is insufficient for model size, data volume, or wall-clock time constraints.

## Why This Exists

**Problem**: A single GPU has a fixed memory ceiling (24–80 GB) and a fixed compute budget — large models OOM at batch size 1, and even models that fit require days or weeks per epoch, making iteration impractical.

**Key insight**: Data parallelism (DDP/FSDP) and model parallelism (TP/PP) attack different bottlenecks — DDP accelerates training by splitting data across identical model replicas; FSDP/TP/PP make oversized models fit by distributing the model itself across devices.

**Reach for this when**: Your model or batch size exceeds a single GPU's memory, or training wall-clock time is unacceptable. Use DDP first (simplest), escalate to FSDP when the model itself doesn't fit one GPU, and add TP/PP only for 70B+ models where FSDP alone is insufficient.

---

## 1 · Why Distributed Training Exists

| Problem | Symptom | Solution Category |
|---------|---------|-------------------|
| Training too slow on 1 GPU | Days/weeks per epoch | Data parallelism (DDP) |
| Model fits 1 GPU but barely | OOM with large batches | Data parallelism (DDP) |
| Model doesn't fit 1 GPU | OOM even at batch=1 | Model parallelism (FSDP/TP/PP) |
| Both model AND data are huge | LLM pre-training | Hybrid (FSDP + TP + PP) |

Core insight: **data parallelism** replicates the model and splits data; **model parallelism** splits the model itself across devices.

---

## 2 · DDP (DistributedDataParallel)

The workhorse of multi-GPU training. Each GPU holds a full model replica.

**How it works:**
1. Each rank gets a different mini-batch (via `DistributedSampler`)
2. Forward pass runs independently on each rank
3. Backward pass computes local gradients
4. `all-reduce` synchronizes gradients across ranks (ring or tree)
5. Each rank applies identical optimizer step → models stay in sync

**Key properties:**
- Communication: gradient all-reduce once per backward pass
- Memory: full model + optimizer + gradients on EACH GPU
- Scaling: near-linear up to ~64 GPUs (communication-bound beyond)
- Overlap: gradient comm overlaps with backward computation (bucket fusion)

```python
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def train_ddp(rank, world_size):
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Model on this rank's GPU
    model = nn.Linear(1024, 512).to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Shard data across ranks
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    for epoch in range(10):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        for batch in loader:
            batch = batch.to(rank)
            loss = model(batch).sum()
            loss.backward()       # Gradients all-reduced automatically
            optimizer.step()
            optimizer.zero_grad()

    dist.destroy_process_group()

# Launch: torchrun --nproc_per_node=4 script.py
```

**Launch methods:**
- `torchrun --nproc_per_node=N script.py` (recommended)
- `torch.multiprocessing.spawn(fn, nprocs=N)`
- SLURM: `torchrun --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID`

---

## 3 · FSDP (FullyShardedDataParallel)

When the model doesn't fit in one GPU's memory. Shards parameters, gradients, AND optimizer states across ranks.

**How it works (ZeRO-3 equivalent):**
1. Parameters sharded across ranks (each rank holds 1/N of params)
2. Before forward: `all-gather` to reconstruct full layer params
3. After forward: discard non-owned params (free memory)
4. Before backward: `all-gather` again for that layer
5. After backward: `reduce-scatter` gradients (each rank gets its shard)
6. Optimizer step on local shard only

**Memory savings vs DDP:**

| Component | DDP (per GPU) | FSDP (per GPU) |
|-----------|---------------|----------------|
| Parameters | Full model | 1/N of model |
| Gradients | Full | 1/N |
| Optimizer states | Full | 1/N |
| Peak activation | Same | Same (unless activation checkpointing) |

For a 7B model with AdamW (mixed precision): DDP needs ~56GB/GPU, FSDP with 8 GPUs needs ~7GB/GPU for params+optimizer.

```python
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

def train_fsdp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = MyTransformer().to(rank)

    # Wrap policy: shard at transformer block boundaries
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # Mixed precision for memory + speed
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
        mixed_precision=mp_policy,
        auto_wrap_policy=wrap_policy,
        device_id=rank,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for batch in loader:
        batch = batch.to(rank)
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Sharding strategies:**
- `FULL_SHARD` (ZeRO-3): Maximum memory savings, most communication
- `SHARD_GRAD_OP` (ZeRO-2): Shard gradients+optimizer, keep full params
- `NO_SHARD`: Equivalent to DDP (for debugging)
- `HYBRID_SHARD`: Full shard within node, replicate across nodes

---

## 4 · Pipeline Parallelism (PP)

Split model layers across GPUs sequentially. GPU 0 runs layers 0-11, GPU 1 runs layers 12-23, etc.

**Problem:** Naive PP has massive bubble time (GPUs idle while waiting).

**Solutions:**
- **GPipe:** Split mini-batch into micro-batches, pipeline them
- **1F1B (one-forward-one-backward):** Interleave forward/backward of micro-batches to reduce memory peak
- **Interleaved 1F1B:** Assign non-contiguous layers to each stage for better overlap

**When to use:**
- Model too large for FSDP alone (100B+ params)
- Network bandwidth between nodes is limited (PP has less communication than FSDP)
- Combined with TP within nodes

```python
# PyTorch PipelineStage API (torch >= 2.2)
from torch.distributed.pipelining import SplitPoint, pipeline, ScheduleGPipe

# Define split points
pipe = pipeline(
    model,
    mb_args=(micro_batch,),
    split_spec={
        "layers.12": SplitPoint.BEGINNING,  # Split after layer 11
        "layers.24": SplitPoint.BEGINNING,
    },
)

schedule = ScheduleGPipe(pipe, n_microbatches=8)
if rank == last_stage:
    losses = schedule.step(micro_batch)
else:
    schedule.step()
```

**Trade-offs:**
- Pro: Low cross-node communication (only activations at boundaries)
- Con: Pipeline bubble (15-25% GPU idle time typical)
- Con: Uneven layer memory can cause imbalance

---

## 5 · Tensor Parallelism (TP)

Split individual layers across GPUs. Each GPU computes a portion of a matrix multiplication.

**Column-parallel linear:** Split weight columns → each GPU computes partial output → `all-gather` to combine
**Row-parallel linear:** Split weight rows → each GPU needs full input → `reduce-scatter` after matmul

**When to use:**
- Within a single node (requires high-bandwidth NVLink/NVSwitch)
- For attention heads (naturally parallelizable: split heads across GPUs)
- Combined with PP across nodes

```python
# Using torch.distributed.tensor (DTensor) for TP
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

mesh = DeviceMesh("cuda", list(range(world_size)))

# Parallelize MLP: column-split first linear, row-split second
parallelize_module(
    model.mlp,
    mesh,
    {
        "fc1": ColwiseParallel(),  # Split output features
        "fc2": RowwiseParallel(),  # Split input features
    },
)
```

**Trade-offs:**
- Pro: No pipeline bubble, perfect load balance
- Con: Heavy communication (all-reduce per layer)
- Con: Requires NVLink bandwidth (200+ GB/s); unusable across nodes

---

## 6 · Decision Table

| Scenario | Strategy | Why |
|----------|----------|-----|
| Model fits 1 GPU, want faster training | **DDP** | Simplest, near-linear scaling |
| Model barely fits 1 GPU, OOM on large batch | **DDP + gradient accumulation** | Simulate larger batch without more memory |
| Model doesn't fit 1 GPU (7B-13B) | **FSDP** | Shard everything, 1 node sufficient |
| Model 13B-70B, single node | **FSDP + activation checkpointing** | Trade compute for memory |
| Model 70B+, multi-node | **FSDP + TP (intra-node) + PP (inter-node)** | 3D parallelism |
| Fine-tuning large model | **FSDP + LoRA/QLoRA** | Only shard trainable params |
| Slow inter-node network | **PP between nodes + TP/FSDP within** | PP minimizes cross-node traffic |
| Inference (not training) | **TP** | Reduces latency per token |

**Rule of thumb for GPU count:**

| Model Size | Min GPUs (training, bf16) | Recommended Strategy |
|-----------|---------------------------|---------------------|
| < 2B | 1 | DDP if multi-GPU |
| 2B-7B | 1-2 (FSDP) or 4 (DDP) | FSDP |
| 7B-13B | 2-4 | FSDP |
| 13B-70B | 8-16 | FSDP + TP |
| 70B+ | 32+ | FSDP + TP + PP |

---

## 7 · Gotchas

1. **DDP + DataLoader:** Always use `DistributedSampler` and call `sampler.set_epoch(epoch)` — otherwise every rank trains on the same data order each epoch
2. **FSDP checkpoint saving:** Use `FullStateDictConfig(offload_to_cpu=True)` or `ShardedStateDictConfig` — naive `model.state_dict()` on FSDP returns sharded tensors
3. **Mixed precision + FSDP:** Set `reduce_dtype` to match `param_dtype` — mismatched precision causes silent accuracy loss
4. **NCCL timeout:** Set `NCCL_TIMEOUT=1800` for large models — default 30min can be too short for first all-gather
5. **Gradient accumulation with FSDP:** Use `model.no_sync()` context manager for accumulation steps to avoid unnecessary all-reduces
6. **TP requires even division:** Hidden dimensions must be divisible by TP degree (e.g., 4096 / 8 = 512 ✓)
7. **Pipeline bubble:** With N stages and M micro-batches, bubble fraction ≈ (N-1)/(N-1+M). Use M >> N to minimize
8. **FSDP + torch.compile:** As of PyTorch 2.3+, use `torch.compile` BEFORE wrapping with FSDP for best performance

---

## 8 · References

- [PyTorch Distributed Overview](https://pytorch.org/docs/stable/distributed.html)
- [DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Pipeline Parallelism](https://pytorch.org/docs/stable/distributed.pipelining.html)
- [Tensor Parallelism with DTensor](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)
- [ZeRO Paper (DeepSpeed)](https://arxiv.org/abs/1910.02054)
- [Megatron-LM 3D Parallelism](https://arxiv.org/abs/2104.04473)
- [PyTorch FSDP: Experiences on Scaling](https://arxiv.org/abs/2304.11277)

## References

- Official docs (PyTorch distributed): https://pytorch.org/docs/stable/distributed.html
- Official docs (FSDP): https://pytorch.org/docs/stable/fsdp.html
- Paper (ZeRO): https://arxiv.org/abs/1910.02054
- GitHub (DeepSpeed): https://github.com/microsoft/DeepSpeed
