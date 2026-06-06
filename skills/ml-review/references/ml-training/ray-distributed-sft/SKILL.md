---
name: ray-distributed-sft
description: Distributed supervised fine-tuning across multiple GPUs/nodes using Ray Train with TRL, DeepSpeed ZeRO, and FSDP. Use when model or data is too large for single GPU, need multi-node training, or want fault-tolerant distributed SFT with HuggingFace Transformers.
---

# Distributed SFT with Ray Train

Scale SFT from single GPU to multi-node using Ray's `TorchTrainer` wrapper around HuggingFace/TRL.

- **Ray Train docs**: https://docs.ray.io/en/latest/train/train.html
- **Ray + HF guide**: https://docs.ray.io/en/latest/train/getting-started-transformers.html
- **TRL SFTTrainer**: https://huggingface.co/docs/trl/sft_trainer
- **DeepSpeed ZeRO**: https://www.deepspeed.ai/tutorials/zero/

## Why This Exists

**Problem**: `torchrun` and Accelerate handle multi-GPU on a single node well, but fall apart across nodes — checkpoint coordination, worker failure recovery, and elastic scaling require external orchestration that neither provides out of the box.

**Key insight**: Wrapping HuggingFace `Trainer` / TRL `SFTTrainer` with Ray `TorchTrainer` gives you multi-node process management, fault tolerance (auto-restart on worker failure), and cloud-native checkpoint storage (S3/GCS) with almost no changes to the inner training loop.

**Reach for this when**: Your SFT job needs more than one node, your dataset is too large to fit in memory (use Ray Data streaming), or you need fault-tolerant training with automatic worker restart. For single-node multi-GPU SFT, `torchrun` + DeepSpeed is simpler. For distributed GRPO/PPO, use `distributed-grpo` instead.

---

## Architecture

```
Ray Cluster
├── Worker 0 (GPU 0) ── HF Trainer + DeepSpeed ZeRO ── gradient sync via NCCL
├── Worker 1 (GPU 1) ── HF Trainer + DeepSpeed ZeRO ── gradient sync via NCCL
├── Worker 2 (GPU 2) ── HF Trainer + DeepSpeed ZeRO ── gradient sync via NCCL
└── Worker 3 (GPU 3) ── HF Trainer + DeepSpeed ZeRO ── gradient sync via NCCL
```

Ray replaces `torchrun` as process launcher. TorchTrainer handles:
- Process group initialization across nodes
- GPU allocation and rank assignment
- Checkpoint saving to persistent storage (S3/NFS)
- Fault tolerance with automatic worker restart

---

## Complete Example: TRL + Ray Train + DeepSpeed

```python
import os
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import load_dataset

def train_func(config):
    """Runs on each distributed worker."""
    model_name = config["model_name"]

    # Load model + tokenizer INSIDE train_func (not serialized from driver)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset inside train_func
    dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
    # ... format dataset ...

    peft_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
    )

    training_args = SFTConfig(
        output_dir="/tmp/sft_output",
        dataset_text_field="text",
        max_seq_length=2048,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        deepspeed={
            "zero_optimization": {"stage": 2},
            "bf16": {"enabled": True},
            "train_micro_batch_size_per_gpu": 2,
        },
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
    )

    # Ray Train integration
    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)
    trainer.train()

# Launch
ray_trainer = TorchTrainer(
    train_func,
    train_loop_config={"model_name": "meta-llama/Llama-3.1-8B-Instruct"},
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
    run_config=RunConfig(storage_path="s3://your-bucket/experiments"),
)
result = ray_trainer.fit()
```

---

## Key Components

### ScalingConfig
```python
ScalingConfig(
    num_workers=4,              # Total GPUs across cluster
    use_gpu=True,
    resources_per_worker={"GPU": 1, "CPU": 4},
)
```

### RunConfig
```python
RunConfig(
    storage_path="s3://bucket/experiments",  # Required for multi-node checkpoints
    name="llama-sft-run",
    checkpoint_config=CheckpointConfig(num_to_keep=2),
    failure_config=FailureConfig(max_failures=3),  # Auto-restart on failure
)
```

### DeepSpeed ZeRO Stages

| Stage | What's Sharded | VRAM Savings | Best For |
|-------|---------------|-------------|----------|
| ZeRO-1 | Optimizer states | ~4x | Small models |
| ZeRO-2 | + Gradients | ~8x | **Default choice for LoRA** |
| ZeRO-3 | + Parameters | ~16x | Very large models (70B+) |

---

## Ray Data for Streaming Large Datasets

When dataset doesn't fit in memory:

```python
import ray.data

# Stream from S3 without full materialization
ds = ray.data.read_parquet("s3://bucket/training-data/")

def tokenize_batch(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, max_length=2048, padding="max_length")

ds = ds.map_batches(tokenize_batch, fn_kwargs={"tokenizer": tokenizer})
```

Ray Data docs: https://docs.ray.io/en/latest/data/data.html

---

## FSDP Alternative (No DeepSpeed)

For PyTorch-native sharding without DeepSpeed:

```python
training_args = SFTConfig(
    ...,
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        "activation_checkpointing": True,
    },
)
```

FSDP shards model parameters across GPUs — each GPU holds 1/N of the model. Good for large models that don't fit on a single GPU even with LoRA.

---

## Unsloth + Ray (Limited)

Unsloth's Triton kernels only support DDP (not FSDP). If you must use Unsloth distributed:

```python
def train_func():
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
    os.environ["UNSLOTH_DISABLE_TRAINER_PATCHING"] = "1"

    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=False,  # DDP requires 16-bit (QLoRA + DDP broken)
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    )
    # ... rest same, but:
    # - batch_size=1 only (current limitation)
    # - ddp_find_unused_parameters=False required
```

**Limitations**: No 4-bit, batch_size=1, no FSDP. Better to use standard HF + Ray for distributed.

GitHub issue: https://github.com/unslothai/unsloth/issues/2266

---

## Decision Guide

| Scenario | Approach |
|----------|----------|
| Model fits on 1 GPU (≤8B with QLoRA) | Just use Unsloth single-GPU (fastest) |
| SFT on 8B, 4× GPUs | Ray Train + DeepSpeed ZeRO-2 + LoRA |
| SFT on 70B | Ray Train + DeepSpeed ZeRO-3 or FSDP |
| Dataset too large for memory | Ray Data streaming |
| Need fault tolerance / elastic | Ray Train with FailureConfig |
| Want GRPO/PPO distributed | Use OpenRLHF (see `distributed-grpo` skill) |

## References

- Official docs (Ray Train): https://docs.ray.io/en/latest/train/train.html
- GitHub (DeepSpeed): https://github.com/microsoft/DeepSpeed
- HuggingFace Accelerate + DeepSpeed guide: https://huggingface.co/docs/accelerate/main/en/usage_guides/deepspeed
