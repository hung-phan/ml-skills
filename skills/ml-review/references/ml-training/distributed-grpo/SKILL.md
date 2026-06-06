---
name: distributed-grpo
description: Distributed GRPO/PPO reinforcement learning for LLMs across multiple GPUs using OpenRLHF, veRL, or TRL. Covers reward function design, vLLM generation, DeepSpeed training, and Ray orchestration. Use when scaling RL training beyond single GPU, building reasoning models at scale, or need separated inference/training pipelines.
---

# Distributed GRPO & RL Training

For scaling GRPO/PPO beyond single GPU. Separates generation (vLLM) from training (DeepSpeed).

- **OpenRLHF**: https://github.com/OpenRLHF/OpenRLHF
- **OpenRLHF docs**: https://openrlhf.readthedocs.io
- **veRL**: https://github.com/volcengine/verl
- **TRL GRPOTrainer**: https://huggingface.co/docs/trl/grpo_trainer
- **DeepSeek GRPO paper**: https://arxiv.org/abs/2402.03300

## Why This Exists

**Problem**: Single-GPU GRPO (e.g., Unsloth) bottlenecks on generation speed — sampling 8 completions per prompt for a 7B model is slow, and training on larger models (30B+) simply doesn't fit; meanwhile, naively wrapping a policy trainer with DDP still ties fast vLLM inference to the slower DeepSpeed training loop.

**Key insight**: Separating generation (vLLM with tensor parallelism, PagedAttention) from training (DeepSpeed ZeRO sharding) onto independent GPU pools — and synchronizing weights between them — lets each phase run at its optimal throughput instead of blocking the other.

**Reach for this when**: You need GRPO/PPO at scale (multi-GPU, >8B models, or production workloads). Use Unsloth single-GPU for prototyping reward functions first, then graduate to OpenRLHF on Ray for distributed runs, or TRL+Accelerate for a simpler no-Ray path on ≤4 GPUs.

---

## Framework Comparison

| Framework | SFT | GRPO | PPO | Ray Native | vLLM | Ease |
|-----------|-----|------|-----|-----------|------|------|
| **OpenRLHF** | ✅ | ✅ | ✅ | ✅ native | ✅ | ★★★ |
| **veRL** | ✅ | ✅ | ✅ | ✅ native | ✅ | ★★☆ |
| **TRL + Accelerate** | ✅ | ✅ | ✅ | ❌ torchrun | ✅ | ★★★ |
| **TRL + Ray Train** | ✅ | ⚠️ manual | ❌ | ✅ wrapper | ❌ | ★★☆ |
| **Unsloth** | ✅ | ✅ single-GPU | ❌ | ❌ | ✅ | ★★★★ |

**Recommendation**: OpenRLHF for distributed GRPO on Ray. TRL+Accelerate for simpler multi-GPU without Ray.

---

## Option 1: OpenRLHF (Best for GRPO on Ray)

Purpose-built Ray-native RLHF framework. Separates roles across GPU pools.

### Architecture
```
Ray Cluster
├── vLLM Rollout Workers (N GPUs)     ← generate completions fast
│   └── PagedAttention + AutoTP
├── Actor Training Workers (M GPUs)   ← policy gradient updates
│   └── DeepSpeed ZeRO-3
├── Reference Model (shared/offloaded) ← KL penalty computation
└── Reward Function                    ← custom Python or remote HTTP
    ↕ NCCL weight sync between vLLM ↔ DeepSpeed
```

### Installation
```bash
pip install openrlhf[vllm]
```

### Launch GRPO Training
```bash
ray start --head --num-gpus 4

python3 -m openrlhf.cli.train_ppo_ray \
  --algo.advantage.estimator group_norm \
  --actor.model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node 4 \
  --ds.lora.rank 64 \
  --ds.lora.alpha 64 \
  --ds.lora.target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --vllm.num_engines 2 \
  --vllm.tensor_parallel_size 2 \
  --vllm.gpu_memory_utilization 0.7 \
  --rollout.n_samples_per_prompt 8 \
  --reward.remote_url ./reward_func.py \
  --data.input_key prompt \
  --data.prompt_dataset ./prompts.jsonl \
  --train.max_epochs 1 \
  --train.micro_batch_size 4 \
  --actor.adam.lr 5e-6 \
  --ckpt.output_dir ./grpo_output \
  --train.colocate_all \
  --vllm.enable_sleep \
  --ds.enable_sleep
```

### Custom Reward Function
```python
# reward_func.py — OpenRLHF format
import torch

def reward_func(queries, prompts, labels, **kwargs):
    """
    Compute custom rewards for generated responses.
    Args:
        queries: List[str] - Full text (prompt + response)
        prompts: List[str] - Original prompts only
        labels: List[str] - Ground truth from dataset (via --data.label_key)
    Returns:
        dict with:
            - rewards: Tensor for advantage calculation
            - scores: Tensor for dynamic filtering (0-1 range)
    """
    batch_size = len(queries)
    rewards = []
    for text, label in zip(queries, labels):
        answer = extract_answer(text)
        if answer == label:
            rewards.append(3.0)
        elif has_reasoning_tags(text):
            rewards.append(0.5)
        else:
            rewards.append(-1.0)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    return {"rewards": reward_tensor, "scores": reward_tensor}
```

### Key Flags
- `--train.colocate_all`: inference + training on same GPUs (limited hardware)
- `--train.async_enable`: generation and training overlap
- `--algo.advantage.estimator dr_grpo`: DR-GRPO variant (no local std normalization)
- `--ds.lora.rank 64`: LoRA (less VRAM, faster)
- `--vllm.tensor_parallel_size 2`: split vLLM across GPUs

### Resource Allocation (4× L4 24GB)
- **Option A**: 2 GPUs → vLLM rollout (TP=2), 2 GPUs → DeepSpeed training (use separate `--actor.num_gpus_per_node 2`)
- **Option B**: `--train.colocate_all` — all 4 GPUs share both roles (recommended)

---

## Option 2: TRL GRPOTrainer + Accelerate (Simpler, No Ray)

For multi-GPU GRPO without Ray orchestration:

```bash
accelerate launch --num_processes 4 --config_file ds_zero2.yaml train_grpo.py
```

```python
# train_grpo.py
from trl import GRPOTrainer, GRPOConfig

def correctness_reward(prompts, completions, answer, **kwargs):
    responses = [c[0]["content"] for c in completions]
    return [3.0 if extract_answer(r) == a else -3.0
            for r, a in zip(responses, answer)]

def format_reward(prompts, completions, **kwargs):
    responses = [c[0]["content"] for c in completions]
    return [1.0 if "<answer>" in r and "</answer>" in r else -1.0 for r in responses]

config = GRPOConfig(
    output_dir="./grpo_out",
    use_vllm=True,                    # vLLM for fast generation
    vllm_gpu_memory_utilization=0.7,
    num_generations=8,                # Group size
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-6,
    deepspeed="ds_zero2.json",
)

trainer = GRPOTrainer(
    model="meta-llama/Llama-3.1-8B-Instruct",
    reward_funcs=[correctness_reward, format_reward],
    args=config,
    train_dataset=dataset,
)
trainer.train()
```

TRL ≥0.15 supports vLLM as generation backend natively.

TRL GRPO docs: https://huggingface.co/docs/trl/grpo_trainer

---

## Option 3: veRL (Volcano Engine RL)

Hybrid FSDP+TP architecture. GPU pools dynamically switch between training and generation via weight resharding.

- **Repo**: https://github.com/volcengine/verl
- **Paper**: https://arxiv.org/abs/2409.19256

### Architecture
- **Colocated mode**: Same GPUs switch between FSDP-sharded training ↔ vLLM TP-sharded generation
- **Separated mode**: Dedicated GPU pools for each phase
- **Weight resharding**: Single set of GPUs changes parallelism strategy between phases

### Launch
```bash
pip install verl

# Hydra-based config
python -m verl.trainer.main_ppo \
  algorithm=grpo \
  actor_rollout_ref.model.path=meta-llama/Llama-3.1-8B-Instruct \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.n=8 \
  algorithm.kl_ctrl.kl_coef=0.04 \
  trainer.total_epochs=1 \
  trainer.save_path=./verl_output
```

### Reward Function (veRL format)
```python
from verl import DataProto
import torch

def compute_reward_batch(data: DataProto) -> torch.Tensor:
    rewards = []
    for i in range(len(data)):
        response = data.batch['responses'][i]
        ground_truth = data.non_tensor_batch['ground_truth'][i]
        score = 1.0 if check_answer(response, ground_truth) else 0.0
        rewards.append(score)
    return torch.tensor(rewards, dtype=torch.float32)
```

---

## Reward Function Design Patterns

### Pattern 1: Exact Match
```python
def exact_match_reward(prompts, completions, answer, **kwargs):
    return [2.0 if extract(c) == a else -1.0 for c, a in zip(completions, answer)]
```

### Pattern 2: Multi-Signal (Composite)
```python
def composite_reward(prompts, completions, answer, **kwargs):
    scores = []
    for c, a in zip(completions, answer):
        score = 0.0
        if extract_answer(c) == a: score += 3.0      # correctness
        if has_reasoning(c): score += 1.0             # process reward
        if len(c) < 500: score += 0.5                 # conciseness
        if is_well_formatted(c): score += 0.5         # format
        scores.append(score)
    return scores
```

### Pattern 3: LLM-as-Judge (expensive but flexible)
```python
def llm_judge_reward(prompts, completions, **kwargs):
    # Call a judge model (separate API or local) to score
    scores = judge_model.batch_score(prompts, completions, rubric="...")
    return scores
```

---

## Decision Matrix

| Scenario | Best Path |
|----------|-----------|
| Distributed GRPO with custom rewards on Ray | **OpenRLHF** |
| Quick GRPO, ≤8B model, 4 GPUs | **TRL + Accelerate** (no Ray) |
| Maximum GPU efficiency (weight resharding) | **veRL** |
| Production pipeline with fault tolerance | **OpenRLHF on Ray** |
| Single-GPU prototype before scaling | **Unsloth** then graduate |

---

## Practical Strategy: Prototype → Scale

```
Phase 1: Prototype (1 GPU, Unsloth)
  - Design reward functions
  - Validate data format
  - Quick iteration (2x faster)
  - Export LoRA adapter

Phase 2: Scale (Multi-GPU, OpenRLHF)
  - Same reward functions (minor format change)
  - Same model + LoRA config
  - Ray handles distribution
  - vLLM handles generation at scale
```

### Reward Function Portability

```python
# Unsloth/TRL format:
def reward_func(prompts, completions, answer, **kwargs):
    return [score for ...]

# OpenRLHF format (different signature):
def reward_func(queries, prompts, labels, **kwargs):
    return {"rewards": torch.tensor([score for ...]), "scores": torch.tensor([score for ...])}
```

---

## Cluster Setup (OpenRLHF on Ray)

```bash
# Head node
ray start --head --port=6379 --num-gpus=4

# Worker nodes
ray start --address=HEAD_IP:6379 --num-gpus=4

# Verify
ray status

# Launch training
python3 -m openrlhf.cli.train_ppo_ray --algo.advantage.estimator group_norm ...
```

For Slurm clusters, OpenRLHF provides job templates:
https://github.com/OpenRLHF/OpenRLHF/tree/main/examples/scripts

## References

- GitHub (OpenRLHF): https://github.com/OpenRLHF/OpenRLHF
- GitHub (veRL): https://github.com/volcengine/verl
- GitHub (TRL): https://github.com/huggingface/trl
- Paper (DeepSeek-R1 / GRPO): https://arxiv.org/abs/2402.03300
