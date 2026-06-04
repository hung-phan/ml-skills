---
name: nemo
description: NVIDIA NeMo — opinionated NVIDIA-stack training framework for LLMs, multimodal, and speech AI. Built on Megatron-Core for native 5D parallelism (TP × PP × CP × EP × DP), FP8/MXFP8, MoE Parallel Folding, with verified configs for Llama / Qwen / DeepSeek-V3 and a deployment path through TensorRT-LLM / vLLM / NIM. Use when scaling pretraining or post-training across many NVIDIA GPUs (multi-node SLURM/K8s), training MoE models, needing FP8 on Hopper/Blackwell, or shipping to NIM/TRT-LLM.
---

# NVIDIA NeMo Framework

- **Docs**: https://docs.nvidia.com/nemo-framework/user-guide/latest/
- **GitHub org**: https://github.com/NVIDIA-NeMo
- **Megatron-Core**: https://developer.nvidia.com/megatron-core
- **NGC container**: `nvcr.io/nvidia/nemo:<release-tag>` (e.g. `25.07`, `25.09`, `25.11`)

## Why This Exists

**Problem**: Training a 70B-parameter dense LLM or a 671B-parameter MoE across hundreds-to-thousands of GPUs requires composing **tensor + pipeline + context + expert + data parallelism**, FP8/MXFP8 mixed precision, communication overlap, activation recomputation, and CPU offload — all tuned together. Off-the-shelf wrappers (`accelerate`, `Trainer`, plain FSDP) don't expose these axes; gluing them yourself with Megatron-LM + DeepSpeed + custom launchers means re-deriving NVIDIA's tuning every time.

**Key insight**: NeMo is the **opinionated NVIDIA-stack training framework**. It bundles Megatron-Core's 5D parallelism behind a PyTorch Lightning `MegatronStrategy`, ships **verified production configs** (Llama-3 70B = TP4·PP4·CP2 on 64 GPUs; DeepSeek-V3 671B = TP2·PP16·EP64 on 1024 GPUs), and connects training → checkpoint conversion → TensorRT-LLM/NIM deployment in one stack. You inherit NVIDIA's recipes instead of re-tuning them.

**Reach for this when**:
- Pretraining / continued pretraining at **multi-node** scale (8+ GPUs across nodes, often 100s–1000s)
- **MoE** models (DeepSeek-V3, Qwen-MoE, Mixtral) where Expert Parallelism + MoE Parallel Folding matter
- **FP8 / MXFP8** on Hopper (H100/H200) or Blackwell (B100/B200/GB200)
- **Long-context** training requiring Context Parallelism (CP)
- Shipping to **NVIDIA NIM** or **TensorRT-LLM** in production
- You're locked into the NVIDIA stack (CUDA + TransformerEngine + Megatron-Core) and want batteries-included rather than glue code

**Don't reach for this when**:
- Single-GPU SFT / LoRA — use **Unsloth** (`ml-training/unsloth-sft/`)
- Heterogeneous workloads (training + RL rollouts + eval in one program), vendor-portable code, or you want autoscaling clusters — use **Ray Train** (`ml-libraries/ray/`, `ml-training/ray-distributed-sft/`)
- Standard HF model + dense training that fits in ZeRO-3 — **DeepSpeed + accelerate** is simpler
- Quick fine-tuning where any HF SFT recipe works — **TRL / HF Trainer**

## 2026 Ecosystem Reality (read this before installing)

NeMo is **mid-reorganization**. As of 2026:

- **NeMo 1.x is deprecated** since release `25.04`. The YAML/Hydra config style is going away.
- The monolithic `NVIDIA/NeMo` repo **pivoted** to focus on **audio, speech, and multimodal LLM**. v2.7.0 was the last release with broader collections; v2.8.0 removed `llm/`, `nlp/`, `vision/`, `vlm/`, `multimodal/`, `diffusion/`, `speechlm/`.
- LLM training, post-training, and inference are now spread across the **`NVIDIA-NeMo` GitHub org**:

| Repo | Purpose |
|------|---------|
| [`NVIDIA-NeMo/Run`](https://github.com/NVIDIA-NeMo/Run) | Launcher — local / SLURM / Kubernetes (replaces NeMo 1.x launcher scripts) |
| [`NVIDIA-NeMo/Megatron-Bridge`](https://github.com/NVIDIA-NeMo/Megatron-Bridge) | Bidirectional HF ↔ Megatron-Core checkpoint conversion + verified pretraining recipes |
| [`NVIDIA-NeMo/Automodel`](https://github.com/NVIDIA-NeMo/Automodel) | Fine-tune any HF LLM/VLM, save **HF-native checkpoints** (no conversion) |
| [`NVIDIA-NeMo/RL`](https://github.com/NVIDIA-NeMo/RL) | Alignment — DPO, RLHF, REINFORCE (replacing NeMo-Aligner) |
| [`NVIDIA-NeMo/Export-Deploy`](https://github.com/NVIDIA-NeMo/Export-Deploy) | Export to TensorRT-LLM / vLLM / Triton / Ray Serve / NIM |
| [`NVIDIA-NeMo/Evaluator`](https://github.com/NVIDIA-NeMo/Evaluator) | Benchmark + eval harness |
| [`NVIDIA-NeMo/Curator`](https://github.com/NVIDIA-NeMo/Curator) | Data curation / dedup at scale |
| [`NVIDIA-NeMo/NeMo`](https://github.com/NVIDIA-NeMo/NeMo) | Speech + audio + multimodal-LLM only (post v2.8) |

**Pragmatic install**: use the NGC container (`nvcr.io/nvidia/nemo:<release-tag>`) which bundles Megatron-Core, TransformerEngine, and the right CUDA / cuDNN / NCCL versions. Picking individual pip packages requires matching CUDA/PyTorch/TE versions exactly — painful outside the container.

## Architecture

```
                    ┌──────────────────────────────────┐
                    │   NeMo Run (SLURM / K8s / local) │   ← launcher
                    └────────────────┬─────────────────┘
                                     │
                  ┌──────────────────┼─────────────────┐
                  │                  │                 │
            ┌─────▼─────┐    ┌───────▼──────┐    ┌─────▼─────┐
            │ AutoModel │    │ Megatron-    │    │ NeMo (v2) │
            │ (HF SFT/  │    │ Bridge       │    │ collections│
            │  PEFT)    │    │ (HF↔MCore)   │    │ (speech)   │
            └─────┬─────┘    └───────┬──────┘    └─────┬──────┘
                  │                  │                 │
                  └──────────┬───────┴─────────────────┘
                             │
                ┌────────────▼─────────────┐
                │   PyTorch Lightning      │
                │   MegatronStrategy       │  ← parallelism config lives here
                └────────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │   Megatron-Core          │  ← TP/PP/CP/EP/DP, FP8/MXFP8,
                │   (transformer-engine)    │     MoE, comm overlap
                └──────────────────────────┘
                             │
                ┌────────────▼─────────────┐
                │   Export-Deploy          │  ← TRT-LLM / vLLM / Triton /
                │                          │     Ray Serve / NIM
                └──────────────────────────┘
```

## Parallelism — The 5D Formula

```
Total GPUs = TP × PP × CP × EP × DP
```

| Axis | What it splits | When you need it |
|------|---------------|-------------------|
| **TP** (Tensor) | Each linear layer's weight matrix across GPUs in a node | Model layer doesn't fit on one GPU; intra-node high-bandwidth NVLink |
| **PP** (Pipeline) | Layers across nodes; supports **virtual / interleaved** schedule to reduce bubble | Whole model doesn't fit on one node |
| **CP** (Context) | Long sequence dimension across GPUs | Context length > what fits in attention memory (32K+) |
| **EP** (Expert) | MoE experts across GPUs | Mixture-of-Experts models — separate from dense parallelism via **MoE Parallel Folding** |
| **DP** (Data) | Mini-batches across replicas (DDP or FSDP underneath) | Always — fills remaining GPUs after model dims fixed |
| **SP** (Sequence) | Activation memory along sequence dim within TP | Free with TP; usually enable always |

**MoE Parallel Folding** decouples Attention parallelism `(TP × CP × DP × PP)` from MoE parallelism `(ETP × EP × EDP × PP)` so each can use a different topology. Critical for DeepSeek-V3 / Qwen3-235B-A22B style models.

**Verified production configs** (sourced from NVIDIA NeMo runs, surfaced in Megatron-Core docs):

| Model | GPUs | TP | PP | CP | EP | DP |
|-------|-----:|---:|---:|---:|---:|---:|
| Llama-3 70B | 64 | 4 | 4 | 2 | 1 | 2 |
| DeepSeek-V3 671B | 1024 | 2 | 16 | 1 | 64 | 1 |

The **Auto Configurator** searches over `(TP, PP, CP, EP, MBS, ActCkpt)` to find optimal throughput for a given (model, GPU count, memory) — start there before hand-tuning.

## FSDP Variants

NeMo supports several FSDP implementations; pick by workload shape, not preference:

| Variant | When to choose |
|---------|----------------|
| **PyTorch FSDP1** (legacy) | Existing FSDP1 codepaths; not recommended for new work |
| **PyTorch FSDP2** | Vanilla HF models via AutoModel; AutoModel uses this by default |
| **Custom FSDP** | GB200 NVL72 — designed to fully utilize NVL72 fabric |
| **Megatron FSDP** (was `nvFSDP` / packaged as `megatron-fsdp` from `25.07`) | Megatron-Core models when you want sharding instead of 3D parallelism. Migrating from `mcore FSDP` since `25.11` |

**Rule of thumb**: prefer FSDP over 3D parallelism when (a) workloads are unbalanced across pipeline stages, (b) vocabulary is huge, or (c) you want to skip the TP/PP/DP search. Use 3D + Megatron-Core when you need maximum throughput and have time to tune.

## Customization Workflows

| Workflow | Repo | Notes |
|----------|------|-------|
| **Pretraining** | `Megatron-Bridge` | Verified scripts for latest models; converts HF → Megatron checkpoint, trains, exports back |
| **SFT** (full fine-tune) | `Automodel` (HF-native output) **or** `Megatron-Bridge` (Megatron format) | AutoModel saves to `model/consolidated/` directly usable by Transformers/vLLM/lm-eval-harness |
| **PEFT** (LoRA, P-tuning, canonical+performant LoRA, DoRA-class) | `Automodel` | PEFT-library-compatible adapter weights — push directly to HF Hub |
| **Knowledge Distillation** | `Megatron-Bridge` | Supported as part of SFT recipes |
| **Alignment** (DPO, RLHF, REINFORCE, RPO, IPO, Rejection Sampling, SPIN, SteerLM/2.0, DRaFT+, Constitutional AI) | `RL` (replacing `NeMo-Aligner`) | Broad post-training suite |
| **Quantization / Pruning** | `Megatron-Bridge` + Model Optimizer | INT8/FP8/INT4; pairs with TRT-LLM export |

**AutoModel's selling point**: SFT and LoRA outputs land in **native HF format with no conversion** — direct upload to HF Hub, immediately loadable in Transformers / PEFT / vLLM / lm-eval-harness.

## Supported Model Families (2025–2026)

| Modality | Models |
|----------|--------|
| **Dense LLM** | Llama 3 / 3.2 / 3.3 / 4, Qwen 2 / 2.5 / 3, Nemotron (incl. Llama Nemotron Ultra/Super/Nano), Gemma 2 / 3, Mamba |
| **MoE LLM** | DeepSeek-V3 (with DeepEP fast path since `25.09`), Mixtral, Qwen2-57B-A14B, Qwen3-30B-A3B, Qwen3-235B-A22B |
| **Vision-Language** | NeVA, LLaVA-Next, Qwen2-VL, AVLM, CLIP |
| **Image generation** | FLUX, Stable Diffusion (SDXL), Imagen |
| **ASR** | Canary 1.1 / 1b-v2, Parakeet (incl. `parakeet-tdt-0.6b-v3`, large) |
| **Speaker / Diarization** | Sortformer, Streaming Sortformer |
| **TTS** | Magpie-TTS |
| **Multimodal SpeechLM** | SpeechLM2 |

## Deployment / Export Paths

| Target | Mechanism | Notes |
|--------|-----------|-------|
| **TensorRT-LLM** | Direct export from NeMo 2.0; **PyTorch backend** added in `25.07` | Best NVIDIA-native latency |
| **vLLM** | NeMo → HF → vLLM (since `25.09`); **vLLM V1** since `25.04.01` | Use when you want vLLM's batching + LoRA hot-swap |
| **Triton Inference Server** | Megatron-LM and Megatron-Bridge models supported since `25.09` | Multi-framework serving |
| **Ray Serve** | Multi-instance deployment since `25.07` | Pair with Ray Train for end-to-end Ray pipelines |
| **NVIDIA NIM** | NeMo 2.0 → NIM export path (since `24.12`, expanded `25.04.00`) | Productionizes a model as a microservice with OpenAI-compatible API |

For NIM specifically: NeMo trains → exports a NIM-compatible artifact → NIM serves it as a containerized microservice. This is the canonical NVIDIA "internal foundation model" deployment story.

## Code Sketches

### Minimal pretraining recipe (NeMo 2.0 + Megatron-Bridge style)

```python
# Inside the NGC container: nvcr.io/nvidia/nemo:25.11
import nemo_run as run
from megatron.bridge.recipes.llama import llama3_70b
from megatron.bridge import AutoBridge

# Pull a verified pretraining recipe — TP/PP/CP already tuned
recipe = llama3_70b.pretrain_recipe(
    name="llama3-70b-pretrain",
    num_nodes=8,            # 8 nodes × 8 GPUs = 64 GPUs
    num_gpus_per_node=8,
)

# Override what you need
recipe.trainer.strategy.tensor_model_parallel_size = 4
recipe.trainer.strategy.pipeline_model_parallel_size = 4
recipe.trainer.strategy.context_parallel_size = 2
recipe.trainer.strategy.sequence_parallel = True
recipe.trainer.precision = "bf16-mixed"  # or "fp8-mixed" on Hopper+

# Launch on SLURM
executor = run.SlurmExecutor(
    account="your-account",
    partition="batch",
    nodes=8,
    ntasks_per_node=8,
    gpus_per_node=8,
    time="06:00:00",
    container_image="nvcr.io/nvidia/nemo:25.11",
)

run.run(recipe, executor=executor)
```

### LoRA fine-tune (AutoModel — HF-native output)

```python
# pip install nemo-automodel  (or use NGC container)
from nemo_automodel.recipes.llm.peft import lora_finetune

lora_finetune(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset="tatsu-lab/alpaca",
    output_dir="./llama-3.1-8b-alpaca-lora",
    lora_rank=16,
    lora_alpha=32,
    lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    learning_rate=2e-4,
    num_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    fsdp="fsdp2",           # or "ddp" on a single GPU
)

# Output: ./llama-3.1-8b-alpaca-lora/  — adapter is PEFT-library compatible.
# Load directly with `peft.PeftModel.from_pretrained(...)` or push to HF Hub.
```

### HF → Megatron checkpoint conversion (Megatron-Bridge)

```python
from megatron.bridge import AutoBridge

# Import an HF checkpoint into Megatron format for distributed training
AutoBridge.import_ckpt(
    hf_model_id_or_path="meta-llama/Llama-3.1-70B",
    target_path="./megatron-ckpts/llama3.1-70b",
)

# After training, export back to HF for downstream use
AutoBridge.export_ckpt(
    megatron_ckpt_path="./trained-megatron-ckpt",
    target_path="./hf-export/my-llama3.1-70b-tuned",
)
```

### Export to TensorRT-LLM

```python
from nemo.export.tensorrt_llm import TensorRTLLM

trt = TensorRTLLM(model_dir="./trt-engine-out")
trt.export(
    nemo_checkpoint_path="./trained-megatron-ckpt",
    model_type="llama",
    n_gpus=2,                # tensor parallel for inference
    dtype="bf16",            # or "fp8"
    max_input_len=4096,
    max_output_len=1024,
    max_batch_size=64,
)

# Now serve via Triton + TRT-LLM backend, or load via NIM.
```

## Decision Table — When to Pick NeMo

| Scenario | Pick this |
|----------|-----------|
| Single GPU SFT/LoRA on Llama/Qwen, optimize for speed | **Unsloth** (`ml-training/unsloth-sft/`) |
| HF Trainer / TRL works fine, dense model fits in ZeRO-3 | **DeepSpeed + accelerate** |
| Multi-framework distributed compute (training + RL rollouts + data ingestion + autoscaling) | **Ray Train / Ray Serve** (`ml-libraries/ray/`) |
| Vendor-portable code (NVIDIA + AMD + TPU) | **PyTorch Lightning + FSDP** (no NeMo) |
| Pretraining a 70B+ dense model across multi-node NVIDIA cluster | **NeMo + Megatron-Bridge** |
| MoE training (DeepSeek-V3, Qwen-MoE, Mixtral) at scale | **NeMo** (Megatron-Core MoE + Parallel Folding) |
| FP8 / MXFP8 on Hopper / Blackwell with TP comm overlap | **NeMo** (best-in-class) |
| Production deploy to NVIDIA NIM | **NeMo → Export-Deploy → NIM** |
| Long-context training (32K+) | **NeMo** with Context Parallelism |
| Fine-tune any HF model, want HF-native output | **NeMo AutoModel** (or just use HF + Unsloth/TRL if scale fits) |

### NeMo vs Ray Train (the most common confusion)

They operate at **different layers** and **compose** rather than compete:

| Dimension | Ray Train | NeMo |
|-----------|-----------|------|
| What it is | Distributed compute orchestrator | Generative AI training framework |
| Native model parallelism | None — wraps FSDP / DeepSpeed / Megatron | TP×PP×CP×EP×DP via Megatron-Core |
| Verified configs for big models | None — you tune | Yes (Llama-3 70B, DeepSeek-V3 671B, etc.) |
| MoE Parallel Folding | Build it yourself | Native |
| FP8 + TP comm overlap | Whatever you wrap | First-class |
| Hardware | NVIDIA / AMD / TPU / CPU | NVIDIA only |
| Launcher | Ray cluster (autoscaling) | NeMo Run (SLURM / K8s) |
| Composes with the other? | Yes — run NeMo via Ray, export to Ray Serve | Yes — see `ml-training/ray-distributed-sft/` for the inverse |

**Mental model**: pick Ray when you want flexibility and vendor portability; pick NeMo when you want NVIDIA's pre-tuned recipes and FP8/MoE batteries-included.

## Common Pitfalls

1. **Picking NeMo 1.x docs by accident** — anything before `25.04` is the deprecated path. URLs like `docs.nvidia.com/nemo-framework/user-guide/24.09/...` may show YAML/Hydra recipes that don't apply to NeMo 2.0. Always check the release tag in the URL.

2. **Container/CUDA mismatch** — NeMo, Megatron-Core, and TransformerEngine are tightly coupled. Pip-installing them in a fresh venv almost always fails on CUDA / Apex / TE versions. Use the **NGC container** (`nvcr.io/nvidia/nemo:<tag>`) unless you have a specific reason not to.

3. **`Total GPUs ≠ TP × PP × CP × EP × DP`** — silent mis-configuration; jobs hang or OOM. Always assert this multiplicatively in your launcher.

4. **MoE without Parallel Folding** — using a single TP/PP/EP/DP topology for both attention and MoE is suboptimal at scale. Configure folding explicitly when training MoE on 100+ GPUs.

5. **FP8 convergence surprises** — FP8 / MXFP8 can shift loss curves; validate against bf16 baseline before scaling up. Recipe defaults assume Hopper or Blackwell; on Ampere fall back to bf16.

6. **NeMo-Aligner ↔ NVIDIA-NeMo/RL migration** — alignment workflows are moving from `NVIDIA/NeMo-Aligner` to `NVIDIA-NeMo/RL`. Pin to a release tag rather than `main`.

7. **Speech/multimodal users on the wrong repo** — after v2.8, speech / audio / multimodal LLM stay in `NVIDIA-NeMo/NeMo`. LLM-only users should look in `Megatron-Bridge` / `AutoModel` instead.

8. **Checkpoint format drift** — `Megatron-Bridge` (HF↔Megatron), `AutoModel` (HF-native), and Megatron-Core native checkpoints are three different formats. Use the bridge for conversion; don't hand-edit.

9. **NeMo Run on K8s requires cluster setup** — you need a GPU operator + RDMA / NCCL networking configured. Easier to start on SLURM if you have it.

10. **Auto Configurator runtime** — search over (TP, PP, CP, EP, MBS, ActCkpt) is itself a small training job. Budget time for the search; cache results per (model, GPU count) tuple.

## Installation (NGC container — recommended)

```bash
# Pull the latest NeMo container (2026-Q2 example tag)
docker pull nvcr.io/nvidia/nemo:25.11

# Interactive run on a single node, all GPUs
docker run --gpus all --ipc=host --net=host \
    -v $HOME/data:/workspace/data \
    -v $HOME/checkpoints:/workspace/checkpoints \
    -it nvcr.io/nvidia/nemo:25.11

# Inside the container, all of Megatron-Core, TransformerEngine, NeMo Run,
# Megatron-Bridge, AutoModel are pre-installed at compatible versions.
```

For multi-node SLURM, drive the same image via `enroot` / `pyxis` and a NeMo Run `SlurmExecutor`. For Kubernetes, use Run's `KubernetesExecutor` plus the NVIDIA GPU Operator.

## See Also

- `ml-libraries/ray/` — Ray Train as the alternative orchestration story
- `ml-training/ray-distributed-sft/` — running TRL fine-tuning on Ray (compare to AutoModel)
- `ml-training/unsloth-sft/` — single-GPU SFT alternative
- `ml-training/distributed-grpo/` — RLHF / GRPO at scale (compare to NVIDIA-NeMo/RL)
- `ml-training/data-parallel/` — DDP / FSDP background
- `ml-libraries/vllm/`, `ml-libraries/triton-inference-server/` — downstream serving targets

## References

### Primary (NVIDIA)

- **NeMo Framework user guide (latest)**: https://docs.nvidia.com/nemo-framework/user-guide/latest/
- **NeMo overview (25.02)**: https://docs.nvidia.com/nemo-framework/user-guide/25.02/overview.html
- **Changelog (latest)**: https://docs.nvidia.com/nemo-framework/user-guide/latest/changelog.html
- **NeMo 2.0 introduction (24.09)**: https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemo-2.0/index.html
- **Parallelisms feature page (24.12)**: https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/features/parallelisms.html
- **MegatronStrategy (25.07)**: https://docs.nvidia.com/nemo-framework/user-guide/25.07/nemo-2.0/features/megatron.html
- **AutoModel LLM fine-tuning guide**: https://docs.nvidia.com/nemo/automodel/latest/guides/llm/finetune.html
- **Megatron-Core developer page**: https://developer.nvidia.com/megatron-core
- **Megatron-Core parallelism guide**: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
- **Megatron-Core MoE guide (0.15)**: https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/moe.html
- **NIM deployment from NeMo (25.07)**: https://docs.nvidia.com/nemo-framework/user-guide/25.07/deployment/llm/nemo_models/nim.html

### GitHub

- **NVIDIA-NeMo org** (post-2026 split): https://github.com/NVIDIA-NeMo
- **NeMo (speech/multimodal)**: https://github.com/NVIDIA-NeMo/NeMo
- **NeMo-Run launcher**: https://github.com/NVIDIA-NeMo/Run
- **Megatron-Bridge**: https://github.com/NVIDIA-NeMo/Megatron-Bridge
- **AutoModel**: https://github.com/NVIDIA-NeMo/Automodel
- **Megatron-LM (upstream)**: https://github.com/NVIDIA/Megatron-LM

### Papers

- **MoE Parallel Folding** (Liu et al., 2025): https://arxiv.org/abs/2504.14960
