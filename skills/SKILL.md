---
name: ml-skills
description: Comprehensive ML skills library — architectures, libraries, training, inference, GPU kernels, and data prep. Use as an index to find the right skill for any ML/DL task.
---

# ML Skills Library

## Structure

| Folder | Covers |
|--------|--------|
| `ml-architectures/` | Attention, ANN, CNN, RNN, Transformer, Mamba, MoE, GAN, Diffusion, GNN, LLM, Vision, RL, SOM, Autoencoder, Boltzmann, Regression |
| `ml-libraries/` | pandas, numpy, polars, seaborn, plotly, keras, pytorch, scikit-learn, xgboost, huggingface, dspy, litellm, ray, vllm, sglang, triton-inference-server |
| `ml-training/` | Feature selection, training loops, evaluation, data-parallel, Unsloth, Ray distributed SFT, distributed GRPO |
| `data-prep/` | EDA, feature engineering, data validation |
| `gpu-lang/` | Triton (block-level Python kernels), TileLang (shared memory + warp-level Python kernels) |
| `acquire-ml-skill/` | Meta-skill for creating or updating skills in this library |

## Quick Decision

| Task | Go to |
|------|-------|
| Which architecture for my problem? | `ml-architectures/SKILL.md` |
| Attention variants, KV-cache, FlashAttention | `ml-architectures/attention/SKILL.md` |
| How to use a specific library? | `ml-libraries/<lib>/SKILL.md` |
| Train a model (classical or LLM) | `ml-training/SKILL.md` |
| Serve a model in production | `ml-libraries/vllm/` or `sglang/` or `triton-inference-server/` |
| Explore raw data, create features, validate | `data-prep/SKILL.md` |
| Write custom GPU kernels in Python | `gpu-lang/SKILL.md` |
| Distribute work across GPUs with Ray | `ml-libraries/ray/SKILL.md` |
| HuggingFace transformers/PEFT/datasets | `ml-libraries/huggingface/SKILL.md` |
| DDP/FSDP/Pipeline parallelism | `ml-training/data-parallel/SKILL.md` |
| Add or update a skill | `acquire-ml-skill/SKILL.md` |
