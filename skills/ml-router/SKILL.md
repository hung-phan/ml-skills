---
name: ml-router
description: Use FIRST for any ML/DL task to pick the right ml-skills sub-skill (architectures, libraries, training, data-prep, GPU kernels). Routes by problem type, workflow stage, and model family — start here when the user's request is broad ("train a model", "speed up inference", "build a RAG system", "which architecture should I use") and the specific sub-skill isn't obvious.
---

# ML Router

Pick the right sub-skill from any of three angles: **by domain folder**, **by workflow stage**, or **by problem type**. When in doubt, start with the problem-type table — it's the densest mapping.

If the tables below don't surface what you need, **grep the library directly** — it's faster than reading every index.

## Grep the Library

The tables below cover the common cases. When you know a keyword (a paper name, an API, a library, a technique) but not which skill it lives in, search the markdown directly. Skills are plain `SKILL.md` files under `skills/<folder>/<topic>/`, so `grep` and `find` work exactly as you'd expect.

| You know… | Run |
|-----------|-----|
| A keyword or term (e.g. "FlashAttention", "BM25", "DPO") | `grep -rli "<keyword>" skills/` |
| An exact API or class name (case-sensitive) | `grep -rl "<symbol>" skills/` |
| A topic and want to see *every* skill that mentions it | `grep -rl "<topic>" skills/ \| sort` |
| The folder, want to list its sub-skills | `ls skills/<folder>/` |
| Want a one-line summary of every skill | `for f in skills/*/SKILL.md skills/*/*/SKILL.md; do echo "$f:"; grep -m1 "^description:" "$f"; done` |
| Want only sub-skill files (not folder indexes) | `find skills -mindepth 3 -name SKILL.md` |

Decision rule:
- **Workflow-stage tables** when you know *what step you're on* (e.g. "I'm picking metrics" → evaluation).
- **Problem-type tables** when you know *what kind of problem you have* (e.g. tabular forecasting).
- **`grep`** when you know *the keyword* but not which skill owns it (e.g. "where do we mention CUPED?").
- **Domain folders** when you want the full menu in a category.

Examples:

```bash
# Where do we cover speculative decoding?
grep -rli "speculative" skills/
# → skills/ml-training/inference-optimization/SKILL.md
# → skills/ml-architectures/sampling-strategies/SKILL.md

# Find every skill that touches GQA / KV-cache
grep -rl "GQA\|KV.cache" skills/

# Anything mentioning a specific paper / arXiv id
grep -rl "2201.11903" skills/   # CoT (Wei et al. 2022)

# Which skills point at vLLM?
grep -rl "vllm/" skills/

# What sub-skills exist under ml-architectures?
ls skills/ml-architectures/
```

If `grep` returns multiple hits for the same keyword, the **canonical home** is usually the deepest skill folder (e.g. for FlashAttention: `ml-architectures/attention/` is the deep treatment; other files cross-reference it). The `## See Also` block at the bottom of each `SKILL.md` confirms which one is canonical.

## Domain Folders

| Folder | Covers |
|--------|--------|
| `ml-architectures/` | Agents, AI App Architecture, ANN, Attention, Audio, Autoencoder, Boltzmann, CNN, Diffusion, Embeddings, GAN, GNN, LLM, Mamba, Mixture of Experts, Neural Combinatorial Optimization, Quantization, RAG, Regression/Classification, Reinforcement Learning, RNN, Sampling Strategies, SOM, Transformer, Vision, World Models (JEPA / Dreamer / MuZero) |
| `ml-libraries/` | pandas, numpy, polars, seaborn, plotly, keras, pytorch, scikit-learn, xgboost, huggingface, nemo, dspy, litellm, ray, vllm, sglang, triton-inference-server |
| `ml-training/` | feature-selection, training-workflow, evaluation, llm-evaluation, prompt-engineering, inference-optimization, model-merging, gradient-free-optimization, experiment-tracking, hf-jobs-workflow, online-experimentation, online-learning, data-parallel (DDP/FSDP), Unsloth SFT, Unsloth advanced, Ray distributed SFT, distributed GRPO |
| `data-prep/` | EDA, feature engineering, time-series features, dataset curation (FM data), data validation |
| `gpu-lang/` | Triton (block-level Python kernels), TileLang (shared memory + warp-level Python kernels) |
| `acquire-ml-skill/` | Meta-skill for creating a new skill or making a light single-file update. Requires critical thinking — placement, decision tables, and trigger phrases all need judgment, not boilerplate. |
| `refine-ml-skill/` | Meta-skill for deep-researching a topic and propagating updates across every skill it touches. Requires critical thinking — blast-radius mapping, picking a canonical home, and cross-file consistency are judgment calls. |

## By Workflow Stage

| Stage | Go to |
|-------|-------|
| 1. Understand the data | `data-prep/eda/` |
| 2. Validate/clean data | `data-prep/data-validation/` |
| 3. Engineer features (classical ML) | `data-prep/feature-engineering/` |
| 3b. Curate FM datasets (SFT, preference, CoT, RAG indexing) | `data-prep/dataset-curation/` |
| 4. Pick a model | `ml-architectures/SKILL.md` (decision tree at the bottom) |
| 5. Pick a library | `ml-libraries/SKILL.md` |
| 6. Train | `ml-training/training-workflow/` |
| 6b. Run training on managed cloud GPUs (HF Jobs) | `ml-training/hf-jobs-workflow/` |
| 6c. Black-box / non-differentiable optimization (HPO via CMA-ES, NAS, ES) | `ml-training/gradient-free-optimization/` |
| 7. Track experiments | `ml-training/experiment-tracking/` |
| 8. Evaluate (classical ML metrics) | `ml-training/evaluation/` |
| 8b. Evaluate an LLM, RAG, or agent (judges, faithfulness, benchmarks) | `ml-training/llm-evaluation/` |
| 9. Scale training across GPUs | `ml-training/data-parallel/` (DDP/FSDP) or `ml-libraries/ray/` |
| 10. Fine-tune an LLM | `ml-training/unsloth-sft/` → `ml-training/unsloth-advanced/` → `ml-training/distributed-grpo/` |
| 10b. Combine multiple finetuned checkpoints | `ml-training/model-merging/` |
| 11. Compress for inference | `ml-architectures/quantization/` |
| 11b. Optimize LLM inference latency / cost (engine, batching, spec-decoding, caching) | `ml-training/inference-optimization/` |
| 12. Serve | `ml-libraries/vllm/`, `ml-libraries/sglang/`, or `ml-libraries/triton-inference-server/` |
| 12a. Design / debug the prompt | `ml-training/prompt-engineering/` |
| 12b. Tune LLM decoding (temperature, top-p, min-p, structured output, self-consistency) | `ml-architectures/sampling-strategies/` |
| 12c. Build a RAG pipeline (chunk, retrieve, rerank, eval faithfulness) | `ml-architectures/rag/` |
| 12d. Build a tool-using / autonomous agent | `ml-architectures/agents/` |
| 12e. Architect the full LLM app (gateway, guardrails, caching, observability) | `ml-architectures/ai-app-architecture/` |
| 13. Ship to live traffic (A/B, bandits, canary) | `ml-training/online-experimentation/` |
| 14. Adapt to streaming data, drift detection, hot-swap weights | `ml-training/online-learning/` |
| 15. Write custom GPU kernels | `gpu-lang/triton/` or `gpu-lang/tilelang/` |
| 16. Solve combinatorial problems with neural heuristics (TSP/VRP/JSSP/MaxCut/SAT) | `ml-architectures/neural-combinatorial-optimization/` |

## By Problem Type

### Data modality

| Modality | Start with |
|----------|------------|
| Tabular (rows × columns) | `ml-architectures/regression-classification/` → `ml-libraries/xgboost/` or `ml-libraries/scikit-learn/` |
| Text — classification, NER, embeddings | `ml-libraries/huggingface/` + `ml-architectures/embeddings/` |
| Text — generation, chat, instruction | `ml-architectures/llm/` |
| Images — classification, detection, segmentation | `ml-architectures/vision/` (ViT, CLIP, SAM, YOLO) or `ml-architectures/cnn/` |
| Images — generation | `ml-architectures/diffusion/` (preferred) or `ml-architectures/gan/` |
| Audio — recognition (Whisper, spectrograms) | `ml-architectures/cnn/` (spectrograms) or `ml-architectures/transformer/` (Whisper-style) |
| Audio — generation (TTS, music, voice cloning) | `ml-architectures/audio/` |
| Time series — tabular forecasting (XGBoost/LightGBM) | `data-prep/time-series-features/` (lags, windows, calendar, purged CV) |
| Time series — short, sequential | `ml-architectures/rnn/` (LSTM/GRU) |
| Time series — long, parallel training | `ml-architectures/transformer/` |
| Time series — ultra-long (>8K), streaming | `ml-architectures/mamba/` |
| Graphs / networks | `ml-architectures/gnn/` |
| Multimodal (text + image) | `ml-architectures/vision/` (CLIP) + `ml-architectures/llm/` |
| Sequential decision making | `ml-architectures/reinforcement-learning/` |
| Self-supervised pretraining (no augmentations / negatives) | `ml-architectures/world-models/` (I-JEPA / V-JEPA) |
| Agent that plans inside a learned simulator | `ml-architectures/world-models/` (Dreamer V3, MuZero, TD-MPC2, DIAMOND) |

### LLM-specific

| Task | Go to |
|------|-------|
| Pick an LLM architecture (GPT/BERT/T5, RoPE, GQA, SwiGLU) | `ml-architectures/llm/` |
| Implement a transformer from scratch | `ml-architectures/transformer/` |
| Attention variants (MHA/MQA/GQA/MLA) and KV-cache math | `ml-architectures/attention/` |
| FlashAttention / PagedAttention | `ml-architectures/attention/` |
| Mixture of Experts (Switch, Mixtral, DeepSeek) | `ml-architectures/mixture-of-experts/` |
| Supervised fine-tuning (single GPU, fast) | `ml-training/unsloth-sft/` |
| LoRA / QLoRA / DoRA tuning | `ml-training/unsloth-sft/` → `ml-architectures/llm/` |
| DPO, ORPO, KTO preference tuning | `ml-training/unsloth-advanced/` |
| GRPO / RLHF at scale | `ml-training/distributed-grpo/` |
| Distributed SFT across many GPUs | `ml-training/ray-distributed-sft/` |
| Prompt programs, automated prompt optimization | `ml-libraries/dspy/` (programmatic) or `ml-training/prompt-engineering/` (hand-crafted) |
| Design / debug prompts; defend against injection | `ml-training/prompt-engineering/` |
| Tune decoding (temperature, top-p, min-p, structured, self-consistency) | `ml-architectures/sampling-strategies/` |
| Multi-provider LLM calls | `ml-libraries/litellm/` |
| High-throughput LLM serving | `ml-libraries/vllm/` or `ml-libraries/sglang/` |
| Speed up / reduce cost of LLM inference (engine choice, batching, spec-decoding, caching) | `ml-training/inference-optimization/` |
| Quantize an LLM (AWQ/GPTQ/FP8/GGUF) | `ml-architectures/quantization/` |
| RAG pipeline (chunk, retrieve+rerank, faithfulness eval) | `ml-architectures/rag/` |
| Just embeddings + vector search | `ml-architectures/embeddings/` |
| Tool-using / autonomous agent | `ml-architectures/agents/` |
| Architect a production LLM app (gateway, guardrails, caching, observability) | `ml-architectures/ai-app-architecture/` |
| Evaluate the LLM, RAG, or agent (judges, faithfulness, benchmarks) | `ml-training/llm-evaluation/` |
| Curate SFT / preference / RAG datasets for FMs | `data-prep/dataset-curation/` |
| Combine multiple finetuned checkpoints (TIES, DARE, SLERP) | `ml-training/model-merging/` |

### Generative

| Task | Go to |
|------|-------|
| Image generation (Stable Diffusion-style) | `ml-architectures/diffusion/` |
| Image generation (older / GAN-based) | `ml-architectures/gan/` |
| Text generation | `ml-architectures/llm/` |
| Speech synthesis (TTS) | `ml-architectures/audio/` (codec-LM, flow-matching) |
| Music / sound effects generation | `ml-architectures/audio/` (MusicGen, Stable Audio Open) |
| Voice cloning / voice conversion | `ml-architectures/audio/` (XTTS-v2, F5-TTS, RVC) |
| Unsupervised feature learning | `ml-architectures/autoencoder/` (VAE) or `ml-architectures/boltzmann/` |

### Training & scale

| Task | Go to |
|------|-------|
| Standard training loop | `ml-training/training-workflow/` |
| DDP / FSDP / pipeline parallelism | `ml-training/data-parallel/` |
| Distribute work with Ray | `ml-libraries/ray/` |
| Track runs (W&B, MLflow, TensorBoard) | `ml-training/experiment-tracking/` |
| Pick metrics, avoid leakage, calibrate (classical ML) | `ml-training/evaluation/` |
| Eval LLM, RAG, agent | `ml-training/llm-evaluation/` |
| Feature selection | `ml-training/feature-selection/` |
| Black-box / non-differentiable optimization (HPO via CMA-ES, NAS, ES, BayesOpt) | `ml-training/gradient-free-optimization/` |
| Combine multiple finetuned checkpoints | `ml-training/model-merging/` |

### Inference & deployment

| Task | Go to |
|------|-------|
| Serve LLMs at high throughput | `ml-libraries/vllm/` |
| Serve LLMs with structured / constrained output | `ml-libraries/sglang/` |
| Serve any model in production (REST/gRPC) | `ml-libraries/triton-inference-server/` |
| Reduce memory / latency | `ml-architectures/quantization/` + `ml-architectures/attention/` (GQA/MLA/FlashAttention) |
| Optimize end-to-end LLM serving (TTFT/TPOT, batching, spec-decoding, prefix cache) | `ml-training/inference-optimization/` |
| Architect the full LLM app (gateway, guardrails, caching, observability) | `ml-architectures/ai-app-architecture/` |
| A/B test, canary, multi-armed bandit a model rollout | `ml-training/online-experimentation/` |
| Hot-swap model weights, drift monitoring, online updates | `ml-training/online-learning/` |
| Custom CUDA kernel in Python | `gpu-lang/triton/` |
| Shared-memory / warp-level kernel | `gpu-lang/tilelang/` |

### Data work

| Task | Go to |
|------|-------|
| Explore a new dataset | `data-prep/eda/` + `ml-libraries/pandas/` (or `polars/`) |
| Visualize | `ml-libraries/seaborn/` (statistical) or `ml-libraries/plotly/` (interactive) |
| Validate schemas / detect drift | `data-prep/data-validation/` |
| Build features | `data-prep/feature-engineering/` |
| Numerical heavy lifting | `ml-libraries/numpy/` |
| Out-of-core / fast dataframes | `ml-libraries/polars/` |

### Classical ML / non-deep

| Task | Go to |
|------|-------|
| Tabular baselines, pipelines, cross-validation | `ml-libraries/scikit-learn/` |
| Gradient boosting on tabular | `ml-libraries/xgboost/` |
| Loss functions, metrics, class imbalance, calibration | `ml-architectures/regression-classification/` |
| Clustering / topology visualization | `ml-architectures/som/` |

## Architecture Decision Tree

For "which model architecture?" jump straight to `ml-architectures/SKILL.md` — it has the full decision table. Quick version:

| If the input is… | And you need… | Use |
|------------------|---------------|-----|
| Tabular | Strong baseline | XGBoost (`ml-libraries/xgboost/`) |
| Tabular | Deep model | MLP (`ml-architectures/ann/`) |
| Sequence, short | Quick prototype | LSTM/GRU (`ml-architectures/rnn/`) |
| Sequence, long | Parallel training | Transformer (`ml-architectures/transformer/`) |
| Sequence, very long | Linear-time scan | Mamba (`ml-architectures/mamba/`) |
| Image | Classification / detection | CNN or ViT (`ml-architectures/vision/`) |
| Image | Generation | Diffusion (`ml-architectures/diffusion/`) |
| Graph | Node/edge tasks | GNN (`ml-architectures/gnn/`) |
| Text | Generation | LLM (`ml-architectures/llm/`) |
| Audio (text/voice/music) | Generation | Audio (`ml-architectures/audio/`) |
| Anything | Scale params without compute | MoE (`ml-architectures/mixture-of-experts/`) |
| State-action env | Decision policy | RL (`ml-architectures/reinforcement-learning/`) |
| Image / video | SSL pretraining without augmentations | World Models — JEPA (`ml-architectures/world-models/`) |
| State-action env | Plan in a learned simulator | World Models — Dreamer / MuZero / TD-MPC2 / DIAMOND (`ml-architectures/world-models/`) |

## Contributing to this library

Both contributor skills require **critical thinking, not just template-filling**. Don't reach for them expecting a checklist that produces a finished skill — they encode judgment about placement, scope, blast radius, and cross-file consistency that you have to actually exercise.

| Task | Go to |
|------|-------|
| Add a new skill (single file, deliberate placement) | `acquire-ml-skill/` |
| Light edit to one existing skill | `acquire-ml-skill/` (update workflow) |
| Deep-research a topic and propagate updates across all related skills | `refine-ml-skill/` |
| Audit / refresh coverage of an area after a major release | `refine-ml-skill/` |
| Restructure the router itself | edit `ml-router/SKILL.md` directly |

Rule of thumb: **one file in scope → `acquire-ml-skill`. Two or more files in scope, or stale coverage suspected → `refine-ml-skill`.**
