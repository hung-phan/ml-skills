# ML Skills

A Claude Code plugin with a comprehensive ML skills library — reference guides with working code, decision tables, and "why this exists" context for every major ML topic.

**Author**: [hung-phan](https://github.com/hung-phan)

## Install

Register as a plugin marketplace, then install:

```
/plugin marketplace add hung-phan/ml-skills
/plugin install ml-skills@ml-skills
```

### Local install (for development)

```bash
claude plugin validate .claude-plugin/plugin.json
claude plugin add /path/to/ml-skills
```

## How to Use

Skills are invoked as slash commands. **The fastest way: ask `/ml-router` first whenever you're unsure.**

### Start here: the router

```
/ml-router I need to fine-tune Llama-3 on a single GPU with limited VRAM
/ml-router my XGBoost model passes CV but fails in production — what do I check?
/ml-router I have a Kafka stream and want to build features for a fraud model
```

`/ml-router` reads your question and points you to the right sub-skill. Reach for it whenever the right skill isn't obvious.

### Jump straight to a skill when you know what you want

| You want to... | Invoke |
|----------------|--------|
| Pick a model architecture (CNN vs Transformer vs Mamba vs Diffusion...) | `/ml-architectures` |
| Use a specific library (PyTorch, HuggingFace, vLLM, Ray, polars...) | `/ml-libraries` |
| Set up training, eval, fine-tuning, distributed training | `/ml-training` |
| Explore, validate, or feature-engineer data | `/data-prep` |
| Write custom GPU kernels (Triton, TileLang) | `/gpu-lang` |

Each top-level skill is itself a router — invoking `/ml-architectures` returns a decision table that points to the specific sub-skill (`attention/`, `transformer/`, `diffusion/`, ...). Sub-skills can also be invoked by their nested name from inside the parent skill.

### Three usage patterns

1. **Broad question, unsure where to start** → `/ml-router <your question>`. The router routes you.
2. **You know the area** → `/ml-architectures`, `/ml-libraries`, `/ml-training`, `/data-prep`, or `/gpu-lang`. The skill's index points you to the right sub-skill.
3. **You know the exact topic** → ask Claude directly ("show me the attention skill", "how do I do CUPED variance reduction?"). The relevant skill activates automatically because skill descriptions match the trigger phrases.

### Workflow examples

**End-to-end training a tabular model:**
```
/data-prep         → EDA, feature engineering, validation
/ml-training       → split strategy, CV, Optuna, tracking
                     (then: evaluation, online-experimentation when shipping)
```

**Fine-tuning an LLM:**
```
/data-prep          → dataset-curation (curate SFT / preference / CoT data)
/ml-libraries       → HuggingFace, PEFT
/ml-training        → unsloth-sft → unsloth-advanced (DPO/GRPO) → distributed-grpo
/ml-training        → llm-evaluation (build eval scorecard, judge, hallucination check)
/ml-training        → model-merging (combine specialist checkpoints — TIES/DARE/SLERP)
/ml-architectures   → quantization (for inference)
/ml-training        → inference-optimization (TTFT/TPOT, spec-decoding, prefix cache)
/ml-libraries       → vllm or sglang (for serving)
```

**Building an LLM application (RAG / agent / production):**
```
/ml-training        → prompt-engineering (chat templates, CoT, injection defenses)
/ml-architectures   → rag (chunking, BM25/dense/hybrid, rerankers, faithfulness eval)
/ml-architectures   → agents (tool use, ReAct, function calling, eval benchmarks)
/ml-architectures   → sampling-strategies (temperature, top-p, structured generation)
/ml-architectures   → ai-app-architecture (gateway, guardrails, caching, observability)
/ml-training        → llm-evaluation (judges, faithfulness, regression-test prompts)
```

**Building a forecasting pipeline:**
```
/ml-router "tabular forecasting with covariates"
  → time-series-features (or Chronos/AutoGluon-TS for zero-shot)
  → training-workflow (purged + embargoed CV)
  → online-learning (drift detection in production)
```

**Shipping a model to live traffic:**
```
/ml-training/online-experimentation   → A/B, CUPED, sequential testing, bandits
/ml-training/online-learning          → drift detection, hot-swap, train/score separation
```

## What's Included

| Folder | Skills |
|--------|--------|
| `ml-router/` | Top-level routing index — start here for any ML/DL task |
| `ml-architectures/` | Agents, AI App Architecture, ANN, Attention, Audio, Autoencoder, Boltzmann, CNN, Diffusion, Embeddings, GAN, GNN, LLM, Mamba, MoE, **Neural Combinatorial Optimization**, Quantization, **RAG**, Regression/Classification, Reinforcement Learning, RNN, **Sampling Strategies**, SOM, Transformer, Vision, World Models |
| `ml-libraries/` | PyTorch, HuggingFace, scikit-learn, XGBoost, pandas, polars, numpy, Ray, NeMo, DSPy, LiteLLM, vLLM, SGLang, Triton Inference Server, keras, seaborn, plotly |
| `ml-training/` | feature-selection, training-workflow, **evaluation** (classical) + **llm-evaluation** (FM-specific), **prompt-engineering**, **inference-optimization**, **model-merging**, **gradient-free-optimization**, experiment-tracking, hf-jobs-workflow, **online-experimentation** (A/B, CUPED, bandits), **online-learning** (drift, incremental updates), data-parallel (DDP/FSDP), Unsloth SFT, Unsloth advanced (GRPO/DPO), Ray distributed SFT, distributed GRPO |
| `data-prep/` | EDA, feature engineering, **time-series-features** (lags, windows, point-process, purged CV), **dataset-curation** (FM data: SFT, preference, CoT, synthesis, dedup), data validation |
| `gpu-lang/` | Triton, TileLang |

> **Maintainer guidelines** (not runtime skills) live in `docs/`:
> - [`docs/acquire-ml-skill.md`](docs/acquire-ml-skill.md) — add a new skill or update one file
> - [`docs/refine-ml-skill.md`](docs/refine-ml-skill.md) — deep-research + propagate updates across multiple skills

## Skill Format

Every skill follows a consistent format:
- **Why This Exists** — the problem it solves and when to reach for it
- **Code examples** — PyTorch and/or sklearn, realistic and runnable
- **Decision tables** — when to use this vs alternatives
- **References** — verified links to docs, papers, and repos
- **See Also** — cross-references to adjacent skills so you can navigate the library by topic

## Contributing

Maintainer guidelines live in `docs/` (not under `skills/`, so they don't burn context tokens for end users). Read the relevant one before editing:

| Task | Read |
|------|------|
| Add one new skill or lightly edit one file | [`docs/acquire-ml-skill.md`](docs/acquire-ml-skill.md) |
| Refresh a topic that spans multiple skills (e.g. a new attention variant touches `attention/`, `llm/`, `vllm/`) | [`docs/refine-ml-skill.md`](docs/refine-ml-skill.md) |
| Audit coverage of an area after a major release | [`docs/refine-ml-skill.md`](docs/refine-ml-skill.md) |
| Restructure the router itself | edit `skills/ml-router/SKILL.md` directly |

These docs encode the quality standards, folder conventions, intake questions, and formatting rules. Hand them to Claude as the spec for the change you want — e.g. "follow `docs/acquire-ml-skill.md` to add a skill on rotary positional embeddings (RoPE)".
