# ML Skills

A Claude Code plugin that ships **one** skill — `ml-review` — and a curated reference library of ~80 ML topics. The skill routes broad ML/DL questions, reviews ML approaches, analyzes pipelines, and suggests solutions; the references hold the deep treatment of each topic and are read on demand by the skill.

**Author**: [hung-phan](https://github.com/hung-phan)

## Install

Register as a plugin marketplace, then install:

```
/plugin marketplace add hung-phan/ml-skills
/plugin install ml-skills@ml-skills
```

### Local install (for development)

```bash
claude plugin marketplace add ./
claude plugin install ml-skills@ml-skills
```

## How to Use

There is one skill: `/ml-review`. Reach for it whenever you have an ML/DL question, a plan to critique, or a system that's misbehaving.

```
/ml-review I need to fine-tune Llama-3 on a single GPU with limited VRAM
/ml-review my XGBoost model passes CV but fails in production — what do I check?
/ml-review review my plan: train a transformer on 50K rows of tabular data
/ml-review p95 TTFT is 800ms on vLLM with Llama-3-8B — what can I tune?
/ml-review which architecture should I use for ultra-long-sequence forecasting?
```

`/ml-review` reads your question and does one of three things:

1. **Routes** — points you to the right reference doc (e.g. `references/ml-architectures/attention/`).
2. **Reviews** — critiques an ML plan against the reference library, surfacing the top 2-3 risks.
3. **Analyzes / suggests** — diagnoses a system or proposes an end-to-end approach with citations.

It does this by reading reference files **on demand** — the references are not preloaded into context.

## Repository Layout

```
.claude-plugin/plugin.json           # plugin manifest (declares the single ml-review skill)
skills/
  ml-review/
    SKILL.md                         # the only auto-loaded skill — routing tables + review/analyze/suggest procedures
    references/                      # ~80 reference docs, read on demand by ml-review
      ml-architectures/              # Agents, Attention, Audio, CNN, Diffusion, LLM, Mamba, MoE, RAG, RL, Vision, World Models, …
      ml-libraries/                  # PyTorch, HuggingFace, sklearn, XGBoost, pandas, polars, numpy, Ray, vLLM, SGLang, …
      ml-training/                   # training-workflow, evaluation, llm-evaluation, prompt-engineering, inference-optimization, model-merging, online-experimentation, online-learning, data-parallel, Unsloth, GRPO, …
      data-prep/                     # eda, feature-engineering, time-series-features, dataset-curation, data-validation
      gpu-lang/                      # triton, tilelang
docs/
  acquire-ml-skill.md                # maintainer guide: add or update a single reference
  refine-ml-skill.md                 # maintainer guide: deep-research + propagate updates across multiple references
README.md
```

> The `references/` folder follows Claude's skill convention — files there are not auto-loaded into context but the skill reads them with the Read tool when needed. This keeps the per-session token footprint small.

### What lives in references/

| Folder | Topics |
|--------|--------|
| `ml-architectures/` | Agents, AI App Architecture, ANN, Attention, Audio, Autoencoder, Boltzmann, CNN, Diffusion, Embeddings, GAN, GNN, LLM, Mamba, MoE, **Neural Combinatorial Optimization**, Quantization, **RAG**, Regression/Classification, Reinforcement Learning, RNN, **Sampling Strategies**, SOM, Transformer, Vision, World Models |
| `ml-libraries/` | PyTorch, HuggingFace, scikit-learn, XGBoost, pandas, polars, numpy, Ray, NeMo, DSPy, LiteLLM, vLLM, SGLang, Triton Inference Server, keras, seaborn, plotly |
| `ml-training/` | feature-selection, training-workflow, **evaluation** (classical) + **llm-evaluation** (FM-specific), **prompt-engineering**, **inference-optimization**, **model-merging**, **gradient-free-optimization**, experiment-tracking, hf-jobs-workflow, **online-experimentation** (A/B, CUPED, bandits), **online-learning** (drift, incremental updates), data-parallel (DDP/FSDP), Unsloth SFT, Unsloth advanced (GRPO/DPO), Ray distributed SFT, distributed GRPO |
| `data-prep/` | EDA, feature engineering, **time-series-features** (lags, windows, point-process, purged CV), **dataset-curation** (FM data: SFT, preference, CoT, synthesis, dedup), data validation |
| `gpu-lang/` | Triton, TileLang |

## How the Skill Picks References

`ml-review` uses three lookup paths, in this order of preference:

1. **Workflow-stage table** — when you know what step you're on (e.g. "I'm picking metrics" → `references/ml-training/evaluation/`).
2. **Problem-type tables** — when you know what kind of problem you have (e.g. "tabular forecasting" → `references/data-prep/time-series-features/`).
3. **`grep`** — when you know a keyword (paper name, API, library) but not which reference owns it.

See [`skills/ml-review/SKILL.md`](skills/ml-review/SKILL.md) for the full tables and the review/analyze/suggest procedures.

## Reference Format

Every reference doc under `references/` follows a consistent format:
- **Why This Exists** — the problem it solves and when to reach for it
- **Code examples** — PyTorch and/or sklearn, realistic and runnable
- **Decision tables** — when to use this vs alternatives
- **References** — verified links to docs, papers, and repos
- **See Also** — cross-links so the skill can navigate by topic

## Workflow Examples

**Building a forecasting pipeline:**
```
/ml-review tabular forecasting with covariates
  → references/data-prep/time-series-features/  (lags, windows, calendar, purged CV)
  → references/ml-training/training-workflow/   (purged + embargoed CV, nested CV)
  → references/ml-training/online-learning/     (drift detection in production)
```

**Fine-tuning an LLM:**
```
/ml-review I want to fine-tune Llama-3 for customer support
  → references/data-prep/dataset-curation/      (SFT / preference data)
  → references/ml-training/unsloth-sft/         (single-GPU LoRA)
  → references/ml-training/unsloth-advanced/    (DPO/GRPO)
  → references/ml-training/llm-evaluation/      (judges, faithfulness)
  → references/ml-architectures/quantization/   (compress for inference)
  → references/ml-libraries/vllm/               (serve)
```

**Building an LLM application (RAG / agent):**
```
/ml-review architect a production RAG system
  → references/ml-architectures/rag/            (chunking, BM25/dense/hybrid, rerankers)
  → references/ml-architectures/embeddings/     (model choice, vector search)
  → references/ml-architectures/ai-app-architecture/  (gateway, guardrails, observability)
  → references/ml-training/llm-evaluation/      (faithfulness, regression eval)
```

## Contributing

Maintainer guidelines live in `docs/` (kept out of `references/` so they don't burn context tokens for end users). Read the relevant one before editing:

| Task | Read |
|------|------|
| Add one new reference or lightly edit one file | [`docs/acquire-ml-skill.md`](docs/acquire-ml-skill.md) |
| Refresh a topic that spans multiple references (e.g. a new attention variant touches `attention/`, `llm/`, `vllm/`) | [`docs/refine-ml-skill.md`](docs/refine-ml-skill.md) |
| Audit coverage of an area after a major release | [`docs/refine-ml-skill.md`](docs/refine-ml-skill.md) |
| Restructure the SKILL.md itself | edit `skills/ml-review/SKILL.md` directly |

These docs encode the quality standards, folder conventions, intake questions, and formatting rules. Hand them to Claude as the spec for the change you want — e.g. "follow `docs/acquire-ml-skill.md` to add a reference on rotary positional embeddings (RoPE)".
