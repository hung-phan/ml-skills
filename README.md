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

## What's Included

| Folder | Skills |
|--------|--------|
| `ml-architectures/` | Attention, ANN, CNN, RNN, Transformer, Mamba, MoE, GAN, Diffusion, GNN, LLM, Vision, RL, Autoencoder, Boltzmann, Quantization, Embeddings, Regression/Classification |
| `ml-libraries/` | PyTorch, HuggingFace, scikit-learn, XGBoost, pandas, polars, numpy, Ray, DSPy, LiteLLM, vLLM, SGLang, Triton Inference Server, keras, seaborn, plotly |
| `ml-training/` | Feature selection, training workflow, evaluation, DDP/FSDP, Unsloth SFT, Unsloth advanced (GRPO/DPO), Ray distributed SFT, distributed GRPO, experiment tracking |
| `data-prep/` | EDA, feature engineering, data validation |
| `gpu-lang/` | Triton, TileLang |

## Skill Format

Every skill follows a consistent format:
- **Why This Exists** — the problem it solves and when to reach for it
- **Code examples** — PyTorch and/or sklearn, realistic and runnable
- **Decision tables** — when to use this vs alternatives
- **References** — verified links to docs, papers, and repos

## Contributing

Use the `acquire-ml-skill` meta-skill to add or update skills — it encodes all the quality standards, folder conventions, and formatting rules for this library.

In Claude Code, activate it with:

```
/acquire-ml-skill
```

Then tell Claude what topic to add or improve. The skill will guide research, writing, and placement automatically.

See [`skills/acquire-ml-skill/SKILL.md`](skills/acquire-ml-skill/SKILL.md) for the full workflow.
