---
name: ray
description: Distributed Python framework for ML workloads. Sub-skills cover Core (tasks/actors), Data (streaming ETL), Serve (model deployment), and Tune (hyperparameter search). Use when scaling any ML workflow beyond a single machine.
---

# Ray

- **Docs**: https://docs.ray.io
- **GitHub**: https://github.com/ray-project/ray
- **Install**: `pip install "ray[default]"`

## Why This Exists

**Problem**: Python's GIL serializes CPU-bound work, and `multiprocessing` is too low-level to coordinate tasks, actors, data pipelines, and model serving across a cluster — each requires different libraries with incompatible abstractions.

**Key insight**: Ray gives every ML workload — parallel tasks, stateful services, streaming data, hyperparameter search, distributed training, and model serving — a single unified API on top of one distributed scheduler.

**Reach for this when**: You need to scale any Python ML workload beyond one machine, or compose multiple distributed components (e.g., data preprocessing → training → serving) without gluing together separate infrastructure tools.

## Sub-Skills

| Skill | Path | Use When |
|-------|------|----------|
| **Core** | `core/SKILL.md` | Parallelizing tasks, building stateful actors, resource management |
| **Data** | `data/SKILL.md` | Preprocessing datasets too large for memory, streaming tokenization |
| **Serve** | `serve/SKILL.md` | Deploying models with autoscaling, batching, multi-model DAGs |
| **Tune** | `tune/SKILL.md` | Hyperparameter search with early stopping (ASHA, PBT, Optuna) |

Also see: `ml-training/ray-distributed-sft/` for Ray Train + TRL fine-tuning.

## Quick Decision

| Task | Ray Component |
|------|--------------|
| Run function on 100 GPUs in parallel | **Core** — `@ray.remote` tasks |
| Persistent service with state | **Core** — actors |
| Tokenize 1TB of text for training | **Data** — `map_batches` |
| Serve LLM with autoscaling | **Serve** — `build_openai_app` |
| Find best learning rate | **Tune** — `Tuner` + ASHA |
| Distributed SFT/GRPO training | **Train** — see `ml-training/` |

## References

- Official docs: https://docs.ray.io/en/latest/
- GitHub: https://github.com/ray-project/ray
- Paper: https://arxiv.org/abs/1712.05889 (Ray: A Distributed Framework for Emerging AI Applications)
