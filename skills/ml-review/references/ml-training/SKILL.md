---
name: ml-training
description: Complete ML training pipeline — feature selection, training loops (sklearn + PyTorch), evaluation, class imbalance, plus LLM fine-tuning (Unsloth, Ray Train, OpenRLHF, GRPO). Use when training any model from classical ML to LLMs.
---

# ML Training

## Skills

| Skill | Covers |
|-------|--------|
| [feature-selection](feature-selection/) | Filter/wrapper/embedded methods, SHAP, Boruta, VIF, PCA, stability selection |
| [training-workflow](training-workflow/) | CV strategies, hyperparameter tuning (Optuna/Ray Tune), experiment tracking, pipelines |
| [evaluation](evaluation/) | Metrics beyond accuracy, SMOTE/ADASYN, calibration, threshold optimization, statistical comparison |
| [data-parallel](data-parallel/) | DDP, FSDP, pipeline/tensor parallelism — scaling training across GPUs |
| [unsloth-sft](unsloth-sft/) | Single-GPU LLM fine-tuning with LoRA/QLoRA (2x speed, 60% less VRAM) |
| [unsloth-advanced](unsloth-advanced/) | GRPO, DPO, ORPO, vision fine-tuning, continued pretraining |
| [ray-distributed-sft](ray-distributed-sft/) | Multi-GPU/node SFT with Ray Train + DeepSpeed ZeRO + FSDP |
| [distributed-grpo](distributed-grpo/) | Distributed GRPO/PPO with OpenRLHF, veRL, TRL |
| [experiment-tracking](experiment-tracking/) | MLflow, Weights & Biases — logging runs, model registry, sweeps |
| [online-experimentation](online-experimentation/) | A/B testing pitfalls, Welch t-test, Benjamini-Hochberg, CUPED, sequential testing, multi-armed bandits, Thompson sampling — for shipping models to live traffic |
| [online-learning](online-learning/) | Incremental learning (`partial_fit`, `river`), drift detection (PSI/KS/ADWIN/DDM), train/score separation, model-as-data control-stream, event-time windows |

## Quick Decision

| Task | Skill |
|------|-------|
| Which features matter? | `feature-selection/` |
| Cross-validation, Optuna tuning | `training-workflow/` |
| Run training on HF Jobs / managed cloud GPUs | `hf-jobs-workflow/` |
| Metrics, imbalanced classes, calibration (classical ML) | `evaluation/` |
| Evaluate an LLM, RAG, or agent — judges, faithfulness, benchmarks | `llm-evaluation/` |
| Design / debug prompts; defend against injection | `prompt-engineering/` |
| Speed up or shrink LLM inference, pick a serving engine | `inference-optimization/` |
| Combine multiple finetuned checkpoints without retraining | `model-merging/` |
| Black-box / non-differentiable optimization (HPO, NAS, ES, BayesOpt) | `gradient-free-optimization/` |
| Fine-tune LLM on 1 GPU | `unsloth-sft/` |
| GRPO/DPO/vision on 1 GPU | `unsloth-advanced/` |
| Distribute SFT across GPUs | `ray-distributed-sft/` |
| Distribute GRPO at scale | `distributed-grpo/` |
| Log experiments, compare runs, model registry | `experiment-tracking/` |
| Ship a model to live traffic (A/B, bandits, CUPED) | `online-experimentation/` |
| Adapt model to streaming data, detect drift, hot-swap weights | `online-learning/` |
