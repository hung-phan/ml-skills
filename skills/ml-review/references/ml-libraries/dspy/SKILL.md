---
name: dspy
description: DSPy programming framework for LLM pipelines — modules, optimizers, async training, evaluation, and production serving patterns. Use when building optimizable LLM programs with DSPy.
---

# DSPy Skills

Production patterns for DSPy training, optimization, and serving with async services (Milvus, Redis, litellm).

- **Docs**: https://dspy.ai
- **GitHub**: https://github.com/stanfordnlp/dspy

## Why This Exists

**Problem**: LLM pipelines built with raw prompts break when you change the model or data — you're manually tuning strings instead of optimizing a program. Every model switch requires rewriting prompts by hand, and there's no principled way to improve quality over time.

**Key insight**: DSPy replaces prompt engineering with declarative modules that are automatically optimized against a metric, so the framework finds the best instructions and demonstrations for you.

**Reach for this when**: You need repeatable LLM pipeline quality that survives model changes, you want to optimize prompts systematically instead of by intuition, or you're building multi-step reasoning programs where individual prompt tweaks compound unpredictably.

## Skills

| Skill | Description |
|-------|-------------|
| [modules-and-techniques](modules-and-techniques/) | All DSPy modules (Predict, ChainOfThought, ReAct, RLM), signatures, tools, async, streaming, multimodal |
| [optimization](optimization/) | MIPROv2, GEPA, BootstrapFewShot — optimizer selection, config, before/after eval |
| [evaluation](evaluation/) | dspy.Example construction, metric design (F1, LLM-as-judge, GEPA feedback), Evaluate harness |
| [production](production/) | Serving settings (cache, history, workers), async infrastructure for training, TrainingTemplate ABC |

## References

- [DSPy Documentation](https://dspy.ai/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [DSPy Paper (arXiv:2310.03714)](https://arxiv.org/abs/2310.03714)
