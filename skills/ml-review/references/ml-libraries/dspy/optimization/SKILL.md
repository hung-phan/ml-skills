---
name: dspy-optimization
description: DSPy optimizer configuration — MIPROv2, GEPA, BootstrapFewShot, auto presets, before/after evaluation, and optimizer selection. Use when choosing a DSPy optimizer, configuring trials, or doing before/after comparison.
---

# Optimization

## Why This Exists

**Problem**: Manually writing few-shot examples and prompt instructions is time-consuming, doesn't generalize across models, and produces prompts that are brittle to data distribution shifts. There's no principled way to know if your prompt is near-optimal or if a different phrasing would improve accuracy by 15%.

**Key insight**: DSPy optimizers (MIPROv2, GEPA, BootstrapFewShot) automatically search the space of instructions and demonstrations for your module, guided by your metric — turning prompt engineering into a measurable, reproducible optimization problem.

**Reach for this when**: You have a DSPy module and a metric and want to maximize performance, you need to select between optimizer strategies (reflective vs Bayesian vs bootstrap), or you want a before/after evaluation to validate that optimization actually helped.

## Optimizer Selection

| Optimizer | Strategy | Metric Needs | Best For |
|-----------|----------|--------------|----------|
| `dspy.GEPA` | Reflective prompt evolution | `Prediction(score, feedback)` | Default recommendation |
| `dspy.MIPROv2` | Bayesian instruction + demo search | `float` 0-1 | Joint prompt+demo optimization |
| `dspy.BootstrapFewShot` | Filter demos by pass/fail | `bool` | Quick few-shot selection |
| `dspy.BootstrapFinetune` | Fine-tune on bootstrapped traces | `bool` | Weight optimization |
| `dspy.BetterTogether` | Weight + prompt alternation | `float` | Combined approach |

## MIPROv2 Configuration

```python
optimizer = dspy.MIPROv2(
    metric=my_metric,
    auto="medium",            # "light" | "medium" | "heavy"
    init_temperature=1.4,     # diversity of initial proposals
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_threads=4,
)
optimized = optimizer.compile(module, trainset=trainset)
```

`auto` and manual trial control are mutually exclusive: when `auto` is set, do NOT pass `num_candidates` or `num_trials` to `compile()` (DSPy raises `ValueError`). For manual control, set `auto=None` in the constructor and pass BOTH `num_candidates` and `num_trials` to `compile()`.

| Preset | `num_candidates` (n) | Trials (single predictor) | Use Case |
|--------|--------|-------------|----------|
| `light` | 6 | ~9-10 | Quick iteration |
| `medium` | 12 | ~18 | Standard |
| `heavy` | 18 | ~27 | Maximum quality |

Trials are derived as `int(max(2*num_vars*log2(n), 1.5*n))` and scale with predictor count. `max_bootstrapped_demos` is a separate constructor arg (default 4), not governed by the auto preset.

## GEPA Configuration

```python
optimizer = dspy.GEPA(
    metric=gepa_metric,  # must return Prediction(score, feedback)
    auto="light",        # budget: exactly one of auto / max_full_evals / max_metric_calls
    reflection_lm=dspy.LM("openai/gpt-4o", temperature=1.0, max_tokens=32000),
    num_threads=4,
)
optimized = optimizer.compile(module, trainset=trainset)
```

GEPA requires exactly one of `auto` / `max_full_evals` / `max_metric_calls` to set the budget, plus a `reflection_lm` (or a custom `instruction_proposer`) — omitting either raises `AssertionError`. GEPA reads `feedback` verbatim in its reflection prompt — this is the primary optimization lever. See **metrics-and-feedback** skill for GEPA metric design.

## Before/After Testing

```python
evaluator = dspy.Evaluate(devset=valset, metric=metric, num_threads=4)
baseline = evaluator(module)
optimized = optimizer.compile(module, trainset=trainset)
improved = evaluator(optimized)
print(f"{baseline.score:.2f} → {improved.score:.2f} ({(improved.score-baseline.score)/baseline.score*100:.1f}%)")
```

`Evaluate.__call__` returns an `EvaluationResult` (score + per-example `results`), not a bare float — access `.score` for the aggregate metric.

## Save/Load

```python
optimized.save("my_module.json")
loaded = MyModule()
loaded.load("my_module.json")
```

## Gotchas

- `num_trials` (number of Bayesian optimization trials) and `auto` are mutually exclusive — set one or the other, never both
- MIPROv2 metric must return float, not int or bool
- `num_threads` = parallel LLM calls — budget accordingly
- `init_temperature > 1.0` encourages exploration early
- Always seed random splits for reproducible comparisons
- `max_bootstrapped_demos=0` for prompt-only optimization (no few-shot)
- `dspy.context(lm=...)` sets the LM for a block — use for before/after with same model

---

## References

- [DSPy Documentation](https://dspy.ai)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)