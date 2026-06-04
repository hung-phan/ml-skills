---
name: optimization
description: DSPy optimizer configuration — MIPROv2, GEPA, BootstrapFewShot. Auto presets, before/after evaluation, optimizer selection guide. Use when choosing an optimizer, configuring trials, or doing before/after comparison.
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
optimized = optimizer.compile(module, trainset=trainset, num_trials=len(trainset))
```

| Preset | Trials | Bootstrapped | Use Case |
|--------|--------|-------------|----------|
| `light` | ~10 | 2 | Quick iteration |
| `medium` | ~25 | 4 | Standard |
| `heavy` | ~50+ | 8 | Maximum quality |

## GEPA Configuration

```python
optimizer = dspy.GEPA(
    metric=gepa_metric,  # must return Prediction(score, feedback)
    num_threads=4,
)
optimized = optimizer.compile(module, trainset=trainset)
```

GEPA reads `feedback` verbatim in its reflection prompt — this is the primary optimization lever. See **metrics-and-feedback** skill for GEPA metric design.

## Before/After Testing

```python
evaluator = dspy.Evaluate(devset=valset, metric=metric, num_threads=4)
baseline = evaluator(module)
optimized = optimizer.compile(module, trainset=trainset)
improved = evaluator(optimized)
print(f"{baseline:.2f} → {improved:.2f} ({(improved-baseline)/baseline*100:.1f}%)")
```

## Save/Load

```python
optimized.save("my_module.json")
loaded = MyModule()
loaded.load("my_module.json")
```

## Gotchas

- `num_trials ≥ len(trainset)` — fewer means some examples never seen
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