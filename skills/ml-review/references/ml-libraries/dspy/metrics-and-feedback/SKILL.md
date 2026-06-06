---
name: dspy-metrics-and-feedback
description: DSPy metric design patterns — F1, semantic similarity, LLM-as-judge, and GEPA feedback. Use when writing metrics for DSPy optimization, choosing between metric types, or designing feedback for GEPA's reflection loop.
---

# Metrics and Feedback

## Why This Exists

**Problem**: The DSPy optimizer needs a differentiable signal — a function that returns a score (0–1) for each (input, output) pair. Bad metrics produce useless optimizations: binary metrics give GEPA nothing to reflect on, opaque scalar metrics give MIPROv2 a flat landscape to search, and expensive metrics make compilation unaffordable.

**Key insight**: The metric is the optimization target — its shape (bool vs float vs Prediction with feedback) determines which optimizer you can use and how well it converges.

**Reach for this when**: You need to write or choose a DSPy metric, design feedback for GEPA's reflection loop, or debug why an optimizer is plateauing or producing nonsensical results.

## Metric Contract

```python
def metric(example, pred, trace=None) -> bool | float | dspy.Prediction
```

| Return Type | Used By | Behavior |
|-------------|---------|----------|
| `bool` | BootstrapFewShot | Pass/fail filtering |
| `float` | MIPROv2, BetterTogether | Bayesian optimization signal |
| `Prediction(score, feedback)` | GEPA | Reflection LM reads feedback |

## Built-in Metrics

| Metric | Returns | Use |
|--------|---------|-----|
| `answer_exact_match` | bool | Factoid QA |
| `answer_passage_match` | bool | Passage contains gold |
| `SemanticF1` | Prediction | Token overlap + semantic |
| `CompleteAndGrounded` | Prediction | RAG completeness + groundedness |

## Token F1

```python
def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    common = set(pred_tokens) & set(ref_tokens)
    if not common: return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(ref_tokens)
    return 2 * p * r / (p + r)
```

## Set F1 (multi-label, entity extraction)

```python
def set_f1(predicted: set, gold: set) -> float:
    if not predicted and not gold: return 1.0
    if not predicted or not gold: return 0.0
    p = len(predicted & gold) / len(predicted)
    r = len(predicted & gold) / len(gold)
    return 2 * p * r / (p + r) if (p + r) else 0.0
```

## GEPA Metrics

GEPA reads `feedback` verbatim in its reflection prompt. This is the primary lever.

```python
def gepa_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    issues = []
    scores = []

    accuracy = check_facts(pred.response, example.response)
    scores.append(accuracy)
    if accuracy < 1.0:
        issues.append(f"Accuracy {accuracy:.0%}: missing/wrong claims")

    completeness = check_coverage(pred.response, example.response)
    scores.append(completeness)
    if completeness < 0.8:
        issues.append(f"Covers only {completeness:.0%} of required points")

    score = sum(scores) / len(scores)
    feedback = "; ".join(issues) if issues else "Perfect."
    return dspy.Prediction(score=score, feedback=feedback)
```

### Good vs Bad GEPA Feedback

| ✅ Good | ❌ Bad |
|---------|--------|
| "Missing fact: user's timezone not included" | "Needs improvement" |
| "Format wrong: expected bullet list, got paragraph" | "Low score" |
| Fine-grained float (0.73) | Binary (True/False) |
| Per-criterion breakdown | Single opaque number |

## GEPA vs MIPROv2

| Property | GEPA | MIPROv2 |
|----------|------|---------|
| Return type | `Prediction(score, feedback)` | `float` sufficient |
| Feedback | **Critical** — drives mutations | Not used |
| Granularity | Fine-grained required | Coarse OK |
| `pred_name`/`pred_trace` | Used for per-predictor scoring | Not used |

## Composite Pattern

```python
def composite(example, pred, trace=None):
    scores = {
        "accuracy": check_accuracy(example, pred),
        "format": float(check_format(pred)),
        "concise": check_conciseness(pred),
    }
    weights = {"accuracy": 0.6, "format": 0.2, "concise": 0.2}
    score = sum(w * scores[k] for k, w in weights.items())
    issues = [f"{k}: {v:.2f}" for k, v in scores.items() if v < 0.8]
    feedback = f"Weak: {', '.join(issues)}" if issues else "All met."
    return dspy.Prediction(score=score, feedback=feedback)
```

## Pitfalls

- **Plateau**: All candidates score 0.85-0.90 → add more fine-grained criteria
- **Doesn't correlate**: Validate metric vs human judgment on 20+ examples first
- **Too expensive**: Use cheap LMs for scoring, cache repeated comparisons
- **Binary + GEPA**: Reflection LM gets nothing to reflect on → always return float + feedback

---

## References

- [DSPy Documentation](https://dspy.ai)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)