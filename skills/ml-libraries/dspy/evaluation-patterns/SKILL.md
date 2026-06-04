---
name: dspy-evaluation-patterns
description: Test case and evaluation harness design for DSPy optimization. Use when constructing dspy.Example datasets, designing LLM-as-judge metrics, or structuring train/val splits.
---

# Evaluation Patterns

## Why This Exists

**Problem**: Knowing whether your DSPy program improved after optimization requires evaluating on a held-out devset with the right metric — without this you're flying blind and may be overfit to the trainset or regressed on edge cases.

**Key insight**: `dspy.Example` separates what the module receives (inputs) from what the metric compares against (gold labels) using `with_inputs()`, making it the atomic unit for both training and evaluation.

**Reach for this when**: You need to construct a trainset or devset for DSPy, design an LLM-as-judge metric, or run a before/after comparison to validate an optimizer's effect.

## Evaluation Approach Comparison

| Approach | Metric type | Cost | When to use |
|----------|------------|------|-------------|
| Exact match / token F1 | `bool` or `float` | Cheap | Factoid QA, entity extraction |
| Rule-based composite | `float` | Cheap | Format + content checks |
| LLM-as-judge | `float` | Medium | Open-ended generation, semantic quality |
| Multi-dimensional LLM judge | `Prediction(score, feedback)` | Medium | GEPA optimization |

## dspy.Example Construction

```python
example = dspy.Example(
    query="Play something like Radiohead",
    context="User wants music similar to Radiohead's style",
).with_inputs("query")  # fields in with_inputs() = module input; rest = gold labels
```

**Key rule**: `with_inputs()` separates what the module sees (input) from what metrics compare against (gold).

## Test Case Taxonomy

### Single-turn
```python
dspy.Example(query="Play Taylor Swift", context="Explicit artist request").with_inputs("query")
```

### Multi-turn
```python
dspy.Example(
    query="play more like this",
    history="Now playing: 'Blinding Lights' by The Weeknd",
    context="Wants similar synth-pop continuation",
).with_inputs("query", "history")
```

### Edge cases
```python
# Misspelling
dspy.Example(query="play biyonsay", context="Beyoncé request (misspelled)").with_inputs("query")
# Negation
dspy.Example(query="anything but country", context="Exclude country genre").with_inputs("query")
# Vague
dspy.Example(query="play something good", context="No preference, use history").with_inputs("query")
```

## LLM-as-Judge Pattern

```python
class JudgeSignature(dspy.Signature):
    gold_context: str = dspy.InputField()
    predicted_context: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    score: int = dspy.OutputField(desc="1-10")

def metric(example, pred, trace=None):
    result = dspy.ChainOfThought(JudgeSignature)(
        gold_context=example.context, predicted_context=pred.context
    )
    score = float(result.score) / 10.0
    if trace is not None:
        return score >= 0.7
    return score
```

## Multi-Dimensional Scoring

```python
def metric(example, pred, trace=None):
    score = (
        0.4 * intent_accuracy(example, pred) +
        0.3 * completeness(example, pred) +
        0.2 * specificity(pred) +
        0.1 * conciseness(pred)
    ) / 10.0
    return score >= 0.6 if trace else score
```

## Evaluate Harness

```python
evaluator = dspy.Evaluate(devset=valset, metric=metric, num_threads=8, failure_score=0.0)
result = evaluator(program)  # returns aggregate score
```

## Gotchas

- Minimum 6-8 examples for MIPROv2 — fewer and bootstrapping fails
- 20 diverse cases > 50 similar cases
- Gold context should be concise — judges compare semantics not wording
- Use a stronger model as judge than the target model
- `with_inputs()` is required — without it DSPy doesn't know input vs gold

---

## References

- [DSPy Documentation](https://dspy.ai)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)