---
name: dspy-evaluation
description: DSPy evaluation — constructing dspy.Example datasets, designing metrics (F1, semantic, LLM-as-judge, GEPA feedback), train/val splits, and running the Evaluate harness. Use when writing metrics for DSPy optimization, choosing between metric types, or structuring test cases.
---

# DSPy Evaluation

## Why This Exists

**Problem**: DSPy optimization is only as good as its signal. Two things must be right: (1) your test cases must cover the real input distribution, and (2) your metric must return a signal the optimizer can act on. A binary metric gives GEPA nothing to reflect on; a poorly constructed Example leaks gold labels into inputs; a metric that plateaus at 0.85–0.90 means your criteria aren't fine-grained enough.

**Key insight**: `dspy.Example` separates what the module receives (inputs) from what the metric compares against (gold labels) via `with_inputs()`. The metric's return type (`bool` vs `float` vs `Prediction(score, feedback)`) determines which optimizer can run and how well it converges.

**Reach for this when**: Constructing trainsets/devsets, writing or choosing a metric, designing GEPA feedback, debugging why an optimizer plateaus, or running before/after comparisons.

---

## dspy.Example Construction

```python
# with_inputs() separates input fields from gold label fields
example = dspy.Example(
    query="Play something like Radiohead",
    context="User wants music similar to Radiohead's style",  # gold label
).with_inputs("query")  # only "query" is passed to the module

# Multi-turn
dspy.Example(
    query="play more like this",
    history="Now playing: 'Blinding Lights' by The Weeknd",
    context="Wants similar synth-pop continuation",
).with_inputs("query", "history")
```

**Key rule**: Fields in `with_inputs()` are what the module sees. Everything else is a gold label. Forgetting `with_inputs()` means DSPy can't distinguish input from ground truth.

### Edge Cases to Always Include

```python
dspy.Example(query="play biyonsay",       context="Beyoncé (misspelled)").with_inputs("query")
dspy.Example(query="anything but country", context="Exclude genre").with_inputs("query")
dspy.Example(query="play something good", context="No preference").with_inputs("query")
```

**Minimum viable devset**: 6–8 examples for MIPROv2. 20 diverse > 50 similar.

---

## Metric Contract

```python
def metric(example, pred, trace=None) -> bool | float | dspy.Prediction
```

| Return Type | Used By | Behavior |
|-------------|---------|----------|
| `bool` | BootstrapFewShot | Pass/fail demo filtering |
| `float` (0–1) | MIPROv2, BetterTogether | Bayesian optimization signal |
| `Prediction(score, feedback)` | GEPA | Reflection LM reads feedback verbatim |

The `trace` argument is non-None during optimization — use it to threshold float scores:
```python
def metric(example, pred, trace=None):
    score = compute_score(example, pred)
    return score >= 0.7 if trace else score
```

---

## Built-in Metrics

| Metric | Returns | Use |
|--------|---------|-----|
| `dspy.answer_exact_match` | bool | Factoid QA |
| `dspy.answer_passage_match` | bool | Passage contains gold answer |
| `dspy.SemanticF1` | Prediction | Token overlap + semantic similarity |
| `dspy.CompleteAndGrounded` | Prediction | RAG completeness + groundedness |

---

## Custom Metrics

### Token F1

```python
def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    common = set(pred_tokens) & set(ref_tokens)
    if not common: return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(ref_tokens)
    return 2 * p * r / (p + r)
```

### Set F1 (entity extraction)

```python
def set_f1(predicted: set, gold: set) -> float:
    if not predicted and not gold: return 1.0
    if not predicted or not gold: return 0.0
    p = len(predicted & gold) / len(predicted)
    r = len(predicted & gold) / len(gold)
    return 2 * p * r / (p + r) if (p + r) else 0.0
```

### LLM-as-Judge

```python
class JudgeSignature(dspy.Signature):
    """Score how well the prediction matches the gold context."""
    gold_context: str = dspy.InputField()
    predicted_context: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    score: int = dspy.OutputField(desc="1-10 integer")

def metric(example, pred, trace=None):
    result = dspy.ChainOfThought(JudgeSignature)(
        gold_context=example.context,
        predicted_context=pred.context,
    )
    score = float(result.score) / 10.0
    return score >= 0.7 if trace else score
```

Use a stronger model as judge than the target model. Cache judge calls — they're expensive.

---

## GEPA Metrics (score + feedback)

GEPA reads `feedback` verbatim in its reflection prompt — vague feedback = no improvement.

```python
def gepa_metric(example, pred, trace=None):
    scores, issues = [], []

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

### Good vs Bad Feedback

| ✅ Good | ❌ Bad |
|---------|--------|
| "Missing fact: user's timezone not included" | "Needs improvement" |
| "Format wrong: expected bullet list, got paragraph" | "Low score" |
| Fine-grained float (0.73) + per-criterion breakdown | Binary True/False |

### Composite GEPA Pattern

```python
def composite_metric(example, pred, trace=None):
    scores = {
        "accuracy": check_accuracy(example, pred),
        "format":   float(check_format(pred)),
        "concise":  check_conciseness(pred),
    }
    weights = {"accuracy": 0.6, "format": 0.2, "concise": 0.2}
    score = sum(w * scores[k] for k, w in weights.items())
    issues = [f"{k}: {v:.2f}" for k, v in scores.items() if v < 0.8]
    feedback = f"Weak: {', '.join(issues)}" if issues else "All criteria met."
    return dspy.Prediction(score=score, feedback=feedback)
```

---

## Optimizer ↔ Metric Compatibility

| Property | GEPA | MIPROv2 | BootstrapFewShot |
|----------|------|---------|-----------------|
| Return type | `Prediction(score, feedback)` | `float` | `bool` |
| Feedback | **Critical** — drives mutations | Not used | Not used |
| Granularity | Fine-grained required | Coarse OK | Pass/fail OK |

---

## Evaluate Harness

```python
evaluator = dspy.Evaluate(
    devset=valset,
    metric=metric,
    num_threads=8,
    failure_score=0.0,
    display_progress=True,
)
baseline = evaluator(module)
optimized = optimizer.compile(module, trainset=trainset)
improved = evaluator(optimized)
print(f"{baseline:.2f} → {improved:.2f} ({(improved-baseline)/baseline*100:+.1f}%)")
```

---

## Pitfalls

- **Plateau at 0.85–0.90**: criteria aren't fine-grained enough — add more dimensions
- **Metric doesn't correlate with quality**: validate against human judgment on 20+ examples first
- **Too expensive**: use cheap LMs for scoring; cache repeated comparisons
- **Binary + GEPA**: reflection LM gets nothing to reflect on — always return float + feedback for GEPA
- **Gold context too verbose**: judges compare semantics, not wording — keep gold labels concise

---

## References

- [DSPy Documentation](https://dspy.ai/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [DSPy Paper (arXiv:2310.03714)](https://arxiv.org/abs/2310.03714)
