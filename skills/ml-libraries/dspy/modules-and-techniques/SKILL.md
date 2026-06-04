---
name: modules-and-techniques
description: Comprehensive reference for all DSPy modules, optimizers, signatures, tools, async, streaming, and caching. Use when choosing a DSPy module for a task, configuring optimizers, integrating tools, or using advanced features like RLM, streaming, or multimodal inputs.
---

# DSPy Modules and Techniques

## Why This Exists

**Problem**: Writing LLM programs as f-strings makes them brittle and unoptimizable — changing the model or task requires rewriting prompts by hand, there's no type safety on inputs/outputs, and there's no way for an optimizer to improve the program systematically.

**Key insight**: DSPy modules (Predict, ChainOfThought, ReAct, etc.) are composable building blocks with typed signatures that the optimizer can tune — they replace hand-crafted prompts with a program structure that survives model changes.

**Reach for this when**: You need to choose the right DSPy module for a reasoning task, configure an optimizer, integrate external tools with ReAct, or use advanced features like streaming, multimodal inputs, or async execution.

## Core Modules

| Module | Purpose | When to Use |
|--------|---------|-------------|
| `dspy.Predict` | Direct LM call with typed signature | Simple tasks, single-step |
| `dspy.ChainOfThought` | Step-by-step reasoning before output | Anything requiring reasoning |
| `dspy.ReAct` | Thought → tool → observation loop | Multi-step tasks with tools |
| `dspy.ProgramOfThought` | Generates + runs code to produce answers | Math, data processing |
| `dspy.RLM` | Recursive exploration via sandboxed REPL | Large contexts, decomposition |
| `dspy.CodeAct` | Code-execution agent (lighter than RLM) | Simple code tasks |
| `dspy.BestOfN` | Sample N, pick best by metric | When variance is high |
| `dspy.Refine` | Iterative self-refinement | Quality-sensitive generation |
| `dspy.Parallel` | Run modules concurrently | Independent subtasks |

## Optimizers

| Optimizer | Strategy | Metric Needs |
|-----------|----------|--------------|
| `dspy.GEPA` | Reflective prompt evolution | `Prediction(score, feedback)` |
| `dspy.MIPROv2` | Bayesian instruction + demo search | `float` 0-1 |
| `dspy.BootstrapFewShot` | Filter demos by pass/fail | `bool` |
| `dspy.BootstrapFinetune` | Fine-tune on bootstrapped traces | `bool` |
| `dspy.BetterTogether` | Weight + prompt alternation | `float` |
| `dspy.SIMBA` | Scalable iterative meta-bootstrapping | `float` |

## Signatures

```python
# Inline
predict = dspy.Predict("question -> answer")

# Class-based (full control)
class QA(dspy.Signature):
    """Answer questions concisely."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="1-2 sentences")
```

## Tools (ReAct integration)

```python
# From function (auto-introspects types + docstring)
tool = dspy.Tool(my_function)

# From MCP server
tool = dspy.Tool.from_mcp_tool(session, mcp_tool)

# Usage with ReAct
react = dspy.ReAct(signature, tools=[tool1, tool2])
result = react(question="...")
```

## Async Support

```python
# Built-in modules
result = await predict.acall(question="...")

# Custom module
class MyModule(dspy.Module):
    async def aforward(self, question):
        return await self.predict.acall(question=question)

result = await module.acall(question="...")
```

## RLM (Recursive Language Model)

For tasks where context is too large for a single prompt:

```python
rlm = dspy.RLM(
    signature,
    max_iterations=20,    # REPL turns
    max_llm_calls=50,     # sub-LM budget
    tools=[...],          # additional callables
)
```

- Inputs live as Python variables in a sandboxed REPL
- Model writes code to explore data, calls `llm_query()` for sub-tasks
- `SUBMIT(...)` ends the loop and returns typed output

## Multimodal

```python
class AnalyzeChart(dspy.Signature):
    chart: dspy.Image = dspy.InputField()
    trend: str = dspy.OutputField()

analyze = dspy.Predict(AnalyzeChart)
analyze(chart=dspy.Image("chart.png"))  # Also: URL, bytes, PIL.Image
```

Also: `dspy.Audio` for audio inputs.

## Streaming

```python
stream = dspy.streamify(
    program,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
)
async for chunk in stream(question="..."):
    if isinstance(chunk, dspy.streaming.StreamResponse):
        print(chunk.chunk)  # token
    elif isinstance(chunk, dspy.Prediction):
        final = chunk
```

## Caching

See **production** skill for DSPy serving cache/history settings.

Provider-side prompt caching (reduces cost for repeated system prompts):
```python
lm = dspy.LM("anthropic/claude-sonnet-4-5-20250929",
    cache_control_injection_points=[{"location": "message", "role": "system"}]
)
```

## Save/Load

```python
optimized.save("my_module.json")
loaded = MyModule()
loaded.load("my_module.json")
```

---

## References

- [DSPy Documentation](https://dspy.ai)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)