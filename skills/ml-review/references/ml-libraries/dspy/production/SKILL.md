---
name: dspy-production
description: DSPy production settings for serving — cache, history, tracing, and async workers. Use when deploying DSPy modules to production services or tuning DSPy for high-throughput inference.
---

# DSPy Production Settings

## Why This Exists

**Problem**: A DSPy program that performs well in development needs to be deployed with the right settings — default DSPy config accumulates prompt history in memory, writes to disk cache, and adds tracing overhead on every call. Without production-specific settings you ship a program that leaks memory, wastes disk IO, and slows down under concurrency.

**Key insight**: DSPy has separate concerns for compilation-time (caching and history are useful) vs serving-time (disable caches, history, and traces; maximize async workers), and the defaults favor development.

**Reach for this when**: You're deploying a DSPy module to a production service, tuning DSPy for high-throughput inference, or debugging a production DSPy deployment that is slow or consuming excessive memory.

Tune DSPy for serving (not training):

```python
import dspy

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
dspy.settings.configure(
    async_max_workers=32,           # concurrent IO tasks
    disable_history=True,           # no prompt history in prod
    max_history_size=0,
    max_trace_size=0,               # no tracing overhead in prod
)
```

## Settings Reference

| Setting | Dev | Prod | Why |
|---------|-----|------|-----|
| `enable_disk_cache` | True | False | No disk IO in containers |
| `enable_memory_cache` | True | False | Each request is unique in prod |
| `async_max_workers` | 8 | 32 | Match expected concurrency |
| `disable_history` | False | True | History accumulates memory |
| `max_trace_size` | 10000 | 0 | Tracing adds overhead per call |

## Feature-Flag for Debugging

```python
dspy.settings.configure(
    disable_history=not is_debug_enabled(),
    max_trace_size=10_000 if is_debug_enabled() else 0,
)
```

## Provider-Side Prompt Caching

Reduces cost for repeated system prompts (Anthropic):

```python
lm = dspy.LM("anthropic/claude-sonnet-4-5-20250929",
    cache_control_injection_points=[{"location": "message", "role": "system"}]
)
```

## Gotchas

- `async_max_workers` should match your service's concurrent LLM calls — too high wastes memory
- Disable caches in prod unless you have deterministic inputs (training uses cache, serving doesn't)
- Provider-side caching is different from DSPy's cache — it's server-managed prefix caching

---

## Async Infrastructure for Training

DSPy optimizers (MIPROv2, BootstrapFewShot) are synchronous and threaded, but production modules that call async services (Milvus, Redis, litellm) need a running event loop — causing `RuntimeError: no running event loop` at compile time.

**Fix**: run a singleton background event loop on a daemon thread and propagate contextvars into optimizer threads.

### async_utils.py

```python
import asyncio, contextvars, threading
from typing import Any

class BackgroundEventLoop:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"): return
        self._loop = None; self._thread = None; self._initialized = True

    @property
    def loop(self):
        if self._loop is None:
            with self._lock:
                if self._loop is None:
                    self._loop = asyncio.new_event_loop()
                    self._thread = threading.Thread(
                        target=lambda l: (asyncio.set_event_loop(l), l.run_forever()),
                        args=(self._loop,), daemon=True)
                    self._thread.start()
        return self._loop

    def run_async(self, coro) -> Any:
        ctx = contextvars.copy_context()
        return ctx.run(asyncio.run_coroutine_threadsafe, coro, self.loop).result()

def run_async(coro) -> Any:
    return BackgroundEventLoop().run_async(coro)
```

### Patch DSPy's thread pool (call once before compile)

```python
import contextvars
from concurrent.futures import ThreadPoolExecutor

class ContextPropagatingThreadPoolExecutor(ThreadPoolExecutor):
    def submit(self, fn, /, *args, **kwargs):
        ctx = contextvars.copy_context()
        return super().submit(ctx.run, fn, *args, **kwargs)

def patch_dspy_parallelizer_contextvars():
    import dspy.utils.parallelizer as p
    if getattr(p, "_contextvars_patched", False): return
    p.ThreadPoolExecutor = ContextPropagatingThreadPoolExecutor
    p._contextvars_patched = True
```

### Wrap async modules for sync optimizer

```python
from types import MethodType
import dspy

def syncify(program: dspy.Module) -> dspy.Module:
    def forward(self, *args, **kwargs):
        return run_async(self.aforward(*args, **kwargs))
    program.forward = MethodType(forward, program)
    return program
```

### Usage

```python
patch_dspy_parallelizer_contextvars()          # patch FIRST
module = syncify(run_async(create_module()))   # wrap async module
with dspy.context(lm=lm):
    optimized = optimizer.compile(module, trainset=train_set)
```

| Trainset size | Use async infra? |
|---------------|-----------------|
| < 20 examples | No |
| 20–100 | Optional (only if LLM calls > 2s) |
| > 100 | Yes |
| Any + async services | Required |

---

## References

- [DSPy Documentation](https://dspy.ai/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)