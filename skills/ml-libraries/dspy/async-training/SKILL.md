---
name: dspy-async-training
description: Async infrastructure for DSPy training — BackgroundEventLoop, contextvar propagation, syncify, and TrainingTemplate ABC. Use when training DSPy modules that call async services, or when you see "RuntimeError: no running event loop" during optimization.
---

# Async Training

Solves: DSPy optimizers are sync + threaded, your production modules are async.

## Why This Exists

**Problem**: DSPy optimization (MIPROv2, BootstrapFewShot) makes many sequential LLM calls during compilation — for large trainsets this takes hours. The optimizers are synchronous and threaded, but production modules that call async services (Milvus, Redis, litellm) require a running event loop, causing `RuntimeError: no running event loop` at compile time.

**Key insight**: A singleton background event loop on a daemon thread lets sync DSPy optimizer threads call async services safely, while context-propagating thread pools ensure request-scoped state (auth tokens, trace IDs) flows correctly into worker threads.

**Reach for this when**: Your DSPy module uses `async`/`await` internally, you see `RuntimeError: no running event loop` during `compile()`, or you need DSPy optimization to parallelize across an async service stack.

## When to Use Async Optimization

| Trainset size | Use async? | Why |
|---------------|-----------|-----|
| < 20 examples | No — sequential is fine | Overhead of async setup not worth it |
| 20–100 examples | Optional | Use if each LLM call is slow (>2s) |
| > 100 examples | Yes | Parallel execution cuts wall-clock time proportionally |
| Any size + async services | Yes | Required to avoid event loop errors |

## Architecture

```
patch_dspy_parallelizer_contextvars()     ← call once at startup
  └── ContextPropagatingThreadPoolExecutor (copies contextvars into workers)

syncify(async_module)                      ← wrap each async module
  └── forward() calls run_async(aforward())
        └── BackgroundEventLoop.run_async(coro)
              └── singleton daemon thread with asyncio loop
                    └── awaits your async services ✅
```

## Infrastructure (3 files)

### `async_utils.py`

```python
import asyncio, contextvars, threading
from typing import Any, Self

class BackgroundEventLoop:
    """Singleton background event loop for running async code from sync contexts."""
    _instance: Self | None = None
    _lock = threading.Lock()

    def __new__(cls) -> Self:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._initialized = True

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            with self._lock:
                if self._loop is None:
                    self._loop = asyncio.new_event_loop()
                    self._thread = threading.Thread(
                        target=self._run_loop, args=(self._loop,), daemon=True
                    )
                    self._thread.start()
        return self._loop

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def run_async(self, coro) -> Any:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            if running_loop is self.loop:
                raise RuntimeError("run_async() called from within the background event loop thread")
        ctx = contextvars.copy_context()
        return ctx.run(asyncio.run_coroutine_threadsafe, coro, self.loop).result()

_background_loop = BackgroundEventLoop()

def run_async(coro) -> Any:
    """Run an async coroutine from any sync context."""
    return _background_loop.run_async(coro)
```

### `dspy_patching.py`

```python
import contextvars
from concurrent.futures import ThreadPoolExecutor

class ContextPropagatingThreadPoolExecutor(ThreadPoolExecutor):
    def submit(self, fn, /, *args, **kwargs):
        ctx = contextvars.copy_context()
        return super().submit(ctx.run, fn, *args, **kwargs)

def patch_dspy_parallelizer_contextvars() -> None:
    """Patch DSPy's parallelizer to propagate contextvars. Call once before compile()."""
    try:
        import dspy.utils.parallelizer as parallelizer
    except Exception as e:
        raise RuntimeError(f"Failed to import dspy.utils.parallelizer: {e}") from e
    if getattr(parallelizer, "_contextvars_patched", False):
        return
    if hasattr(parallelizer, "ThreadPoolExecutor"):
        parallelizer.ThreadPoolExecutor = ContextPropagatingThreadPoolExecutor
    else:
        raise RuntimeError("dspy.utils.parallelizer has no ThreadPoolExecutor symbol to patch.")
    parallelizer._contextvars_patched = True
```

### `dspy_utils.py`

```python
from types import MethodType
from dspy.primitives.module import Module
from .async_utils import run_async

def syncify(program: Module, in_place: bool = True) -> Module:
    """Convert async DSPy module (aforward) to sync (forward) via run_async."""
    if in_place:
        def forward(self, *args, **kwargs):
            return run_async(self.aforward(*args, **kwargs))
        program.forward = MethodType(forward, program)
        return program

    class SyncWrapper(Module):
        def __init__(self, inner_program: Module):
            super().__init__()
            self.program = inner_program
        def forward(self, *args, **kwargs):
            return run_async(self.program.aforward(*args, **kwargs))
    return SyncWrapper(program)
```

## TrainingTemplate ABC

```python
from abc import ABC, abstractmethod
import dspy

class TrainingTemplate(ABC):
    @abstractmethod
    def load_test_cases(self) -> list[dspy.Example]: ...

    @abstractmethod
    def create_optimizer(self) -> dspy.teleprompt.Teleprompter: ...

    def compile(self, student, *, trainset, teacher=None, valset=None) -> dspy.Module:
        return self.create_optimizer().compile(student, trainset=trainset, teacher=teacher, valset=valset)
```

## Training Script Pattern

```python
# 1. Patch FIRST
patch_dspy_parallelizer_contextvars()

# 2. Subclass + syncify async modules
class MyTraining(TrainingTemplate):
    def __init__(self):
        self.lm = run_async(get_lm(model="bedrock/us.amazon.nova-pro-v1:0"))
        self.module = syncify(run_async(create_my_module()))
        self.judge = syncify(EvaluationModule())

    def load_test_cases(self): ...
    def create_optimizer(self): ...

# 3. Compile with context propagation
with create_customer_context_span(ctx):  # propagates via patched threads
    with mlflow.start_run(), dspy.context(lm=trainer.lm):
        optimized = trainer.compile(trainer.module, trainset=train_set, valset=val_set)
```

## Gotchas

- `patch_dspy_parallelizer_contextvars()` must run before ANY DSPy compile/eval
- `syncify()` is idempotent — safe to call multiple times
- `run_async()` detects deadlock (called from within background loop → RuntimeError)
- Size async pools (Milvus, Redis) ≥ `num_threads`
- DSPy version changes may move `dspy.utils.parallelizer` — patch has guard

---

## References

- [DSPy Documentation](https://dspy.ai)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)