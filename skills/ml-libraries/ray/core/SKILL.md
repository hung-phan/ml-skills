---
name: ray-core
description: Distributed computing with Ray tasks and actors. Covers @ray.remote functions, stateful actors, ActorPool, placement groups, fault tolerance, and resource management. Use when parallelizing Python across multiple CPUs/GPUs, building distributed services, or orchestrating multi-node workloads.
---

# Ray Core

- **Docs**: https://docs.ray.io/en/latest/ray-core/walkthrough.html
- **API**: https://docs.ray.io/en/latest/ray-core/api/core.html
- **GitHub**: https://github.com/ray-project/ray

## Why This Exists

**Problem**: Python's GIL blocks true CPU parallelism, and `multiprocessing` requires manual serialization, process management, and provides no way to share large objects between workers without copying.

**Key insight**: Ray turns any Python function or class into a distributed primitive with one decorator — `@ray.remote` — and its shared object store enables zero-copy reads of large arrays across all workers on a node.

**Reach for this when**: You need to parallelize CPU/GPU-bound Python work across cores or machines, maintain stateful workers (model servers, parameter servers), or share large data (embeddings, model weights) between many concurrent tasks.

## Concepts

| Primitive | What | Lifecycle |
|-----------|------|-----------|
| **Task** | `@ray.remote` function, stateless, returns `ObjectRef` | Fire and forget |
| **Actor** | `@ray.remote` class, stateful, pinned to worker | Lives until killed or driver exits |
| **ObjectRef** | Future/handle to value in object store | Resolved by `ray.get()` |

## Tasks (Stateless Parallelism)

```python
import ray
ray.init()

@ray.remote(num_cpus=1, num_gpus=0)
def process(item: int) -> int:
    return item ** 2

# Launch 10 in parallel
refs = [process.remote(i) for i in range(10)]
results = ray.get(refs)  # blocks until all done

# Incremental with ray.wait
remaining = refs
while remaining:
    ready, remaining = ray.wait(remaining, num_returns=1)
    print(ray.get(ready[0]))
```

## Actors (Stateful Services)

```python
@ray.remote(num_gpus=1)
class ModelServer:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.count = 0

    def predict(self, x):
        self.count += 1
        return self.model(x)

    def get_count(self) -> int:
        return self.count

server = ModelServer.remote("/models/v1")
result = ray.get(server.predict.remote(input_data))
ray.kill(server)  # explicit cleanup
```

### Fault Tolerance
```python
@ray.remote(max_restarts=3, max_task_retries=2)
class ResilientActor:
    ...
```

## Actor Pool

```python
from ray.util.actor_pool import ActorPool

@ray.remote(num_gpus=0.5)  # 2 actors per GPU
class Worker:
    def process(self, item):
        return item * 2

pool = ActorPool([Worker.remote() for _ in range(4)])
results = list(pool.map(lambda a, v: a.process.remote(v), range(100)))
```

## Async Actors

```python
@ray.remote
class AsyncWorker:
    async def fetch(self, url: str) -> str:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.get(url) as r:
                return await r.text()
```

## Placement Groups (Co-location)

```python
from ray.util.placement_group import placement_group

# Reserve 2 GPUs on same node
pg = placement_group([{"GPU": 1}, {"GPU": 1}], strategy="PACK")
ray.get(pg.ready())

actor1 = MyActor.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(pg, bundle_index=0)
).remote()
```

Strategies: `PACK` (same node), `SPREAD` (different nodes), `STRICT_PACK`, `STRICT_SPREAD`

## Resource Requests

```python
@ray.remote(num_cpus=2, num_gpus=1, memory=4 * 1024**3)  # 4GB
def heavy_task(): ...

# Fractional GPUs (multiple actors share one GPU)
@ray.remote(num_gpus=0.25)  # 4 actors per GPU
class LightModel: ...

# Custom resources
@ray.remote(resources={"TPU": 1})
def tpu_task(): ...
```

## Anti-Patterns

| Don't | Do |
|-------|-----|
| `ray.get()` inside a loop | Batch: `ray.get([ref1, ref2, ...])` |
| Pass large objects via args | `ray.put(obj)` then pass ref |
| Create millions of tiny tasks | Batch work into larger chunks |
| Forget to call `ray.init()` | Call once at program start |

## Object Store

```python
# Put large object once, share ref
big_data = ray.put(large_array)  # returns ObjectRef

@ray.remote
def use_data(data_ref):
    data = ray.get(data_ref)  # zero-copy if on same node
    return process(data)

refs = [use_data.remote(big_data) for _ in range(10)]
```

## References

- Official docs: https://docs.ray.io/en/latest/ray-core/key-concepts.html
- GitHub: https://github.com/ray-project/ray
- Paper: https://arxiv.org/abs/1712.05889 (Ray: A Distributed Framework for Emerging AI Applications)
