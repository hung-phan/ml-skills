---
name: ray-serve
description: Scalable model serving with autoscaling, batching, model composition, and vLLM integration. Covers @serve.deployment, DeploymentHandle DAGs, @serve.batch, streaming, FastAPI ingress, and LLM serving. Use when deploying ML models to production, building multi-model pipelines, or need autoscaling inference endpoints.
---

# Ray Serve

- **Docs**: https://docs.ray.io/en/latest/serve/index.html
- **LLM serving**: https://docs.ray.io/en/latest/serve/llm/index.html
- **API**: https://docs.ray.io/en/latest/serve/api/index.html

## Why This Exists

**Problem**: Deploying ML models as HTTP services requires autoscaling, request batching, GPU resource management, and multi-model composition — none of which plain FastAPI/uvicorn provides, and Kubernetes-based solutions require managing separate infrastructure outside the Ray cluster.

**Key insight**: Ray Serve runs deployments as Ray actors, so autoscaling, batching, and multi-model DAG routing are first-class features expressed in Python — no YAML Kubernetes manifests or separate serving infrastructure needed.

**Reach for this when**: You need autoscaling GPU inference, want to compose multiple models into a pipeline (e.g., embedder → reranker → LLM), need OpenAI-compatible LLM serving via vLLM, or require request batching for throughput — use plain FastAPI only for single-model, low-traffic endpoints.

## Basic Deployment

```python
from ray import serve

@serve.deployment(num_replicas=2)
class MyModel:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    async def __call__(self, request):
        data = await request.json()
        return {"prediction": self.model.predict(data["input"])}

app = MyModel.bind("/models/v1")
serve.run(app, host="0.0.0.0", port=8000)
```

## Autoscaling

```python
@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_ongoing_requests": 5,  # scale trigger
        "upscale_delay_s": 10,
        "downscale_delay_s": 300,
    },
    ray_actor_options={"num_gpus": 1},
)
class AutoscaledModel:
    ...
```

## Request Batching

```python
@serve.deployment
class BatchedModel:
    def __init__(self):
        self.model = load_model()

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def predict(self, inputs: list[str]) -> list[float]:
        # Called with a batch of accumulated requests
        return self.model.batch_predict(inputs)

    async def __call__(self, request):
        data = await request.json()
        return await self.predict(data["text"])  # auto-batched
```

## Multi-Model Pipeline (DAG)

```python
from ray.serve.handle import DeploymentHandle

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 0.5})
class Embedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    async def embed(self, texts: list[str]):
        return self.model.encode(texts).tolist()

@serve.deployment
class RAGPipeline:
    def __init__(self, embedder: DeploymentHandle, llm: DeploymentHandle):
        self.embedder = embedder
        self.llm = llm

    async def __call__(self, request):
        body = await request.json()
        embeddings = await self.embedder.embed.remote(body["texts"])
        # ... retrieve docs using embeddings ...
        return await self.llm.generate.remote(context + body["query"])

app = RAGPipeline.bind(Embedder.bind(), LLMDeployment.bind())
serve.run(app)
```

## LLM Serving with vLLM

```python
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config={
        "model_id": "llama-3-8b",
        "model_source": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    deployment_config={
        "autoscaling_config": {"min_replicas": 1, "max_replicas": 4}
    },
    accelerator_type="L4",
    engine_kwargs={"tensor_parallel_size": 1, "max_model_len": 4096},
)

app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app)
# OpenAI-compatible: POST /v1/chat/completions
```

## Streaming Responses

```python
from starlette.responses import StreamingResponse

@serve.deployment
class StreamingLLM:
    async def __call__(self, request):
        async def generate():
            for token in model.stream(prompt):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
```

## FastAPI Integration

```python
from fastapi import FastAPI

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class APIServer:
    @app.post("/predict")
    async def predict(self, request: dict):
        return {"result": self.model(request["input"])}

    @app.get("/health")
    async def health(self):
        return {"status": "ok"}
```

## Model Multiplexing

```python
@serve.deployment
class MultiModel:
    @serve.multiplexed(max_num_models_per_replica=5)
    async def get_model(self, model_id: str):
        return load_model(f"s3://models/{model_id}")  # LRU-cached

    async def __call__(self, request):
        body = await request.json()
        model = await self.get_model(body["model_id"])
        return model.predict(body["input"])
```

## Production Config (YAML)

```yaml
# serve_config.yaml
applications:
  - name: my-app
    route_prefix: /
    import_path: my_module:app
    deployments:
      - name: MyModel
        num_replicas: auto
        autoscaling_config:
          min_replicas: 1
          max_replicas: 8
          target_ongoing_requests: 5
        ray_actor_options:
          num_gpus: 1
```

```bash
serve run serve_config.yaml
```

## When to Use

| Scenario | Ray Serve? |
|----------|-----------|
| Single model, low traffic | Simpler: FastAPI + uvicorn |
| Multi-model pipeline | ✅ DAG composition |
| Autoscaling GPU inference | ✅ Native autoscaling |
| LLM serving (OpenAI-compatible) | ✅ `build_openai_app` |
| Request batching for throughput | ✅ `@serve.batch` |
| A/B testing / canary | ✅ Traffic splitting |

## References

- Official docs: https://docs.ray.io/en/latest/serve/index.html
- Production guide: https://docs.ray.io/en/latest/serve/production-guide/index.html
- GitHub: https://github.com/ray-project/ray
