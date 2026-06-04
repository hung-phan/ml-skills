---
name: usage
description: litellm core usage — unified LLM interface across providers, completion/acompletion, streaming, function calling, embeddings, Router load balancing, fallbacks, retries, cost tracking. Use when calling LLMs via litellm, setting up multi-provider routing, or configuring fallback strategies.
---

# litellm Usage

Unified interface for 100+ LLM providers via `provider/model_name` format.

## Why This Exists

**Problem**: Using multiple LLM providers (OpenAI, Anthropic, Bedrock, Gemini) means maintaining separate SDK calls, different response shapes, and per-provider retry/fallback logic — coupling business logic to specific provider implementations.

**Key insight**: `litellm.completion(model="provider/model", ...)` normalizes every provider's response to the OpenAI format, so call sites look identical regardless of which model is behind them.

**Reach for this when**: Writing any code that calls an LLM — use this instead of provider SDKs directly so you can swap models, add fallbacks, or enable cost tracking without touching call sites.

## Basic Completion

```python
from litellm import completion, acompletion

# Sync
response = completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=500,
)
print(response.choices[0].message.content)

# Async
response = await acompletion(
    model="bedrock/us.amazon.nova-pro-v1:0",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Provider Routing Format

```python
completion(model="openai/gpt-4o", ...)
completion(model="anthropic/claude-sonnet-4-5-20250929", ...)
completion(model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0", ...)
completion(model="azure/<deployment_name>", ...)
completion(model="vertex_ai/gemini-1.5-pro", ...)
completion(model="ollama/llama2", ..., api_base="http://localhost:11434")
```

## Streaming

```python
# Sync
for chunk in completion(model="openai/gpt-4o", messages=msgs, stream=True):
    print(chunk.choices[0].delta.content or "", end="")

# Async
async for chunk in await acompletion(model="openai/gpt-4o", messages=msgs, stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

## Function Calling / Tools

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}]

response = completion(model="openai/gpt-4o", messages=msgs, tools=tools)
tool_calls = response.choices[0].message.tool_calls
```

## Embeddings

```python
from litellm import embedding, aembedding

response = embedding(model="openai/text-embedding-3-small", input=["hello world"])
vector = response.data[0]["embedding"]

# Async
response = await aembedding(model="openai/text-embedding-3-small", input=texts)
```

## Fallbacks & Retries

```python
response = completion(
    model="gpt-4o",
    messages=msgs,
    fallbacks=["anthropic/claude-sonnet-4-5-20250929", "bedrock/us.amazon.nova-pro-v1:0"],
    num_retries=3,
    timeout=30,
)
```

## Router (Load Balancing)

```python
from litellm import Router

model_list = [
    {
        "model_name": "gpt-4",  # alias for the group
        "litellm_params": {
            "model": "azure/gpt-4-east",
            "api_key": os.getenv("AZURE_EAST_KEY"),
            "api_base": "https://east.openai.azure.com/",
            "rpm": 900,
            "tpm": 100000,
        },
    },
    {
        "model_name": "gpt-4",
        "litellm_params": {
            "model": "openai/gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "rpm": 500,
        },
    },
]

router = Router(
    model_list=model_list,
    routing_strategy="simple-shuffle",  # weighted random by rpm/tpm
    num_retries=3,
    allowed_fails=3,
    cooldown_time=5,
)

response = await router.acompletion(model="gpt-4", messages=msgs)
```

### Routing Strategies

| Strategy | Best For |
|----------|----------|
| `simple-shuffle` | Default. Weighted random by rpm/tpm |
| `latency-based-routing` | Latency-sensitive workloads |
| `usage-based-routing-v2` | Distribute load evenly |
| `least-busy` | Fewest in-flight requests |
| `cost-based-routing` | Minimize cost |

### Router Fallbacks

```python
router = Router(
    model_list=model_list,
    fallbacks=[{"gpt-4": ["claude-3-opus"]}],
    context_window_fallbacks=[{"gpt-3.5-turbo": ["gpt-4-32k"]}],
)
```

## Callbacks (Observability)

```python
import litellm

# Built-in
litellm.callbacks = ["otel"]  # OpenTelemetry spans

# Custom
def my_callback(kwargs, completion_response, start_time, end_time):
    print(f"Model: {kwargs['model']}, Cost: ${completion_response._hidden_params.get('response_cost', 0):.4f}")

litellm.success_callback = [my_callback]
litellm.failure_callback = [my_callback]
```

## Cost Tracking

```python
from litellm import completion_cost

response = completion(model="gpt-4o", messages=msgs)
cost = completion_cost(completion_response=response)
print(f"${cost:.6f}")
```

## Global Settings

```python
import litellm

litellm.drop_params = True       # ignore unsupported params per provider
litellm.modify_params = True     # auto-adjust for provider compatibility
litellm.set_verbose = True       # debug logging
```

## Gotchas

- `drop_params=True` essential when routing across providers with different param sets
- Bedrock requires `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME`
- Azure requires `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`
- Router `rpm`/`tpm` limits are soft — used for weighting, not hard enforcement
- `num_retries` only retries on APIError, Timeout, ServiceUnavailable (not 400s)
- Async streaming returns the response object itself as the async iterator

---

## References

- [litellm Documentation](https://docs.litellm.ai)
- [litellm GitHub](https://github.com/BerriAI/litellm)