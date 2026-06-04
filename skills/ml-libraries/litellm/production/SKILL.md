---
name: litellm-production
description: Production litellm configuration — shared aiohttp session for connection pooling, OTel tracing, Redis/disk cache tiering, param tolerance. Use when initializing litellm for high-throughput services or configuring LLM response caching.
---

# litellm Production Config

## Why This Exists

**Problem**: LiteLLM's defaults (new HTTP session per request, no caching, no tracing) cause TCP churn, file-descriptor exhaustion, high P99 latency, and silent cost overruns when running at production scale.

**Key insight**: Three settings — a shared `aiohttp` session, a response cache, and OTel callbacks — eliminate the most common production failure modes with minimal configuration.

**Reach for this when**: Deploying a service that makes high-throughput LLM calls, integrating LiteLLM into a FastAPI app, or needing reproducible behavior in production vs. local dev.

## Deployment Option Decision Table

| Deployment Mode | Best For | Connection Pooling | Caching | Multi-Instance Support |
|---|---|---|---|---|
| **Direct SDK** (`litellm.completion`) | Single-process apps, scripts, notebooks | Manual (shared session) | In-process only | No shared state |
| **Router** (`litellm.Router`) | Load balancing across model groups, fallbacks | Per-router session | In-process only | No shared state |
| **LiteLLM Proxy** (`litellm --config`) | Multi-team shared gateway, centralized auth/logging | Managed by proxy | Redis shared across instances | Yes — single control plane |

**Rule of thumb**: Single service calling LLMs → direct SDK with shared session. Multiple services or teams → LiteLLM Proxy with Redis cache.

## Shared aiohttp Session (Connection Pooling)

litellm creates a new `aiohttp.ClientSession` per request by default → TCP churn, FD exhaustion, high P99. Override with one shared session:

```python
import aiohttp
import litellm
from litellm.constants import AIOHTTP_CONNECTOR_LIMIT, AIOHTTP_CONNECTOR_LIMIT_PER_HOST, AIOHTTP_KEEPALIVE_TIMEOUT
from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

client_session = aiohttp.ClientSession(
    connector=aiohttp.TCPConnector(
        limit=AIOHTTP_CONNECTOR_LIMIT,
        limit_per_host=AIOHTTP_CONNECTOR_LIMIT_PER_HOST,
        keepalive_timeout=AIOHTTP_KEEPALIVE_TIMEOUT,
    )
)
litellm.base_llm_aiohttp_handler = BaseLLMAIOHTTPHandler(client_session=client_session)
```

Close on shutdown: `await client_session.close()`

## Core Settings

```python
litellm.drop_params = True       # ignore unsupported params per provider (prevents crashes)
litellm.modify_params = True     # auto-adjust params for provider compatibility
litellm.callbacks = ["otel"]     # OpenTelemetry spans: model, tokens, latency, cost
```

## Response Cache (Redis / Disk)

```python
from litellm.types.caching import LiteLLMCacheType

# Production: Redis cluster (AWS Valkey/ElastiCache)
litellm.cache = litellm.Cache(
    type=LiteLLMCacheType.REDIS,
    redis_startup_nodes=[{"host": cache_host, "port": cache_port}],
    host=cache_host,
    port=cache_port,
    ssl=True,
    ttl=86400,  # 24h
)

# Local dev: disk cache
litellm.cache = litellm.Cache(
    type=LiteLLMCacheType.DISK,
    disk_cache_dir=str(Path.cwd() / ".llm_cache"),
)
```

## FastAPI Lifespan Integration

```python
from collections.abc import AsyncGenerator

is_initialized = False

async def init_llm() -> AsyncGenerator[None, None]:
    global is_initialized
    if is_initialized:
        yield
        return
    is_initialized = True

    # ... setup above ...
    yield
    await client_session.close()
```

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    async for _ in init_llm():
        yield

app = FastAPI(lifespan=lifespan)
```

## Gotchas

- `is_initialized` guard prevents double-init on hot reload
- `client_session.close()` after yield ensures no leaked connections
- `drop_params=True` is essential when routing to multiple providers (Bedrock, OpenAI, Anthropic)
- Redis `ssl=True` required for AWS managed Redis/Valkey
- `callbacks=["otel"]` requires an OTel SDK configured in your process

---

## References

- [litellm Documentation](https://docs.litellm.ai)
- [litellm GitHub](https://github.com/BerriAI/litellm)