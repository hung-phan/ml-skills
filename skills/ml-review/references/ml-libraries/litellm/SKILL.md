---
name: litellm
description: Unified LLM interface across 100+ providers — routing, fallbacks, caching, rate limiting, and production proxy deployment. Use when calling multiple LLM providers with one API.
---

# litellm Skills

Unified API for OpenAI, Anthropic, Bedrock, Vertex, Azure, and 100+ more.

- **Docs**: https://docs.litellm.ai
- **GitHub**: https://github.com/BerriAI/litellm

## Why This Exists

**Problem**: Each LLM provider (OpenAI, Anthropic, Bedrock, Gemini, Azure) ships its own SDK with a different API shape — switching providers or adding fallbacks requires rewriting call sites, error handling, and retry logic throughout the codebase.

**Key insight**: LiteLLM exposes one OpenAI-compatible `completion()` / `acompletion()` interface that routes to 100+ providers via a `provider/model_name` string, so provider changes become one-line swaps.

**Reach for this when**: You call more than one LLM provider, need automatic fallbacks/retries across providers, want cost tracking or observability without per-provider instrumentation, or are building a service that must stay provider-agnostic.

## Skills

| Skill | Description |
|-------|-------------|
| [usage](usage/) | Core API: completion, streaming, function calling, embeddings, Router, fallbacks, cost tracking |
| [production](production/) | Prod config: shared aiohttp session, OTel tracing, Redis/disk cache, FastAPI lifespan |

---

## References

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [LiteLLM Proxy Quick Start](https://docs.litellm.ai/docs/proxy/quick_start)
