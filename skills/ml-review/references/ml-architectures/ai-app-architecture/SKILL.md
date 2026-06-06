---
name: ai-app-architecture
description: Production architecture for foundation-model apps — context construction, input/output guardrails (PII, injection, format, factuality), model router and gateway (LiteLLM, Portkey), exact and semantic caching, orchestration loops, observability (Langfuse, LangSmith, Phoenix), user-feedback collection, versioning, canary rollout. Use when scaling an LLM app from POC to production, adding reliability or guardrails, picking between framework and custom orchestration, or debugging production incidents.
---

## Why This Exists

**Problem**: Most LLM POCs ship as a single API call wrapped in Streamlit. That works until the first incident: a user pastes a phone number into a third-party API; a JSON response comes back malformed and crashes a downstream service; the model provider has a 30-minute outage; bills hit five figures because the same expensive question gets re-asked 10,000 times. Production needs guardrails, fallback, caching, multi-model routing, audit trails, observability — and these aren't optional. They are what separates a demo from a service.

**Key insight (Huyen, AI Engineering Ch. 10)**: Don't build the full architecture upfront. Start with one model API call and add components only when you have evidence the missing component is hurting you. Each layer trades simplicity for capability — guardrails add latency, routers add complexity, semantic caches reduce correctness. Add them in response to a real failure, never preemptively.

**Reach for this when**:
- Architecting a new LLM app and trying to decide what's in v1 vs v2
- Adding the Nth component to an existing app and wondering where it fits
- Picking between a framework (LangChain/LangGraph) and a custom orchestrator
- Debugging a production incident where you don't know which step misbehaved
- Building a data flywheel from user feedback

This skill is the umbrella. It links to deeper skills: ../rag/ for retrieval, ../agents/ for tool-use loops, ../../ml-training/inference-optimization/ for KV/prefix caches, ../../ml-training/llm-evaluation/ for online metrics, ../../ml-training/online-experimentation/ for canary rollout.

---

# AI Application Architecture

## The Layered Progression

Huyen's architecture is incremental. Each layer answers a specific failure mode. **The right order to build is the order in which the failures bite you**, not top-down.

```
Layer 0  query -> model API -> response                  (POC)
Layer 1  + context construction (RAG, tools, files)      (model lacks knowledge)
Layer 2  + guardrails (input PII / output validation)    (security / format failures)
Layer 3  + router + gateway (multi-model, fallback)      (cost / reliability)
Layer 4  + caches (exact, semantic, prefix, tool-result) (latency / cost)
Layer 5  + orchestrator (loops, parallel, conditional)   (multi-step flows)
Layer 6  + observability + feedback                      (always — instrument from day 1)
```

Observability is *not* layer 6 chronologically — it's day 1. It just gets discussed last because it spans every layer.

---

## Layer 0: Just the Model

```python
import openai

def answer(query: str) -> str:
    return openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
    ).choices[0].message.content
```

Ship this. Watch what breaks. Then add the next layer.

---

## Layer 1: Context Construction

The model only knows what was in its training data plus what you give it. Adding RAG (retrieval), tools (function calls), or file uploads is *feature engineering for foundation models*.

| Failure signal | Add this |
|---|---|
| Model says "I don't have access to your docs" | RAG over private corpus — see ../rag/ |
| Model needs current data (weather, prices, stock) | Tool call to web search / API |
| Model needs to take action (send email, write DB) | Tool call with write side-effect — see ../agents/ |
| Users upload PDFs / images per session | File upload + per-session vector index |

**Trade-off**: Provider file-upload (OpenAI Assistants, Claude Files) is fast to ship but limited (chunking, retrieval algorithm, document count). A specialized RAG with your own vector DB scales further but you own the retrieval quality.

See ../rag/ for chunking, embedding choice, hybrid retrieval, reranking, and RAG eval.
See ../agents/ for tool design, planning loops, and write-action safety.

---

## Layer 2: Guardrails

Guardrails sit at two boundaries: before the model sees user input, and before output goes back to the user.

### Input Guardrails

Two risks: **leaking private data to a third-party API**, and **executing a hostile prompt that compromises your system**.

**PII detection + reverse-map** (the standard Huyen pattern):

```python
# Mask PII before sending to external API; unmask on the way back.
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import uuid

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def mask(query: str) -> tuple[str, dict[str, str]]:
    """Return (masked_query, reverse_map). Reverse_map[token] = original."""
    findings = analyzer.analyze(text=query, language="en",
                                 entities=["PHONE_NUMBER", "EMAIL_ADDRESS",
                                           "CREDIT_CARD", "PERSON", "US_SSN"])
    reverse_map: dict[str, str] = {}

    def make_token(entity_type: str, original: str) -> str:
        token = f"<{entity_type}_{uuid.uuid4().hex[:6]}>"
        reverse_map[token] = original
        return token

    operators = {
        f.entity_type: OperatorConfig(
            "custom",
            {"lambda": lambda x, et=f.entity_type: make_token(et, x)},
        )
        for f in findings
    }
    masked = anonymizer.anonymize(text=query, analyzer_results=findings,
                                   operators=operators).text
    return masked, reverse_map

def unmask(response: str, reverse_map: dict[str, str]) -> str:
    for token, original in reverse_map.items():
        response = response.replace(token, original)
    return response

# Usage
masked_q, rmap = mask("My phone is 555-123-4567, please confirm callback.")
# -> "My phone is <PHONE_NUMBER_a8f3c1>, please confirm callback."
# Send masked_q to OpenAI; on return, unmask only if needed for the user.
```

Other input guardrails to consider:
- **Prompt-injection detection** (NeMo Guardrails, prompt-injection classifiers, regex denylists for `ignore previous instructions`-style strings).
- **Off-topic / refuse-list** (a small classifier that drops queries about competitors, politics, illegal activity *before* spending an API call).
- **Token/length cap** to prevent users from blowing your budget with 100k-token paste-bombs.

### Output Guardrails

The model can fail in many ways. Output guardrails (a) catch failures and (b) define the policy for each.

**Format validation with retry** (Pydantic):

```python
from pydantic import BaseModel, ValidationError, Field
import openai, json

class HotelRec(BaseModel):
    name: str
    neighborhood: str
    price_per_night_usd: int = Field(ge=0)

class Recommendations(BaseModel):
    hotels: list[HotelRec] = Field(min_length=1, max_length=5)

def get_hotels(query: str, max_attempts: int = 3) -> Recommendations:
    last_err = None
    for attempt in range(max_attempts):
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "Return ONLY valid JSON matching the schema. "
                            "If you previously failed, fix the listed error."},
                {"role": "user", "content": query +
                 (f"\nPrior error: {last_err}" if last_err else "")},
            ],
            response_format={"type": "json_object"},
        )
        try:
            return Recommendations.model_validate_json(
                resp.choices[0].message.content
            )
        except ValidationError as e:
            last_err = str(e)
    raise RuntimeError(f"format guardrail failed after {max_attempts}: {last_err}")
```

For richer constraint specs (regex, factuality scorers, toxicity, brand checks) reach for [Guardrails AI](https://github.com/guardrails-ai/guardrails) or [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails).

**Other output checks**:
- **Toxicity scorer** (Perspective API, Detoxify, OpenAI moderation) — fail closed (refuse) or soft-fail (rewrite).
- **Factuality / groundedness** — score whether output is supported by retrieved context (AI judge or NLI model).
- **Brand-risk classifier** — does the response disparage your company or recommend competitors?
- **PII leak detection on outputs** — same Presidio pass on the response (catches the case where an internal-tool retrieval pulled PII into context).

### Reliability vs Latency Trade-Off

Every guardrail adds latency. Three patterns:

| Pattern | When to use | Cost |
|---|---|---|
| **Sequential retry** on failure | Cheap models, format failures | 2x latency on failure path |
| **Parallel-redundant** (fire 2x, take first valid) | Latency-critical UX | 2x token cost always |
| **Fall back to human** | High-stakes, ambiguous, or detected-anger | Variable; needs ops staff |

**Streaming complicates output guardrails**. By default, you stream tokens to the user as they arrive — but you can't toxicity-score a half-generated response. Options:

1. Buffer the whole response, score, then ship — kills TTFT.
2. Stream first to the user, score in parallel, *interrupt* the stream if score crosses threshold (visible "[redacted]" or graceful cutoff).
3. Stream only after a "safe prefix" classifier on the first N tokens approves.

Most teams pick (2) for chat UX and (1) for high-stakes domains.

**Some teams skip output guardrails entirely** to protect TTFT. Document this decision; it's a real trade-off, not negligence — but log it so you can revisit when an incident makes the case.

---

## Layer 3: Router & Gateway

These are two distinct things often bundled in one tool.

### Router

A **router** picks *which solution* handles a query. Typical destinations:

- Cheap small model (GPT-4o-mini, Haiku, Llama-8B-served)
- Expensive frontier model (GPT-4o, Claude Sonnet 4.5, Llama-405B)
- FAQ / canned response (no LLM at all)
- Human operator / Zendesk hand-off
- Refusal with stock message ("As a chatbot, I don't vote.")

Router = small **intent classifier** or **next-action predictor**. Fast, cheap, often a fine-tuned BERT or distilled small LLM. Don't burn frontier-model latency just to decide *which* frontier model to call.

```python
# Simple LLM-based router using a small model — replace with fine-tuned
# DistilBERT or similar in production.
import openai, json

ROUTES = {
    "billing":          "human_operator",
    "password_reset":   "faq_password_reset",
    "tech_troubleshoot":"specialist_chatbot",
    "general":          "default_chat",
    "out_of_scope":     "refuse",
}

def route(query: str) -> str:
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",  # cheap & fast
        messages=[
            {"role": "system",
             "content": "Classify the query. Reply with EXACTLY one of: "
                        + ", ".join(ROUTES.keys())},
            {"role": "user", "content": query},
        ],
        max_tokens=10, temperature=0,
    )
    intent = resp.choices[0].message.content.strip().lower()
    return ROUTES.get(intent, "default_chat")
```

**Routers also adjust context**. If you route a 1000-token query to a model with a 4k window and retrieval brings back 8k tokens, you need to either truncate context or reroute to a longer-context model. Build this branching into the router.

### Gateway

A **gateway** is a unified API in front of N model providers. Functions:

- Single interface — change provider in one place, not in every call site
- Key management — rotate, revoke, scope per-team
- Cost tracking + quotas — per user, per route, per day
- Rate-limit handling and provider failover
- Audit logging — every prompt/response for compliance
- Often: caching, guardrails, retries baked in

**LiteLLM gateway with fallback chain**:

```python
# pip install litellm
# LiteLLM exposes one interface (OpenAI-shaped) over 100+ providers
# and handles fallback, retries, cost logging, and key rotation.
import litellm
litellm.set_verbose = False

# Define a fallback ladder: try gpt-4o, then claude-3.5-sonnet, then a
# locally-served Llama on Ollama. Each entry uses the same litellm call shape.
FALLBACK_CHAIN = [
    {"model": "gpt-4o",                "api_key": os.environ["OPENAI_API_KEY"]},
    {"model": "claude-3-5-sonnet-20241022", "api_key": os.environ["ANTHROPIC_API_KEY"]},
    {"model": "ollama/llama3.1:70b",   "api_base": "http://localhost:11434"},
]

def chat(messages, **kwargs):
    return litellm.completion(
        model="gpt-4o",                    # primary
        messages=messages,
        fallbacks=[fb["model"] for fb in FALLBACK_CHAIN[1:]],
        num_retries=2,                     # per-model retries
        timeout=20,
        **kwargs,
    )

resp = chat([{"role": "user", "content": "Summarize today's news."}])
# litellm logs cost, latency, provider used; emits a webhook on each call.
```

Open-source / managed alternatives:
- [LiteLLM](https://github.com/BerriAI/litellm) — most flexible, self-host
- [Portkey](https://docs.portkey.ai/) — managed, strong observability
- [Helicone](https://github.com/Helicone/helicone) — proxy + observability, OSS
- MLflow AI Gateway, Kong, Cloudflare AI Gateway

### Build vs Buy: Gateway

| Situation | Build | Buy |
|---|---|---|
| 1-2 providers, no compliance burden | Yes — 100 lines of Python | No |
| Multiple teams, cost attribution per team | No | LiteLLM (self-host) or Portkey |
| Strict audit/SOC2 requirements | Maybe — need full logging anyway | Portkey, Helicone |
| Air-gapped / regulated data | Self-host LiteLLM behind VPC | No managed |
| You want one weekend to ship | No | Portkey free tier |

**Cost of leaving a gateway**: vendor lock-in to a wrapper, debugging pain when the wrapper hides headers, occasionally lagging on new model releases (the wrapper has to add support).

---

## Layer 4: Caching

Four caches show up in LLM apps. They live at different layers; do not confuse them.

| Cache | Where | Hit on | Hit rate | Risk |
|---|---|---|---|---|
| **Exact-match** | App layer (Redis) | Identical prompt | Low–med | Stale data; cross-user leakage |
| **Semantic** | App layer (vector DB) | Similar prompt (embedding) | Med | Wrong answer if threshold mistuned |
| **Prefix / KV-cache** | Inference engine (vLLM, SGLang) | Shared prompt prefix | Very high for system prompts | None (transparent) |
| **Tool-result cache** | Agent layer | Same tool call args | High for stable tools | Stale data |

### Exact-Match Cache

```python
# Redis-backed exact cache keyed on a hash of (model, system_prompt, user_query).
import hashlib, json, redis

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
TTL_SECONDS = 60 * 60 * 24  # 1 day

def cache_key(model: str, system: str, user: str) -> str:
    payload = json.dumps({"m": model, "s": system, "u": user}, sort_keys=True)
    return "llm:" + hashlib.sha256(payload.encode()).hexdigest()

def cached_complete(model, system, user, *, user_id=None, generate_fn):
    # CRITICAL: include user_id (or membership tier, or any access-scoped attr)
    # in the key if the response varies per user. Otherwise you leak across users.
    key_parts = [model, system, user]
    if user_id is not None:
        key_parts.append(f"uid={user_id}")
    key = cache_key(*key_parts[:3]) if user_id is None else \
          "llm:" + hashlib.sha256(":".join(key_parts).encode()).hexdigest()

    if (hit := r.get(key)) is not None:
        return json.loads(hit), True
    response = generate_fn(model, system, user)
    r.setex(key, TTL_SECONDS, json.dumps(response))
    return response, False
```

**The data-leak warning** (from Huyen, paraphrased): If a "What is the return policy?" answer is personalized by membership tier, caching it under the bare query string will return user X's policy to user Y. **Always include the access-scope (user_id, tenant_id, role) in the cache key when the response is personalized.**

Don't cache time-sensitive ("what's the weather") or one-off ("status of *my* order #1234") queries. Train a small classifier — or use a denylist of substrings — to gate cache writes.

### Semantic Cache

[GPTCache](https://github.com/zilliztech/GPTCache) is the reference implementation. Hand-rolled:

```python
# Semantic cache: embed the query, find the nearest cached query, return its
# answer if cosine similarity > threshold. Threshold tuning is the hard part.
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
SIM_THRESHOLD = 0.92  # tune empirically; too low -> wrong answers

class SemanticCache:
    def __init__(self):
        self.queries: list[str] = []
        self.embeddings: list[np.ndarray] = []
        self.responses: list[str] = []

    def lookup(self, query: str) -> str | None:
        if not self.embeddings:
            return None
        q_emb = embedder.encode(query, normalize_embeddings=True)
        sims = np.array(self.embeddings) @ q_emb
        best = int(np.argmax(sims))
        if sims[best] >= SIM_THRESHOLD:
            return self.responses[best]
        return None

    def store(self, query: str, response: str):
        self.queries.append(query)
        self.embeddings.append(
            embedder.encode(query, normalize_embeddings=True)
        )
        self.responses.append(response)

# In production: replace list with FAISS/Milvus/pgvector for sub-ms lookup.
```

**When semantic caching is worth it**: high-traffic FAQ-shaped apps where many phrasings of the same question are common (customer support, product Q&A). **When it backfires**: precise instructions ("multiply 7.34 by 12.91" and "multiply 7.34 by 12.92" are 0.99 similar but have different answers); time-sensitive content; personalized content.

Always evaluate hit-rate *and* answer-correctness offline before turning on. Many teams launch with `SIM_THRESHOLD = 0.99` and walk it down only if metrics hold.

### Prefix / KV-Cache

This is at the **inference engine** layer, not the app. vLLM and SGLang automatically cache the KV state of shared prompt prefixes (system prompt, few-shot examples, long shared context). For an app that re-uses the same 2k-token system prompt across users, prefix caching can drop TTFT 5-10x with zero correctness risk.

You don't implement this — you turn it on. See ../../ml-libraries/vllm/ and ../../ml-libraries/sglang/ for `--enable-prefix-caching` and the matching APIs. Provider APIs (Anthropic prompt caching, OpenAI cached prompts) expose this with explicit headers.

### Tool-Result Cache

For agentic systems, the *tool calls* are often more expensive than the LLM. Cache deterministic tool outputs (`get_stock_price("AAPL", date="2026-06-04")`) keyed by call args, with a TTL appropriate to the data freshness requirement. See ../agents/ for patterns.

---

## Layer 5: Orchestrator (Agent Patterns)

Once you have loops, parallel branches, conditional logic, retries, and write actions, you need an orchestrator. Two architectural shapes:

| Shape | When | Example |
|---|---|---|
| **DAG / pipeline** | Steps known up front, mostly linear | retrieve → rerank → generate → score → return |
| **While-loop / agent** | Steps decided by the model at runtime | ReAct, tool-use loop, planning agent |

Frameworks:

| Tool | Strength | Cost of leaving |
|---|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | State-machine over LangChain primitives, good visualization | LangChain abstractions leak everywhere; rewriting to plain Python is non-trivial |
| LangChain | Massive ecosystem, every integration | Heavy, opinionated, frequent breaking changes |
| LlamaIndex | RAG-first | Less suited for agent loops |
| smolagents (HF) | Minimal, code-first agents | Younger, smaller community |
| Haystack | Pipeline-oriented, production-tested | Verbose for simple cases |
| Custom | Full control, no surprise abstractions | You write the retry, parallel, tracing yourself |

**Huyen's caution** (and ours): *don't reach for a framework on day 1*. A 200-line Python orchestrator with explicit `await asyncio.gather(...)` and a `for attempt in range(3)` loop is debuggable, readable, and ships. Frameworks pay off when you have many pipelines or many engineers — not when one engineer has one pipeline.

If you do pick a framework: **evaluate it on extensibility** (can you add the model/tool you'll need next quarter?) and **performance** (does it inject hidden API calls? Does it serialize what should be parallel?).

When designing a pipeline with strict latency: **parallelize aggressively**. Routing and PII removal can run concurrently. Multiple retrieval sources can run concurrently. The model call is the long pole; everything else should hide behind it.

See ../agents/ for the agent loop, planning, tool selection, and write-action safety in depth.

---

## Layer 6: Observability & User Feedback

This is layer 6 in *discussion order*, layer 1 in *build order*. Instrument from the first commit.

### Trace Structure

A **trace** is the full execution record of one user request: every LLM call, tool call, retrieval, scorer, latency, token count, cost, and final outcome — linked by a request id, broken into spans.

```python
# Langfuse decorator-based tracing (Phoenix and LangSmith follow similar shapes).
# pip install langfuse
from langfuse import Langfuse, observe
from langfuse.openai import openai  # drop-in replacement; auto-instruments

langfuse = Langfuse()  # reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY

@observe()  # creates a parent span for the whole function
def answer_question(user_id: str, query: str) -> str:
    # User identity attaches to the trace for per-user dashboards.
    langfuse.update_current_trace(user_id=user_id, tags=["prod", "v2.3.1"])

    context = retrieve(query)            # @observe()-decorated separately
    response = generate(query, context)  # tracked OpenAI call below
    score   = grade(response, context)   # AI-as-judge spam check

    langfuse.update_current_trace(
        metadata={"groundedness": score, "ctx_chunks": len(context)}
    )
    return response

@observe()
def generate(query: str, context: list[str]) -> str:
    # Because we imported langfuse.openai, this call is auto-traced:
    # prompt, completion, tokens, cost, latency, model name all captured.
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Use only this context:\n{context}"},
            {"role": "user", "content": query},
        ],
    )
    return resp.choices[0].message.content
```

Comparable tools: [LangSmith](https://docs.langchain.com/langsmith), [Phoenix](https://github.com/Arize-ai/phoenix) (OSS, OpenTelemetry-native), [HoneyHive](https://docs.honeyhive.ai/). All offer traces, dataset replay, online evals, and prompt-version diffs. Self-host vs SaaS is the main axis.

### Online Metrics That Matter

Three DevOps-flavored metrics (Huyen):

- **MTTD** — mean time to detection. How long after a regression starts do you notice?
- **MTTR** — mean time to resolution. How long from notice to fix?
- **CFR** — change failure rate. % of releases that need rollback or hotfix.

LLM-specific online metrics:

| Category | Metric | Why |
|---|---|---|
| Latency | TTFT, TPOT, total | UX; see ../../ml-training/inference-optimization/ |
| Cost | tokens-in, tokens-out, $/request, cache hit-rate | budget |
| Quality | groundedness, format-validity, refusal-rate, false-refusal | regression detection |
| Engagement | regenerations, edits, copy-events, conversation turns, abandonment, completion | implicit feedback |
| Safety | toxicity score, PII-in-output rate, guardrail trigger rate | compliance |

**Always slice metrics by**: prompt/version, model, route, user cohort, time. A 3% groundedness drop overall might be a 20% drop on one route — the average hides the bug.

### User Feedback Collection

User feedback is the data flywheel. Two kinds:

**Explicit** — thumbs up/down, star ratings, "did we solve your problem?" buttons. Cheap to implement, sparse in practice (most users skip), biased (unhappy users complain more), but unambiguous.

**Implicit** — inferred from user actions:

| Signal | Indicates |
|---|---|
| Stop generation halfway | Bad response (or user got the answer they needed early) |
| Regenerate | First response inadequate (or user wants variety) |
| Rephrase ("No, I meant...") | Model misunderstood — strong negative signal |
| Edit the response | Almost-right; the edit is gold for preference data |
| Copy / paste output | Useful response (especially for code assistants) |
| Conversation length | Depends — companion app: good. Customer support: bad. |
| Conversation deletion | Bad conversation (or embarrassing — context-dependent) |
| Hand-off to human | System failed |

**Action-correcting feedback** in agentic apps: "You should also check the GitHub page" — log these; they're free training data for future preference fine-tuning.

**User edits as preference pairs**: every edit gives you `(query, original_response=loser, edited_response=winner)`. Stash these for DPO/IPO. See ../../ml-training/llm-evaluation/ for eval; ../../ml-training/ for alignment fine-tuning.

#### When to Ask for Feedback

| Moment | Pattern |
|---|---|
| First session | Optional calibration ("rate your skill level"). Keep optional — friction kills activation. |
| When the model is uncertain | Side-by-side comparison (Gemini-style: show partials, click expands). |
| When something visibly fails | Inline thumbs-down, regenerate, switch-model buttons. |
| After every interaction | Don't. Apple's HIG: asking after every success implies success is rare. |
| Sparingly, on success | Show 1% of users a "this was great" path to surface flagship features. |

#### Feedback Biases to Plan For

| Bias | Mitigation |
|---|---|
| **Leniency** (everyone clicks 5/4 stars) | Use neutral-language options; look at distribution shape, not mean |
| **Position** (first option wins) | Randomize order; A/B-corrected lift |
| **Length** (longer judged better) | Show length-controlled comparisons in eval |
| **Recency** (last-shown wins) | Randomize order over multi-comparison |
| **Random clicks** | Filter sessions with clearly inattentive patterns |
| **Sycophancy / degenerate loop** | Don't blindly RLHF on user feedback — it teaches the model to flatter (Sharma et al. 2023). Re-eval against an objective benchmark periodically. |

### Drift Detection

Three axes drift independently:

1. **System-prompt drift** — a teammate edits the prompt template; nobody notices until quality drops. **Fix**: hash and log the active system prompt with every request; alert on changes.
2. **User-behavior drift** — users learn to phrase queries differently over time. Average response length might gradually fall not because the model changed, but because users got better at asking concisely. Investigate before "fixing".
3. **Underlying-model drift** — same provider API, but the model behind it was silently updated. Pin model version where possible (`gpt-4o-2024-08-06`, not `gpt-4o`) and run a daily regression eval against a frozen test set.

---

## Operational Patterns

### Versioning Prompts Like Code

- Store prompts in Git, not hard-coded in app config UIs. Diff-able, reviewable, rollback-able.
- SemVer-style: `pricing-v2.3.0`. Bump major on behavior changes, minor on feature adds, patch on typos.
- Log the prompt version with every request — observability dashboards must filter by version.

### Eval-in-CI

Block prompt/model changes on regression eval. Minimum gate:

1. Frozen eval set of 100-1000 representative queries with reference outputs (or LLM-judged criteria).
2. Run eval on PR; fail if quality drops > X% or refuse-rate > Y%.
3. Track cost-per-eval-query — a prompt that adds 500 tokens of instructions costs forever.

See ../../ml-training/llm-evaluation/ for eval set construction, AI judges, and online eval.

### Shadow / Canary Deploy

Same machinery as classical ML rollout — but you also need it for prompt changes, not just model changes.

- **Shadow**: route 100% to old, also send to new in background, compare outputs (maybe via AI judge).
- **Canary**: route 1% → 10% → 50% → 100% over hours/days; auto-rollback if metrics regress.
- **Per-user sticky**: once a user sees v2, keep them on v2 to avoid jarring within-session changes.

See ../../ml-training/online-experimentation/ for A/B harness design.

### Cost Budgets

Three layers of budget enforcement:

1. **Per route** — "the FAQ route may not exceed $0.001/query average." Hard cap; route to the smaller model if exceeded.
2. **Per user** — quota by tier. Free users: 50 queries/day. Pro: 500/day. Block at gateway.
3. **Per query** — max input tokens, max output tokens, timeout. Prevents single-query cost blowups.

LiteLLM and Portkey handle (1) and (2); you set (3) per-call.

### Privacy Modes

Three escalating tiers:

| Tier | Pattern | Use |
|---|---|---|
| Standard | Mask PII before external API; reverse on return | Public/consumer data |
| Zero-data-retention | Provider contract guarantees no logging/training | Regulated industries (Anthropic, OpenAI offer this with enterprise contracts) |
| On-prem inference | Self-host model behind VPC; never leaves network | HIPAA, sovereign data, IP-sensitive |

Redact-then-route: if the query contains protected data, route to the on-prem model; if not, route to the cheaper external API. The router is the decision point.

---

## Latency & Reliability

### Streaming Responses

- **TTFT** (time to first token) is what feels fast to users.
- Always stream user-facing chat; never stream programmatic API responses (the consumer wants the parsed object, not a chunked string).
- Streaming complicates output guardrails (see Layer 2). Pick an interrupt strategy.

### Hedged Requests

```python
# Fire to two providers, take whichever returns first.
# Roughly halves p99 latency; doubles token cost.
import asyncio, litellm

async def hedged(messages):
    a = asyncio.create_task(litellm.acompletion(model="gpt-4o", messages=messages))
    b = asyncio.create_task(litellm.acompletion(model="claude-3-5-sonnet-20241022",
                                                 messages=messages))
    done, pending = await asyncio.wait({a, b}, return_when=asyncio.FIRST_COMPLETED)
    for p in pending:
        p.cancel()
    return done.pop().result()
```

Worth it for tail-latency-sensitive UX (voice, real-time chat). Wasteful for batch.

### Provider Failover Ladder

Order providers by cost ascending, fall through on `RateLimitError | APIError | TimeoutError`. LiteLLM `fallbacks=` does this. Monitor *which* fallback fires — frequent fallbacks to the expensive model means your primary is unhealthy or your rate limits are too low.

### Circuit Breakers

If provider X has failed 50% of calls in the last 60s, stop calling it for 30s and route directly to fallback. Prevents thundering herd at provider recovery and cuts user-perceived latency from "20s timeout per try" to "instant fallback."

---

## Decision Tables

### When to Add Each Layer

| Symptom | Add |
|---|---|
| "Model doesn't know our docs" | Layer 1 — RAG |
| "Model can't take action" | Layer 1 — Tools |
| "PII leaked to OpenAI" | Layer 2 — input PII guardrail |
| "Output JSON crashed downstream" | Layer 2 — output format validator + retry |
| "Toxic response shown to user" | Layer 2 — output toxicity scorer |
| "Bill is 3x what we projected" | Layer 3 — router (cheap vs expensive) + Layer 4 cache |
| "Provider X had a 30min outage" | Layer 3 — gateway with failover |
| "Same FAQ asked 1000x/day" | Layer 4 — exact cache, then maybe semantic |
| "Multi-step flow with branching" | Layer 5 — orchestrator (start custom, framework later) |
| "Don't know which step broke" | Layer 6 — distributed tracing (do this from day 1) |
| "Quality silently regressed last week" | Layer 6 — eval-in-CI + drift detection |

### Single-Model vs Multi-Model Routing

| You have | Use |
|---|---|
| One use case, low traffic, simple queries | Single model. Skip the router. |
| Mixed query difficulty + cost pressure | Two-tier router: cheap for easy, expensive for hard |
| Distinct domains (billing, tech support, sales) | Intent router → specialist models or specialist prompts |
| Compliance-mixed traffic (some PII, some not) | Privacy router → on-prem vs cloud |

### Build vs Buy — Per Layer

| Layer | Build | Buy |
|---|---|---|
| Guardrails | Custom Pydantic + Presidio for narrow needs | Guardrails AI / NeMo for broad coverage |
| Gateway | `litellm` library in your service for 1-2 providers | Portkey / LiteLLM Proxy / Helicone for org-wide |
| Cache | Redis + hash key, ~50 lines | GPTCache for semantic |
| Orchestrator | Plain Python until ~3+ flows or ~3+ engineers | LangGraph, Haystack, or smolagents |
| Observability | Don't. The OSS options (Phoenix, Langfuse OSS) are cheap and good. | Langfuse / LangSmith / Phoenix / HoneyHive |

---

## Anti-Patterns

- **Building all 6 layers before you have users.** Each layer has a maintenance cost. Add only when a real failure motivates it.
- **Frameworks first.** A LangChain orchestration of a single prompt is over-engineered. Plain Python beats it.
- **Caching personalized responses without scoping the key by user.** Causes data-leak bugs.
- **Semantic cache with default thresholds.** 0.85 cosine sim does not mean "same answer." Tune offline.
- **Skipping guardrails for latency without logging the decision.** Document the trade; revisit on incident.
- **One global prompt for all users.** Becomes unversioned, unowned, and undebuggable.
- **No prompt versioning.** A teammate "fixes a typo" and quality drops 5%; you can't tell why for a week.
- **Trusting user feedback indiscriminately.** Sycophancy bias is real; re-eval against held-out objective benchmarks.
- **Output guardrails on streaming responses without an interrupt path.** Either the guardrail is useless or the stream stalls.

---

## See Also

- ../rag/ — retrieval, chunking, hybrid search, reranking, RAG eval (the Layer-1 deep dive)
- ../agents/ — planning, tool design, write-action safety, agent loops (Layer 5 deep dive)
- ../../ml-training/prompt-engineering/ — prompt patterns, few-shot, chain-of-thought
- ../../ml-training/inference-optimization/ — KV cache, prefix cache, speculative decoding (Layer-4 inference layer)
- ../../ml-training/llm-evaluation/ — eval sets, AI judges, regression eval, online metrics
- ../../ml-training/online-experimentation/ — shadow / canary / interleaving / A-B
- ../../ml-libraries/litellm/ — gateway library deep dive
- ../../ml-libraries/vllm/ — high-throughput inference engine with prefix caching
- ../../ml-libraries/sglang/ — structured generation + radix-tree prefix cache
- ../../ml-libraries/triton-inference-server/ — production model serving

## References

- Chip Huyen, *AI Engineering* (O'Reilly, 2024), Ch. 10 — source of the layered progression and feedback taxonomy.
- LiteLLM (gateway, 100+ providers): https://github.com/BerriAI/litellm
- LangGraph (state-machine orchestrator): https://github.com/langchain-ai/langgraph
- Langfuse (open-source LLM observability): https://github.com/langfuse/langfuse
- Phoenix (Arize, OpenTelemetry-native LLM tracing): https://github.com/Arize-ai/phoenix
- GPTCache (semantic cache): https://github.com/zilliztech/GPTCache
- Presidio (PII detection / anonymization): https://github.com/microsoft/presidio
- Guardrails AI (output validators): https://github.com/guardrails-ai/guardrails
- NeMo Guardrails (NVIDIA): https://github.com/NVIDIA-NeMo/Guardrails
- HoneyHive (LLM observability + evals): https://docs.honeyhive.ai/
- LangSmith docs: https://docs.langchain.com/langsmith
- Helicone (proxy + observability, OSS): https://github.com/Helicone/helicone
- Portkey docs: https://docs.portkey.ai/
- Sharma et al. 2023, *Towards Understanding Sycophancy in Language Models* — the canonical sycophancy paper.
- Xu et al. 2022 (FITS dataset) and Yuan et al. 2023 — natural-language feedback taxonomy for conversational bots.
