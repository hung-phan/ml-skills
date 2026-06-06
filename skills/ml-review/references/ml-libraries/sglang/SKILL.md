---
name: sglang
description: High-performance LLM/VLM serving with RadixAttention prefix caching, structured outputs, and a frontend DSL for complex LLM programs. Use when serving multi-turn conversations, branching/agentic workflows, or constrained generation (JSON, regex, grammar).
---

# SGLang

## 1. Why This Exists

Traditional LLM serving systems (vLLM, TGI) optimize single-request throughput but miss cross-request optimization opportunities. Complex LLM programs involve:

- **Multi-turn conversations** sharing system prompts and conversation history
- **Branching** where one prompt forks into parallel completions
- **Constrained generation** (JSON, regex, grammar) requiring FSM-guided decoding
- **Agentic loops** with tool calls reusing the same prefix repeatedly

SGLang solves this with **RadixAttention** — a radix tree that stores and reuses KV-cache across requests sharing common prefixes. Combined with a zero-overhead CPU scheduler, compressed FSM for constrained decoding, and a Python frontend DSL, SGLang delivers 2-5x speedups over alternatives for structured and multi-turn workloads.

**Key insight**: The serving engine should understand the *program structure*, not just individual requests. Shared prefixes, parallel branches, and grammar constraints are first-class optimization targets.

## 2. Key Concepts

### RadixAttention (Prefix KV-Cache Sharing)
- Maintains a **radix tree** of all active KV-cache entries across requests
- Requests sharing a common prefix (system prompt, few-shot examples, conversation history) automatically reuse cached KV states
- Eliminates redundant prefill computation — critical for multi-turn and agentic workloads
- Automatic eviction with LRU policy when memory pressure increases

### Compressed Finite State Machine (Constrained Decoding)
- Uses **XGrammar** (default) for JSON schema, regex, and EBNF constraints
- Compressed FSM representation is ~10x faster than Outlines for structured output
- Guarantees output conforms to schema — no post-hoc validation needed
- Also supports Outlines and llguidance backends

### Continuous Batching + Zero-Overhead Scheduler
- Paged attention with continuous batching (like vLLM)
- Zero-overhead CPU scheduler eliminates scheduling latency
- Prefill-decode disaggregation for optimal GPU utilization
- Chunked prefill prevents long prompts from blocking short requests

### Hardware & Scale
- NVIDIA (GB200/B300/H100/A100/5090), AMD (MI355/MI300), Intel CPU, Google TPU, Ascend NPU
- Tensor/Pipeline/Expert/Data parallelism
- Prefill-Decode disaggregation across nodes
- Multi-LoRA batching
- Speculative decoding (including adaptive)

### Model Support
- LLMs: Llama, Qwen, DeepSeek, Kimi, GLM, GPT-OSS, Gemma, Mistral, MiniMax
- VLMs: Qwen2.5-VL, LLaVA, InternVL, etc.
- Embedding: e5-mistral, gte, mcdse
- Reward models: Skywork
- Diffusion: WAN, Qwen-Image (via SGLang Diffusion)

## 3. Code Examples

### Launch Server

```bash
# Basic single-GPU
sglang serve --model-path meta-llama/Meta-Llama-3.1-8B-Instruct

# Multi-GPU with tensor parallelism
sglang serve --model-path meta-llama/Meta-Llama-3.1-70B-Instruct --tp 4

# With data parallelism
sglang serve --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --dp 2

# FP8 quantization
sglang serve --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --quantization fp8
```

### OpenAI-Compatible API

```python
import openai

client = openai.Client(base_url="http://localhost:30000/v1", api_key="None")

# Standard chat completion
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0,
    max_tokens=128,
)
print(response.choices[0].message.content)
```

### Structured Output — JSON Schema

```python
from pydantic import BaseModel, Field

class CapitalInfo(BaseModel):
    name: str = Field(..., description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Capital of France in JSON format."}],
    temperature=0,
    max_tokens=128,
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "capital", "schema": CapitalInfo.model_json_schema()},
    },
)
# Output guaranteed valid: {"name": "Paris", "population": 2147000}
capital = CapitalInfo.model_validate_json(response.choices[0].message.content)
```

### Structured Output — Regex Constraint

```python
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0,
    max_tokens=128,
    extra_body={"regex": "(Paris|London)"},
)
# Output guaranteed to be exactly "Paris" or "London"
```

### Structured Output — EBNF Grammar

```python
ebnf_grammar = """
root ::= city " is " "the capital of " country
city ::= "London" | "Paris" | "Berlin" | "Rome"
country ::= "England" | "France" | "Germany" | "Italy"
"""

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Capital of France?"}],
    temperature=0,
    max_tokens=32,
    extra_body={"ebnf": ebnf_grammar},
)
# Output: "Paris is the capital of France"
```

### Frontend DSL — Basic

```python
from sglang import function, gen, system, user, assistant
from sglang import RuntimeEndpoint
from sglang.lang.api import set_default_backend

set_default_backend(RuntimeEndpoint("http://localhost:30000"))

@function
def qa(s, question):
    s += system("You are a helpful assistant.")
    s += user(question)
    s += assistant(gen("answer", max_tokens=256))

state = qa("List 3 countries and their capitals.")
print(state["answer"])
```

### Frontend DSL — Multi-Turn (Prefix Sharing)

```python
@function
def multi_turn(s):
    s += system("You are a helpful assistant.")
    s += user("List 3 countries and their capitals.")
    s += assistant(gen("first_answer", max_tokens=256))
    # Second turn reuses KV-cache from system + first turn automatically
    s += user("Now give me 3 more.")
    s += assistant(gen("second_answer", max_tokens=256))

state = multi_turn()
```

### Frontend DSL — Parallel Fork

```python
@function
def parallel_gen(s):
    s += system("You are a helpful assistant.")
    s += user("Give me a one-sentence tip about exercise.")

    # Fork into 3 parallel completions — all share the prefix KV-cache
    forks = s.fork(3)
    for i, f in enumerate(forks):
        f += assistant(gen(f"tip_{i}", max_tokens=100))

    # Collect results
    for i in range(3):
        print(forks[i][f"tip_{i}"])

state = parallel_gen()
```

### Frontend DSL — Constrained Decoding

```python
@function
def constrained(s):
    s += user("What is the IP of Google DNS?")
    s += assistant(gen(
        "answer",
        temperature=0,
        regex=r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)",
    ))

state = constrained()
# Output: "8.8.8.8" (guaranteed valid IPv4)
```

### Frontend DSL — Choices (Select)

```python
@function
def tool_routing(s, question):
    s += user(question)
    s += assistant(
        "I need to use a "
        + gen("tool", choices=["calculator", "search engine", "database"])
        + "."
    )

state = tool_routing("What is 2+2?")
print(state["tool"])  # "calculator"
```

### Batch Inference

```python
@function
def text_qa(s, question):
    s += user(question)
    s += assistant(gen("answer", max_tokens=64))

states = text_qa.run_batch(
    [
        {"question": "Capital of UK?"},
        {"question": "Capital of France?"},
        {"question": "Capital of Japan?"},
    ],
    progress_bar=True,
)
for s in states:
    print(s["answer"])
```

### Offline Engine (No Server)

```python
import sglang as sgl

llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")

prompts = ["Capital of France?", "Capital of Japan?"]
outputs = llm.generate(prompts, {"temperature": 0, "max_new_tokens": 64})
for o in outputs:
    print(o["text"])

llm.shutdown()
```

## 4. Performance

### vs vLLM Benchmarks (where SGLang wins)

| Workload | SGLang Advantage | Why |
|----------|-----------------|-----|
| Multi-turn chat | 2-5x throughput | RadixAttention reuses conversation prefix |
| Structured output (JSON) | 3-10x faster | Compressed FSM vs Outlines' uncompressed FSM |
| Shared system prompts | 2-3x throughput | Prefix KV-cache sharing across users |
| Parallel branching (fork) | 2-4x throughput | Single prefill, multiple decode branches |
| Batch w/ common prefix | 2-5x throughput | Automatic prefix deduplication |
| Long context (128K+) | Competitive | Pipeline parallelism + HiCache |
| Single short request | ~Same | Both use paged attention + continuous batching |

### Published Results
- v0.2 (Jul 2024): Up to 2.7x faster Llama3 serving vs TensorRT-LLM and vLLM
- v0.3 (Sep 2024): 7x faster DeepSeek MLA, 1.5x faster torch.compile
- v0.4 (Dec 2024): Zero-overhead scheduler, cache-aware load balancer
- GB300 (Feb 2026): 25x inference performance on NVIDIA GB300 NVL72

### When vLLM Might Be Better
- Single-request latency with no prefix sharing (equivalent performance)
- You need PagedAttention v2 specific features
- Ecosystem integration with specific vLLM plugins

## 5. Deployment

### Docker

```bash
# NVIDIA GPU
docker run --gpus all -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest \
  sglang serve --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 30000

# AMD GPU (ROCm)
docker run --device=/dev/kfd --device=/dev/dri \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest-rocm \
  sglang serve --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 30000
```

### Serve Your Own Fine-Tuned Model

```bash
# Option 1: Merged model from local path
sglang serve --model-path /path/to/my-merged-model \
  --host 0.0.0.0 --port 30000 \
  --served-model-name my-custom-model

# Option 2: With LoRA adapter
sglang serve --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --lora-paths my-adapter=/path/to/lora-adapter \
  --host 0.0.0.0 --port 30000

# Option 3: Docker with local model mounted
docker run --gpus all -p 30000:30000 \
  -v /home/user/models/my-model:/model \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest \
  sglang serve --model-path /model \
  --host 0.0.0.0 --port 30000

# Option 4: Custom tokenizer (e.g. for merged GGUF)
sglang serve --model-path /path/to/model \
  --tokenizer-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 30000
```

**Full pipeline: Unsloth → SGLang:**
```python
# After fine-tuning with Unsloth:
model.save_pretrained_merged("./my-model", tokenizer, save_method="merged_16bit")

# Serve with SGLang (gets RadixAttention prefix caching for free):
# sglang serve --model-path ./my-model --served-model-name my-model
```

### Tensor Parallelism + Data Parallelism

```bash
# 70B model on 4 GPUs (TP=4)
sglang serve --model-path meta-llama/Meta-Llama-3.1-70B-Instruct --tp 4

# 8B model replicated on 4 GPUs (DP=4) for throughput
sglang serve --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --dp 4

# Combined: 2 replicas, each on 2 GPUs
sglang serve --model-path meta-llama/Meta-Llama-3.1-70B-Instruct --tp 2 --dp 2
```

### Multi-Node

```bash
# Node 0 (head)
sglang serve --model-path deepseek-ai/DeepSeek-V3 \
  --tp 16 --dist-init-addr node0:25000 --nnodes 2 --node-rank 0

# Node 1
sglang serve --model-path deepseek-ai/DeepSeek-V3 \
  --tp 16 --dist-init-addr node0:25000 --nnodes 2 --node-rank 1
```

### Expert Parallelism (MoE)

```bash
# DeepSeek-V3 with expert parallelism
sglang serve --model-path deepseek-ai/DeepSeek-V3 --tp 8 --ep 8
```

### Prefill-Decode Disaggregation

```bash
# Prefill node (high compute)
sglang serve --model-path meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tp 4 --pd-mode prefill --pd-addr prefill-node:5000

# Decode node (high memory bandwidth)
sglang serve --model-path meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tp 4 --pd-mode decode --pd-addr decode-node:5000
```

### RL / Post-Training Integration

SGLang is used as rollout backend by AReaL, Miles, slime, Tunix, verl. Native support via `--sglang-for-rl` mode with vLLM-compatible weight update APIs.

## 6. Decision Table

| Factor | SGLang | vLLM | Triton Inference Server | TGI |
|--------|--------|------|------------------------|-----|
| **Best for** | Multi-turn, structured output, agentic, RL | General serving, broad ecosystem | Multi-framework, enterprise | Simple HF deployment |
| **Prefix caching** | RadixAttention (automatic, cross-request) | APC (per-request, manual enable) | None native | None native |
| **Structured output** | XGrammar compressed FSM (3-10x faster) | Outlines (slower FSM) | External | Outlines |
| **Frontend DSL** | Yes (fork, gen, select, regex) | No | No | No |
| **OpenAI API compat** | ✅ Full | ✅ Full | ✅ Via wrapper | ✅ Full |
| **Hardware** | NVIDIA/AMD/Intel/TPU/Ascend | NVIDIA/AMD/TPU | NVIDIA (primary) | NVIDIA |
| **RL integration** | Native (AReaL, verl) | Via wrapper | No | No |
| **DeepSeek optimized** | ✅ Day-0 (MLA, EP) | ✅ | ❌ | ❌ |
| **Diffusion models** | ✅ SGLang Diffusion | ❌ | ✅ | ❌ |
| **Maturity** | Production (400K+ GPUs) | Production | Production | Production |
| **Multi-LoRA** | ✅ Batched | ✅ | ✅ | ✅ |
| **Quantization** | FP4/FP8/INT4/AWQ/GPTQ | AWQ/GPTQ/FP8 | All | AWQ/GPTQ |

### When to Choose SGLang
- Multi-turn chatbots with long conversation histories (prefix caching)
- Structured output generation (JSON, regex, grammar) at scale
- Agentic systems with repetitive tool-calling loops
- RL/post-training rollout generation
- DeepSeek model family deployment
- Complex LLM programs with branching/forking

### When to Choose Alternatives
- **vLLM**: Broader community plugin ecosystem, specific PagedAttention v2 features
- **Triton**: Multi-framework (PyTorch + TensorFlow + ONNX), enterprise MLOps
- **TGI**: Quick HuggingFace model deployment with minimal config

## 7. Gotchas

1. **Frontend DSL is optional** — Most users just use the OpenAI-compatible API. The DSL (`@function`, `gen`, `fork`) is for advanced LLM program orchestration.

2. **RadixAttention works automatically** — No configuration needed. Any requests sharing prefixes benefit from KV-cache reuse. Monitor with `cached_tokens` in response metadata.

3. **Grammar backend matters** — Default XGrammar is fastest. Fall back to `--grammar-backend outlines` only for edge cases XGrammar doesn't support.

4. **`sglang serve` is the new entrypoint** — `python -m sglang.launch_server` still works but is deprecated.

5. **Port default is 30000** — Not 8000 like vLLM. Change with `--port`.

6. **Token counting** — Response metadata includes `cached_tokens` showing prefix reuse. Use this to verify RadixAttention is working.

7. **Memory pressure** — RadixAttention LRU evicts least-recently-used prefixes. For guaranteed caching, keep request rate high enough to maintain hot prefixes.

8. **Structured output quality** — Always include format instructions in the prompt even when using JSON schema constraints. The constraint guarantees syntactic validity but the prompt guides semantic correctness.

## 8. References

- **GitHub**: https://github.com/sgl-project/sglang
- **Project site**: https://sgl-project.github.io/
- **Documentation**: https://docs.sglang.ai/
- **Documentation (alt)**: https://docs.sglang.io/
- **Paper**: https://arxiv.org/abs/2312.07104 (SGLang: Efficient Execution of Structured Language Model Programs)
- **Compressed FSM Blog**: https://lmsys.org/blog/2024-02-05-compressed-fsm/
- **RadixAttention Blog**: https://lmsys.org/blog/2024-01-17-sglang/
- **v0.4 Release**: https://lmsys.org/blog/2024-12-04-sglang-v0-4/
- **SGLang v0.4 Release**: https://lmsys.org/blog/2024-12-04-sglang-v0-4/
- **PyTorch Ecosystem**: https://pytorch.org/blog/sglang-joins-pytorch/
- **Roadmap**: https://roadmap.sglang.io/
- **Slack Community**: https://slack.sglang.io/
