---
skill: vllm
trigger:
  description: "LLM inference serving, PagedAttention, vLLM deployment, model serving optimization, OpenAI-compatible API server, batch inference, continuous batching"
  triggers:
    - "serve LLM"
    - "deploy model"
    - "vllm"
    - "PagedAttention"
    - "inference server"
    - "OpenAI compatible"
    - "batch inference"
    - "model serving"
    - "GPU memory optimization"
    - "tensor parallel"
    - "quantized inference"
    - "speculative decoding"
    - "multi-LoRA"
    - "structured output"
    - "guided decoding"
globs:
  - "**/*vllm*"
  - "**/serve*.py"
  - "**/inference*.py"
---

# vLLM — High-Throughput LLM Inference Engine

## Why This Exists

LLM inference is **memory-bound**, not compute-bound. The key-value (KV) cache
that stores attention state for each request grows linearly with sequence length
and dominates GPU memory. Naive implementations pre-allocate contiguous memory
per request, causing:

1. **Internal fragmentation** — allocated blocks are larger than needed
2. **External fragmentation** — freed blocks leave unusable gaps
3. **Over-reservation** — max possible length reserved even for short outputs

This wastes 60–80% of GPU memory, directly limiting batch size and throughput.

**vLLM's PagedAttention** solves this the same way an OS solves RAM fragmentation:
virtual memory with paging. KV cache is stored in non-contiguous physical blocks
mapped via a page table, enabling near-zero waste and dramatically higher batch
sizes (2–4× throughput over naive serving).

## References

- **GitHub**: https://github.com/vllm-project/vllm
- **Documentation**: https://docs.vllm.ai
- **Paper**: https://arxiv.org/abs/2309.06180 (SOSP 2023 — "Efficient Memory Management for Large Language Model Serving with PagedAttention")
- **Blog**: https://blog.vllm.ai/2023/06/20/vllm.html
- **Supported models**: 200+ architectures (Llama, Qwen, Gemma, DeepSeek, Mixtral, etc.)

## Key Concepts

### 1. PagedAttention

Maps KV cache to fixed-size blocks (like 4KB OS pages). Requests share a
block table; physical blocks are allocated on demand and freed immediately.
Enables:
- **Zero internal fragmentation** — last block may be partially filled
- **Copy-on-write** — parallel sampling shares prefill KV across beams
- **Memory sharing** — prefix caching reuses common prompt KV blocks

### 2. Continuous Batching

Traditional static batching waits for all sequences to finish. vLLM's
**iteration-level scheduling** immediately fills freed slots with waiting
requests, keeping GPU utilization near 100%.

### 3. Prefix Caching (Automatic)

When multiple requests share a common prompt prefix (system prompt, few-shot
examples), vLLM automatically detects and reuses cached KV blocks, eliminating
redundant computation. Enable with `--enable-prefix-caching` (default in v1).

### 4. Speculative Decoding

Uses a smaller draft model (or n-gram/suffix/EAGLE/DFlash) to propose multiple
tokens, then verifies in parallel with the target model. Reduces latency for
autoregressive generation by accepting multiple tokens per forward pass.

Variants: draft model, n-gram, suffix decoding, EAGLE, MTP (Multi-Token Prediction).

### 5. Tensor/Pipeline/Data/Expert Parallelism

- **Tensor Parallel (TP)**: Split model layers across GPUs on same node
- **Pipeline Parallel (PP)**: Split layers sequentially across nodes
- **Data Parallel (DP)**: Replicate model, shard requests
- **Expert Parallel (EP)**: Distribute MoE experts across GPUs
- **Context Parallel (CP)**: Split long sequences across GPUs

### 6. Quantization

Supported formats (zero code changes — specify at load time):
| Format | Precision | Method |
|--------|-----------|--------|
| FP8 | W8A8 | Per-tensor/per-channel scaling |
| GPTQ | W4A16 | Post-training quantization |
| AWQ | W4A16 | Activation-aware weight quantization |
| INT8 | W8A8 | SmoothQuant-style |
| GGUF | Mixed | llama.cpp format |
| NVFP4 | W4A8 | NVIDIA Blackwell native |
| MXFP8/MXFP4 | Mixed | Microscaling formats |
| BitsAndBytes | 4/8-bit | QLoRA-compatible |

### 7. Structured Outputs (Guided Decoding)

Constrain generation to valid JSON schemas, regex patterns, or grammars using
xgrammar backend. Guarantees well-formed output without post-processing.

### 8. Multi-LoRA Serving

Serve multiple LoRA adapters from a single base model simultaneously. Adapters
are loaded/unloaded dynamically per request via the `--lora-modules` flag or
API parameter. Supports dense and MoE layers.

## Code Examples

### Offline Batch Inference (LLM class)

```python
from vllm import LLM, SamplingParams

# Load model — auto-downloads from HuggingFace
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    gpu_memory_utilization=0.90,      # Use 90% of GPU memory for KV cache
    max_model_len=8192,               # Max context window
    tensor_parallel_size=1,           # Number of GPUs for TP
)

# Sampling configuration
params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    stop=["<|eot_id|>"],
)

# Batch generation — automatically handles scheduling
prompts = [
    "Explain PagedAttention in one paragraph.",
    "Write a Python quicksort in 10 lines.",
    "What is the capital of France?",
]
outputs = llm.generate(prompts, params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Output: {output.outputs[0].text}\n")
```

### Chat Interface (applies chat template automatically)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
params = SamplingParams(temperature=0.8, max_tokens=1024)

messages_list = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain transformers briefly."},
    ],
    [
        {"role": "user", "content": "Write a haiku about GPU memory."},
    ],
]

outputs = llm.chat(messages_list, params)
for out in outputs:
    print(out.outputs[0].text)
```

### Online Serving (OpenAI-Compatible Server)

```bash
# Start server — drop-in replacement for OpenAI API
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --api-key my-secret-key
```

```python
# Client — standard OpenAI SDK
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="my-secret-key")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

### AsyncLLMEngine (Embed in Applications)

```python
import asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

async def main():
    args = AsyncEngineArgs(
        model="meta-llama/Llama-3.1-8B-Instruct",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
    )
    engine = AsyncLLMEngine.from_engine_args(args)

    params = SamplingParams(temperature=0.8, max_tokens=256)

    # Stream tokens as they're generated
    request_id = "req-001"
    async for output in engine.generate("Tell me a joke", params, request_id):
        if output.finished:
            print(output.outputs[0].text)

asyncio.run(main())
```

### Structured Output (JSON Schema)

```python
from vllm import LLM, SamplingParams
import json

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "hobbies": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age", "hobbies"],
}

params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
    guided_decoding={
        "json": schema,  # Guarantees valid JSON matching schema
    },
)

outputs = llm.generate(["Generate a person profile:"], params)
result = json.loads(outputs[0].outputs[0].text)
# result is guaranteed to be valid against the schema
```

### Structured Output (Regex)

```python
params = SamplingParams(
    max_tokens=32,
    guided_decoding={
        "regex": r"\d{3}-\d{2}-\d{4}",  # US SSN format
    },
)
```

### Multi-LoRA Serving

```bash
# Start server with multiple LoRA adapters
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules \
        sql-lora=path/to/sql-adapter \
        code-lora=path/to/code-adapter \
    --max-loras 4 \
    --max-lora-rank 64
```

```python
# Request targets specific adapter
response = client.chat.completions.create(
    model="sql-lora",  # Route to specific LoRA
    messages=[{"role": "user", "content": "SELECT all users..."}],
)
```

### Speculative Decoding

```bash
# Use smaller draft model for faster generation
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.1-8B-Instruct \
    --num-speculative-tokens 5 \
    --speculative-disable-mqa-scorer

# Or use n-gram speculation (no extra model needed)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --speculative-model "[ngram]" \
    --ngram-prompt-lookup-max 4
```

## Performance Tuning

### Critical Parameters

| Parameter | Default | Purpose | Guidance |
|-----------|---------|---------|----------|
| `gpu_memory_utilization` | 0.90 | Fraction of GPU memory for KV cache | Increase to 0.95 for throughput; lower if OOM |
| `max_model_len` | Auto | Maximum sequence length | Set explicitly to avoid over-allocation |
| `tensor_parallel_size` | 1 | GPUs for tensor parallelism | Use when model doesn't fit 1 GPU |
| `enforce_eager` | False | Disable CUDA graphs | Set True for debugging; False for production (2-3× decode speedup) |
| `max_num_seqs` | 256 | Max concurrent sequences | Increase for throughput; decrease for latency |
| `enable_chunked_prefill` | True (v1) | Interleave prefill/decode | Reduces TTFT variance under load |
| `enable_prefix_caching` | True (v1) | Cache common prefixes | Always enable for shared system prompts |
| `quantization` | None | Weight quantization method | "fp8", "awq", "gptq", "gguf" |
| `kv_cache_dtype` | auto | KV cache precision | "fp8_e5m2" saves 50% KV memory |
| `max_num_batched_tokens` | Auto | Tokens per iteration | Tune for prefill/decode balance |

### CUDA Graphs vs Eager Mode

CUDA graphs capture and replay GPU kernel sequences, eliminating CPU launch
overhead. vLLM captures graphs for common batch sizes during warmup.

- **Default (graphs enabled)**: 2-3× faster decode, ~30s warmup
- **`--enforce-eager`**: No warmup, useful for debugging or dynamic shapes
- For variable-length requests: graphs + chunked prefill handles this well

### Memory Conservation Strategies

```bash
# Fit larger models in less memory
vllm serve large-model \
    --quantization fp8 \           # 50% weight memory reduction
    --kv-cache-dtype fp8_e5m2 \    # 50% KV cache reduction
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \         # Limit context if possible
    --enable-prefix-caching         # Share KV across requests
```

## Deployment

### Docker

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2
```

### Serve Your Own Fine-Tuned Model

```bash
# Option 1: Merged model (from Unsloth save_pretrained_merged or HF merge)
# Just point --model to your local directory
vllm serve /path/to/my-merged-model \
  --served-model-name my-custom-model \
  --gpu-memory-utilization 0.9

# Option 2: Base model + LoRA adapter (hot-swappable)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --lora-modules my-adapter=/path/to/lora-adapter \
  --max-lora-rank 64

# Option 3: GGUF quantized model (from Unsloth save_pretrained_gguf)
vllm serve /path/to/model.gguf \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct

# Option 4: Docker with local model volume-mounted
docker run --runtime nvidia --gpus all \
  -v /home/user/models/my-model:/model \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model /model \
  --served-model-name my-custom-model
```

**After Unsloth fine-tuning → vLLM serving pipeline:**
```python
# 1. Fine-tune with Unsloth
model.save_pretrained_merged("./my-model", tokenizer, save_method="merged_16bit")

# 2. Serve immediately
# vllm serve ./my-model --served-model-name my-model

# Or export GGUF for quantized serving:
model.save_pretrained_gguf("./my-model-gguf", tokenizer, quantization_method="q4_k_m")
# vllm serve ./my-model-gguf --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

### Kubernetes (Helm)

```yaml
# values.yaml
replicaCount: 1
image:
  repository: vllm/vllm-openai
  tag: latest
resources:
  limits:
    nvidia.com/gpu: 2
args:
  - "--model"
  - "meta-llama/Llama-3.1-8B-Instruct"
  - "--tensor-parallel-size"
  - "2"
  - "--gpu-memory-utilization"
  - "0.92"
```

### Ray Multi-Node (Tensor Parallel Across Nodes)

```bash
# Head node
ray start --head

# Worker node
ray start --address=<head-ip>:6379

# Serve with 4 GPUs across 2 nodes
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --distributed-executor-backend ray
```

### Production Flags

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --disable-log-requests \          # Reduce log noise
    --max-num-seqs 512 \              # Higher concurrency
    --api-key $VLLM_API_KEY \         # Auth
    --uvicorn-log-level warning \
    --served-model-name my-model      # Custom model name in API
```

## When to Use

| Scenario | Use vLLM? | Why |
|----------|-----------|-----|
| High-throughput serving (100+ req/s) | ✅ Yes | PagedAttention + continuous batching |
| OpenAI API drop-in replacement | ✅ Yes | Full compatibility, streaming, tool calling |
| Multi-LoRA serving | ✅ Yes | Dynamic adapter routing per request |
| Offline batch processing | ✅ Yes | LLM class with auto-batching |
| Structured output / JSON | ✅ Yes | xgrammar guided decoding |
| Multi-modal (vision + text) | ✅ Yes | LLaVA, Qwen-VL, Pixtral, etc. |
| Embedding / reward models | ✅ Yes | Pooling model support |
| Edge / laptop deployment | ❌ No | Use Ollama or llama.cpp |
| Mobile / browser inference | ❌ No | Use MLC-LLM or WebLLM |
| Fine-tuning / training | ❌ No | Use vLLM's RLHF integration with TRL, or dedicated training frameworks |
| Simple single-user local chat | ⚠️ Maybe | Ollama simpler; vLLM if you need throughput |

## Decision Table: vLLM vs Alternatives

| Feature | vLLM | TGI (HF) | Triton + TensorRT-LLM | SGLang | Ollama |
|---------|------|-----------|------------------------|--------|--------|
| **Throughput** | ★★★★★ | ★★★★ | ★★★★★ | ★★★★★ | ★★ |
| **Ease of use** | ★★★★ | ★★★★ | ★★ | ★★★ | ★★★★★ |
| **Model support** | 200+ | ~100 | ~50 | ~100 | ~100 (GGUF) |
| **Quantization** | FP8/AWQ/GPTQ/INT8/GGUF/NVFP4 | AWQ/GPTQ/BnB | FP8/INT8/INT4 | AWQ/FP8 | GGUF only |
| **Multi-LoRA** | ✅ Native | ✅ | ❌ | ✅ | ❌ |
| **Speculative decode** | ✅ Multiple methods | ✅ | ✅ | ✅ | ❌ |
| **Structured output** | ✅ (xgrammar) | ✅ (outlines) | ❌ | ✅ (native) | ❌ |
| **OpenAI API** | ✅ Full | ✅ Partial | ❌ | ✅ Full | ✅ Partial |
| **Multi-node TP** | ✅ (Ray) | ❌ | ✅ | ✅ | ❌ |
| **Hardware** | NVIDIA/AMD/TPU/CPU/Ascend | NVIDIA/AMD | NVIDIA only | NVIDIA/AMD | CPU/NVIDIA (GGUF) |
| **Production ready** | ✅ | ✅ | ✅ | Growing | Dev/hobby |
| **RLHF integration** | ✅ (weight transfer) | ❌ | ❌ | ❌ | ❌ |
| **Best for** | General production LLM serving | HF ecosystem users | Max NVIDIA perf, strict latency SLAs | Research, complex prompting, RadixAttention | Local dev, single-user |

### Quick Selection Guide

- **Default choice for production**: vLLM — broadest model support, easiest deployment
- **Maximum NVIDIA performance**: TensorRT-LLM via Triton — lower latency but harder to operate
- **Complex prompting / research**: SGLang — RadixAttention and native structured generation
- **Local development / experimentation**: Ollama — one-command setup, GGUF models
- **Tight HuggingFace ecosystem**: TGI — seamless HF integration

## Gotchas

1. **First request is slow** — CUDA graph warmup captures kernels for common batch sizes. Use `--enforce-eager` to skip (at the cost of decode speed) or pre-warm with a dummy request.

2. **OOM on long contexts** — `max_model_len` defaults to model's max (often 128K+). Set it explicitly to what you actually need: `--max-model-len 8192`.

3. **Chat template not applied in `llm.generate()`** — The `generate()` method takes raw text. Use `llm.chat()` for messages, or manually apply the template via tokenizer.

4. **`generation_config.json` overrides your params** — vLLM applies the model's generation config by default. Pass `--generation-config vllm` to use vLLM defaults instead.

5. **Tensor parallel requires matching GPU count** — `tensor_parallel_size` must equal the number of available GPUs. Use `CUDA_VISIBLE_DEVICES` to control which GPUs are used.

6. **FlashInfer not bundled** — Pre-built wheels don't include FlashInfer. Install it separately if you want the FlashInfer attention backend.

7. **Model download on first run** — Models are downloaded from HuggingFace on first use. Pre-download with `huggingface-cli download` for containerized deployments.

8. **LoRA with quantized base** — Not all quantization methods support LoRA. FP8 + LoRA works; GPTQ + LoRA has limited support.

9. **Disaggregated prefill is experimental** — Separating prefill and decode workers is available but marked experimental. Test thoroughly before production use.

10. **KV cache dtype fp8 reduces quality slightly** — FP8 KV cache (50% memory saving) introduces small numerical differences. Benchmark your specific model before deploying.

## Installation

```bash
# Recommended: uv (fast Python package manager)
uv pip install vllm --torch-backend=auto

# Or pip
pip install vllm

# For AMD ROCm
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/

# For TPU
uv pip install vllm-tpu
```

Requires: Linux, Python 3.10–3.13, NVIDIA GPU (Ampere+), CUDA 12.x.
Also supports: AMD ROCm 7.0, Google TPU, Intel Gaudi, Apple Silicon (via vLLM-Metal).
