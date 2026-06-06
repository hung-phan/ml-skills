---
name: inference-optimization
description: LLM inference optimization — TTFT/TPOT/goodput metrics, prefill-decode disaggregation, continuous batching, PagedAttention, prefix caching, speculative decoding, inference with reference, model compression, framework choice (vLLM/SGLang/TensorRT-LLM/TGI). Use when latency or cost SLOs are breaching, choosing an inference engine, evaluating managed inference providers, or sizing GPUs for serving.
---

# LLM Inference Optimization

## Why This Exists

**Problem**: A naively-served LLM burns money and frustrates users. Inference has bottlenecks unlike training — autoregressive decoding loads the whole weight matrix from HBM to generate one token, so most of the GPU's compute is idle. Default HuggingFace `transformers.generate` will give you single-digit MFU on an H100. The choice of engine, batching strategy, and KV-cache layout dominates the choice of model size: a well-served 70B at INT4 with continuous batching can be cheaper *and* lower latency than a poorly-served 7B at FP16.

**Key insight**: The two phases want different machines.
- **Prefill** (processing the prompt) is **compute-bound** — every input token gets the full forward pass in parallel, so FLOP/s is the limit.
- **Decode** (generating output tokens) is **memory-bandwidth-bound** — each new token requires loading the full weight matrix from HBM, but only does a tiny amount of math per byte loaded.

If you serve them on the same machine, prefill steals compute from decode and TPOT spikes whenever a long-context request lands. The state-of-the-art (DistServe, Splitwise, DeepSeek serving) is to disaggregate: run prefill on a smaller pool of compute-heavy GPUs, ship the KV cache over NVLink/InfiniBand, and decode on memory-bandwidth-heavy GPUs.

**Reach for this when**:
- TTFT or TPOT is breaching an SLO (or you're about to write one).
- GPU utilization looks fine in nvidia-smi but `cost/1M tokens` is high (it lies — MFU/MBU are the real metrics).
- Choosing between vLLM, SGLang, TensorRT-LLM, TGI, or llama.cpp.
- Evaluating a managed provider's `$/1M tokens` price — and deciding whether self-hosting wins.
- Sizing GPUs (H100 vs L40S vs A100) for a known QPS / context-length distribution.
- Long-context use case where the KV cache eats all your HBM.

---

## 1. Mental Model: Roofline, Prefill vs Decode, MBU vs MFU

The roofline model (Williams et al. 2009) says every operation is either compute-bound or memory-bandwidth-bound, depending on its arithmetic intensity (FLOP per byte loaded). For an H100 SXM with ~989 TFLOP/s (TF32) and ~3 TB/s HBM, the crossover is around 330 FLOP/byte. A matmul with batch_size=1 on a 70B model has arithmetic intensity around 2 FLOP/byte — deeply bandwidth-bound. The same matmul at batch_size=256 has intensity around 500 FLOP/byte — now compute-bound.

**This is why batching is the single biggest throughput win in LLM serving**: batching shifts decode from the bandwidth-bound regime (where the GPU is mostly idle waiting for HBM) into the compute-bound regime (where the tensor cores are doing useful work).

| Phase  | Per-token cost                                  | Bottleneck         | Wants                                |
|--------|-------------------------------------------------|--------------------|--------------------------------------|
| Prefill | O(N²) attention + O(N·d²) FFN, all parallel    | FLOP/s             | High-FLOP chips, long inputs batched |
| Decode  | O(d²) FFN + O(N·d) attention, sequential       | HBM bandwidth      | High-bandwidth chips, large batches  |

**MFU vs MBU**:
- **MFU** (Model FLOP/s Utilization) = observed throughput / peak FLOP/s. High during prefill.
- **MBU** (Model Bandwidth Utilization) = (params × bytes/param × tokens/s) / peak HBM bandwidth. High during decode.

For a 7B model in FP16 on an A100-80GB (2 TB/s HBM) generating 100 tokens/s, MBU = 7B × 2 × 100 / 2TB = 70%. That's healthy. If you observe MBU < 30% for batch=1 decode, your kernel or memory layout is wrong.

For training, MFU > 50% is healthy. For inference *prefill*, MFU > 40% is healthy. For inference *decode*, MFU is usually low — focus on MBU instead. On H100 with vLLM continuous batching at batch=64 on a 7B in FP16, expect MBU around 60–80%.

---

## 2. Performance Metrics — What to Actually Track

| Metric | Definition | Use for | Watch out for |
|--------|------------|---------|---------------|
| **TTFT** | Time to first token (covers prefill) | Chatbot, streaming UX | Scales with input length |
| **TPOT** | Time per output token, post-first | Long-output UX | Often a flat number; varies by batch |
| **ITL / TBT** | Inter-token latency | Stutter detection | NVIDIA uses ITL, LinkedIn uses TBT |
| **e2e latency** | TTFT + TPOT × N_out | Total request time | The user-visible number |
| **Throughput (tok/s)** | Output tokens/s across all users | Cost/1M-tok calc | Decode tok/s ≠ prefill tok/s — track separately |
| **Goodput** | Requests/s satisfying both TTFT and TPOT SLOs | Real production health | Average throughput hides SLO violations |
| **MFU / MBU** | FLOP and HBM utilization | Hardware efficiency | nvidia-smi GPU% is meaningless |
| **KV-cache hit rate** | Fraction of prefill tokens served from prefix cache | Prompt-cache effectiveness | Often dominates cost in agentic/RAG workloads |
| **Cost / 1M output tokens** | (machine $/h × 3600) / (tok/s × 1e6) | Self-host vs managed | Decode cost ≠ prefill cost |

**Always report percentiles** (p50, p95, p99). Mean TTFT is misleading — one 3000ms outlier in a 100-request sample turns 100ms p50 into 130ms mean. Latency is a long-tailed distribution; goodput is what tells you whether the service is actually healthy.

```python
# Compute MFU / MBU from a measurement
def mbu(params: float, bytes_per_param: float, tokens_per_sec: float,
        peak_bandwidth_gb_s: float) -> float:
    """Model Bandwidth Utilization (0..1)."""
    used_bw_gb_s = params * bytes_per_param * tokens_per_sec / 1e9
    return used_bw_gb_s / peak_bandwidth_gb_s

def cost_per_million_output_tokens(machine_cost_per_hour: float,
                                    output_tokens_per_sec: float) -> float:
    return machine_cost_per_hour / 3600 / output_tokens_per_sec * 1e6

# Llama-3-70B INT4 on H100 (3 TB/s HBM, ~$3/h on-demand) at 80 tok/s decode:
# MBU = 70e9 * 0.5 * 80 / 1e9 / 3000 ≈ 0.93   (excellent)
# Cost/1M tok = 3 / 3600 / 80 * 1e6 ≈ $10.4
```

---

## 3. Hardware: The Memory Hierarchy That Matters

LLM serving lives or dies on **HBM bandwidth**. Three tiers, with order-of-magnitude numbers:

| Tier | Size | Bandwidth | Where the weights live |
|------|------|-----------|------------------------|
| GPU SRAM (L1/L2 + shared) | ~40 MB | ~10 TB/s | Tile being currently computed |
| GPU HBM (HBM2e, HBM3, HBM3e) | 24–141 GB | 1.5–5 TB/s | Full model + KV cache |
| CPU DRAM (DDR4/5) | TB | 25–50 GB/s | Cold weights, model loading |

If your model spills out of HBM into CPU DRAM, decode throughput drops by ~30–60×. **Never serve LLMs with weight offloading in latency-sensitive paths.** For very large models that don't fit, use tensor parallelism across GPUs with NVLink (900 GB/s on H100) or InfiniBand (400 Gb/s) — but only if interconnect is fast enough. PCIe Gen4 at 64 GB/s is *not* fast enough for tensor parallelism on a 70B model.

**Tensor core formats by chip generation**:
- A100 (Ampere): FP16 / BF16 / TF32 / INT8.
- H100 (Hopper): adds **FP8** (E4M3, E5M2) — 2× FLOP/s over FP16.
- B100/B200 (Blackwell): adds **FP4** and microscaling (MX) formats — another 2× over FP8.

FP8 on H100 is the new default for serving frontier-scale models. KV cache in FP8 with weights in FP8 gives you ~2× the effective HBM compared to FP16.

---

## 4. Model-Level Optimizations

### 4.1 Quantization

See [`../../ml-architectures/quantization/`](../../ml-architectures/quantization/) for full coverage. Summary for serving:

| Scheme | What's quantized | Hardware | Quality hit | When to use |
|--------|------------------|----------|-------------|-------------|
| **INT8 weight-only** | Weights only; activations stay FP16 | All modern GPUs | Negligible (<0.5% on most evals) | Default first step. Always try this. |
| **INT4 GPTQ / AWQ** | Weights only; activations FP16 | All modern GPUs | Small (1–3%) on most tasks; can be larger on math/code | Memory-constrained serving (70B on 1×H100, 13B on 1×L40S) |
| **FP8 (E4M3 weights + activations)** | Both | H100+, MI300+ | Negligible if calibrated | Frontier serving — uses tensor cores |
| **KV-cache quantization (INT8/FP8)** | KV cache only | Any | Small | Long-context workloads where KV dominates HBM |
| **AQLM / QuIP#** | Weights to ~2 bits | All modern GPUs | Moderate | Research / extreme memory pressure |
| **BitsAndBytes NF4** | Weights to 4-bit (LoRA training) | All modern GPUs | Moderate | QLoRA training, *not* production serving |

**Pareto frontier intuition**: For most production tasks the order from "best quality per byte" to "worst" is: FP16 ≈ BF16 → FP8 → INT8 → INT4 (AWQ/GPTQ) → INT4 (BnB-NF4) → AQLM. Cliff is usually below 4 bits. Math/code/reasoning tasks degrade earlier than chat/summarization. **Always benchmark on your eval set** — generic MMLU numbers don't tell you what happens on your domain.

Quantization is **not free**. The cost is engineering: GPTQ/AWQ require a calibration dataset (~128 samples, drawn from a distribution close to your inference traffic), and a bad calibration set produces a worse model than INT8.

### 4.2 Distillation

Teacher → student pattern: train a small model to imitate a large model's output distribution (KL divergence on logits or response-distillation on generated text). The student is faster *forever*, unlike inference-time tricks. Examples in the wild:
- **DistilBERT** (Sanh et al. 2019): 60% of BERT's size, 97% of GLUE.
- **Phi-2 / Phi-3** (Microsoft 2023–2024): teacher-distilled "textbooks" data → 2.7B–3.8B model competitive with 7B+.
- **TinyLlama**: trained from scratch with Llama-2 distillation losses.
- **Gemma 2 9B** distilled from a much larger Gemma teacher.

Use distillation when you control training and have a fixed task. For general-purpose chat, off-the-shelf small models (Llama-3-8B, Qwen-2.5-7B) usually win.

### 4.3 Pruning

Two flavors:
- **Structured pruning**: remove whole heads, neurons, layers. Yields a smaller dense model. Hardware-friendly. Gradual magnitude pruning during/after training.
- **Unstructured pruning**: zero individual weights. Yields a sparse model. Requires hardware support (NVIDIA 2:4 sparsity on A100+ for ~2× speedup) — without it, you save memory but not compute.

In practice, pruning underdelivers vs paper claims. The "lottery ticket hypothesis" papers (Frankle & Carbin 2019) showed 90% sparsity can preserve accuracy, but the delivered speedup on real hardware is usually 1.2–1.5×, not 10×. **Quantization beats pruning on most workloads.** Consider pruning only when (a) you control training, (b) you've maxed out quantization, (c) your hardware exploits the sparsity pattern.

---

## 5. Decoding-Bottleneck Optimizations

Autoregressive decode is sequential by definition: token N+1 depends on token N. Every trick below tries to break or amortize that dependency by exploiting the fact that decode is bandwidth-bound — the GPU has spare FLOPs you're not using.

### 5.1 Speculative Decoding

A small **draft model** proposes K tokens, the **target model** verifies all K in a single parallel forward pass. Accept the longest prefix the target agrees with, plus one bonus token from the target. Worst case: target generates one token (same as no spec). Best case: K+1 tokens per target call.

Why it works: target verification is parallel (compute-bound, like prefill), but generation is sequential (bandwidth-bound). At low MBU, you have free FLOPs to run the verification.

```python
# Speculative decoding skeleton (illustrative — vLLM/SGLang implement this for you)
import torch

@torch.no_grad()
def speculative_step(target_model, draft_model, prefix_ids, K=5, temperature=0.0):
    """Generate up to K+1 new tokens with one target forward pass."""
    # 1. Draft proposes K tokens (sequentially, but cheap because draft is small)
    draft_ids = prefix_ids.clone()
    draft_logits = []
    for _ in range(K):
        out = draft_model(draft_ids).logits[:, -1, :]
        next_tok = out.argmax(dim=-1, keepdim=True) if temperature == 0 \
                   else torch.multinomial(out.softmax(-1), 1)
        draft_logits.append(out)
        draft_ids = torch.cat([draft_ids, next_tok], dim=-1)

    # 2. Target verifies all K in ONE forward pass
    target_logits = target_model(draft_ids).logits[:, -K-1:, :]  # [batch, K+1, vocab]

    # 3. Accept longest prefix where target's argmax == draft's choice
    accepted = 0
    for i in range(K):
        target_next = target_logits[:, i, :].argmax(dim=-1)
        draft_next = draft_ids[:, prefix_ids.shape[-1] + i]
        if (target_next == draft_next).all():
            accepted += 1
        else:
            break

    # 4. Bonus token from target on the prefix where draft was correct
    bonus = target_logits[:, accepted, :].argmax(dim=-1, keepdim=True)
    new_ids = torch.cat([prefix_ids,
                         draft_ids[:, prefix_ids.shape[-1]:prefix_ids.shape[-1] + accepted],
                         bonus], dim=-1)
    return new_ids, accepted + 1
```

**Tuning**:
- **Acceptance rate** is domain-dependent: 60–80% on code/structured tasks, 40–60% on chat, 20–40% on creative writing. Below 40%, the overhead can outweigh the savings.
- **Draft model size**: usually 5–20× smaller than target. Llama-3-70B + Llama-3-8B works. Train a custom 1B draft for a 70B target if you can.
- **K** (draft length): larger K = fewer target calls but lower acceptance. K=4–7 is the sweet spot.
- **Don't combine with already-saturated MBU**: if your MBU is already 90% (large batch, well-tuned engine), spec decoding gives nothing — the FLOPs aren't free.
- vLLM, TensorRT-LLM, SGLang, llama.cpp all support it. In vLLM: `speculative_model="..."` + `num_speculative_tokens=K`.

### 5.2 Inference with Reference (LLMA)

(Yang et al. 2023.) When the output is likely to repeat spans from the input — RAG, code editing, agentic tool-use, multi-turn conversations — *copy* the draft tokens directly from the prompt instead of running a draft model. No extra model required. ~2× speedup on retrieval-augmented and code-completion workloads.

The algorithm: at each decoding step, find the longest matching span between the recent output and any span in the prompt; speculatively use that span as the draft; verify with the target model in parallel. Use this when you know your workload has high prompt-output overlap.

### 5.3 Self-Speculation: Medusa, Lookahead, EAGLE

No separate draft model. The target model itself proposes its future tokens.

| Method | Mechanism | Train extra? | Typical speedup |
|--------|-----------|--------------|-----------------|
| **Medusa** (Cai et al. 2024) | Add K extra "decoding heads" to the model, each predicts position N+k+1. Tree-attention verifies. | Yes — train heads (frozen base) | 1.5–2× on Llama-2 |
| **Lookahead decoding** (Fu et al. 2024) | Jacobi iteration on multiple positions in parallel; no extra params | No | 1.5–2.3× |
| **EAGLE / EAGLE-2** | Draft is a small transformer over the target's hidden states | Yes — train draft | 2–3× on chat |

NVIDIA reported Medusa gave Llama-3.1 ~1.9× decoding speedup on H200. Most production engines now support at least one of these.

---

## 6. Attention-Level Optimizations

See [`../../ml-architectures/attention/`](../../ml-architectures/attention/) for FlashAttention, GQA, MLA, and PagedAttention depth. Serving-relevant summary:

### 6.1 PagedAttention (vLLM)

The KV cache is the second-biggest HBM consumer after the weights. Naively, you allocate `max_seq_len × hidden × 2` per request — most of which is wasted (most requests don't reach `max_seq_len`).

**PagedAttention** (Kwon et al. 2023, the vLLM paper) divides the KV cache into fixed-size **pages** (typically 16 tokens × hidden). Pages are allocated lazily as the sequence grows. Pages can be **shared** across requests (prefix caching), and freed pages can be reused. Memory fragmentation drops from ~60–80% wasted to under 5%, which roughly **doubles** effective batch size.

```text
Logical KV cache (per request)         Physical pages (shared pool)
  request A: [tok0...tok47]   --------> [P0][P1][P2]
  request B: [tok0...tok31]   --------> [P3][P4]
  request C: [shares prefix of A]
             [tok0..tok15][tok16..]   ->[P0]   [P5]    <-- P0 SHARED with A
```

PagedAttention is the default in vLLM and SGLang and the de facto baseline; if your engine doesn't have a paged KV cache, switch.

### 6.2 FlashAttention 1/2/3

Tiling + kernel fusion that keeps attention computation in SRAM instead of round-tripping through HBM. FlashAttention-1 (Dao et al. 2022) gave ~2× speedup. FA-2 (Dao 2023) reorganized parallelism for ~2× over FA-1. FA-3 (Shah et al. 2024) targets H100's async tensor cores and FP8, ~1.5–2× over FA-2.

You don't write FlashAttention — you make sure your engine uses it. vLLM, SGLang, TensorRT-LLM all do.

### 6.3 GQA / MLA — KV-cache reduction by architecture

- **MQA** (Shazeer 2019): one KV head, many query heads. Maximum compression but quality hit.
- **GQA** (Ainslie et al. 2023): groups of query heads share a KV head. Llama-2-70B uses 8 KV heads / 64 query heads — 8× smaller KV cache than MHA. Most modern open models (Llama-3, Mistral, Qwen-2.5) use GQA.
- **MLA** (DeepSeek-V2/V3): low-rank latent projection of KV; ~10× smaller KV cache than MHA. Requires the model to be designed for it.

This is a model-architecture choice — you pick a model that already has GQA/MLA. You can't bolt it on at serving time.

### 6.4 KV-Cache Sharing: Prefix Caching / RadixAttention

Two requests with the same prefix can share that prefix's KV cache. Crucial for:
- System prompts (1k–10k tokens repeated across all requests).
- Multi-turn conversations (each turn adds to a shared prefix).
- RAG with shared retrieved chunks.
- Few-shot prompts.

**SGLang's RadixAttention** indexes KV pages by token-prefix in a radix tree, giving automatic O(1) prefix-hit detection. **vLLM**'s `enable_prefix_caching=True` does the same with a hash-based scheme. Anthropic, OpenAI, and Google offer prompt caching as a managed feature with 50–90% cost reduction on cached input tokens.

When prefix-cache hit rate is high (>50%), it's typically the single biggest cost reduction available — bigger than quantization.

---

## 7. Service-Level Optimizations

### 7.1 Continuous (In-Flight) Batching — The Biggest Win

In **static batching**, a batch waits for all requests to finish before any can return. If request A wants 10 tokens and request B wants 1000 tokens, A waits for B. Throughput is wasted; tail latency explodes.

In **continuous batching** (Orca, Yu et al. 2022; vLLM): when any request in the batch finishes, it leaves the batch immediately, and a new request slots into the freed sequence position on the next decode step. The batch composition changes every step.

| Batching | Throughput | TTFT | TPOT under skew | Implementation |
|----------|------------|------|------------------|----------------|
| **Static** | Low — pads to max_len | High (waits to fill) | Bad — slowest dominates | Easy |
| **Dynamic** | Medium — flushes on time/size | Bounded | Bad — same problem within batch | Medium |
| **Continuous / in-flight** | High — slots refill mid-batch | Low | Good — independent per request | Hard (engine internals) |

Continuous batching is the default in vLLM, SGLang, TensorRT-LLM, and TGI. **If your engine doesn't have it, change engines** — this single change is typically a 5–20× throughput improvement on mixed-length traffic.

### 7.2 Prefill–Decode Disaggregation (PD-Disagg)

Prefill is compute-bound; decode is bandwidth-bound. On the same machine they fight: a single long-prompt request triggers a heavy prefill that stalls all in-flight decodes, spiking TPOT. **Solution**: run prefill on one set of GPUs, ship the KV cache over NVLink/InfiniBand, decode on another set.

Adopted by:
- **DistServe** (Zhong et al. 2024) — academic baseline.
- **Splitwise** (Patel et al. 2024, MSR) — Azure-internal.
- **DeepSeek's serving infra** — H800-based, with custom KV-transfer over IB.

Ratio of prefill GPUs to decode GPUs depends on the workload:
- Long inputs, short outputs (RAG QA, coding assistant): **2:1 to 4:1** prefill:decode.
- Short inputs, long outputs (creative writing, agents): **1:2 to 1:1** prefill:decode.

Communication overhead is real but small — KV-cache transfer at 600 GB/s NVLink takes ~10–50 ms for a typical 70B request.

When to deploy: when TTFT and TPOT SLOs *both* matter and you're seeing TPOT spikes correlated with prefill traffic. For small clusters (1–4 GPUs), don't bother — the disaggregation overhead exceeds the gain.

### 7.3 Tensor / Pipeline / Expert Parallelism at Serving Time

- **Tensor parallelism (TP)**: shard each matmul across GPUs (column-parallel + row-parallel). Reduces both memory and latency. Communication: AllReduce per layer. Needs **fast interconnect** — NVLink (900 GB/s on H100) is fine; PCIe is not. Standard for 70B+ on 2–8 GPUs in one node.
- **Pipeline parallelism (PP)**: shard layers across GPUs. Adds latency (sequential pipeline stages), hurts TTFT. Used when models exceed a single node and the alternative is offload-to-CPU.
- **Expert parallelism (EP)**: shard MoE experts across GPUs. Used by DeepSeek-V3, Mixtral. Requires AlltoAll communication — IB ≥ 200 Gbps recommended.

Default at serving time: **TP within a node, replica across nodes**. PP only when forced to.

### 7.4 Streaming Responses

Stream tokens as generated. Doesn't reduce real latency — reduces *perceived* latency. Always do this for chat. Engines emit tokens via SSE / WebSocket / gRPC streaming.

### 7.5 Online vs Batch APIs

| API | Cost | Latency | When |
|-----|------|---------|------|
| **Online** | Higher | Seconds | Chat, code-completion, anything user-facing |
| **Batch** | ~50% off (OpenAI, Gemini) | Hours | Synthetic data, periodic reports, reindexing |

If you self-host, replicate this internally: tag low-urgency requests, push them to a separate queue served by a higher-batch-size engine on cheaper hardware.

---

## 8. Framework Decision Table

| Engine | Best for | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| **vLLM** | General-purpose default | PagedAttention, continuous batching, prefix caching, broad model support, easy to deploy | Less aggressive on lowest-latency single-request than TRT-LLM |
| **SGLang** | Structured outputs, multi-turn, agents | RadixAttention prefix caching, structured generation, fastest in many recent benchmarks | Younger, smaller community |
| **TensorRT-LLM** | Lowest latency on NVIDIA | Fastest single-request on Hopper/Blackwell, FP8 native, deep TRT integration | Hard to deploy, NVIDIA-only, build-time engine compilation per model+shape |
| **TGI (HuggingFace Text Generation Inference)** | Managed-style HF stack | Solid defaults, HF Hub integration, Rust core | Behind vLLM/SGLang on raw throughput |
| **llama.cpp** | CPU / edge / quantized GGUF | Runs anywhere — Mac, ARM, x86, mobile; great GGUF quant support | Not a high-throughput multi-tenant server |
| **Triton Inference Server** | Multi-model serving, framework-agnostic | Deploy LLM + embeddings + reranker behind one server, with TRT-LLM/vLLM/Python backends | Not LLM-specific itself; you still pick an LLM backend |

**Decision rule**:
- Default to **vLLM**.
- Switch to **SGLang** if you have heavy prefix-sharing / multi-turn / structured-output workloads.
- Switch to **TensorRT-LLM** if you are H100/H200/B200-only and need the absolute lowest p50 TTFT — accept the deploy complexity.
- **llama.cpp** for CPU/Mac/edge/laptop or aggressive quantization (GGUF Q4_K_M is excellent for local).
- **Triton** when you serve multiple models behind one endpoint (LLM + embedder + reranker).

### 8.1 vLLM Example: Speculative + Prefix Caching + INT4

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    quantization="awq",                      # INT4 weight-only via AWQ
    dtype="float16",                         # activations stay FP16
    tensor_parallel_size=2,                  # shard across 2 GPUs
    max_model_len=8192,
    enable_prefix_caching=True,              # share KV across same-prefix requests
    speculative_model="meta-llama/Meta-Llama-3-8B-Instruct",
    num_speculative_tokens=5,
    gpu_memory_utilization=0.9,              # 90% of HBM for weights+KV
    swap_space=4,                            # GB CPU swap for spillover
)

sampling = SamplingParams(temperature=0.7, max_tokens=512)

# Continuous batching is automatic — just send many requests
prompts = [
    "Explain prefill vs decode in 3 sentences.",
    "Write a Python function to compute MFU.",
    # ... 100 more
]
outputs = llm.generate(prompts, sampling)
for o in outputs:
    print(o.outputs[0].text)
```

### 8.2 vLLM OpenAI-Compatible Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --quantization awq \
    --tensor-parallel-size 2 \
    --enable-prefix-caching \
    --max-model-len 8192 \
    --speculative-model meta-llama/Meta-Llama-3-8B-Instruct \
    --num-speculative-tokens 5
```

Now your existing OpenAI SDK code points at `http://localhost:8000/v1` and works unchanged.

### 8.3 SGLang Example: Heavy Prefix Sharing

```python
import sglang as sgl

sgl.set_default_backend(
    sgl.RuntimeEndpoint("http://localhost:30000")
)

@sgl.function
def multi_turn_qa(s, system_prompt, history, question):
    s += sgl.system(system_prompt)              # cached across all calls
    for turn in history:                         # cached across same-history calls
        s += sgl.user(turn["q"]) + sgl.assistant(turn["a"])
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=300))

# Launch the server with RadixAttention prefix cache enabled (default)
# python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-70B-Instruct \
#     --tp 2 --enable-flashinfer --mem-fraction-static 0.85
```

---

## 9. Decision Tables

### 9.1 Engine Choice (Quick)

| Workload | Choose | Reason |
|----------|--------|--------|
| General chat / code-completion / RAG | vLLM | Solid default; great throughput |
| Agents with deep prefix sharing | SGLang | RadixAttention dominates here |
| Structured / JSON output enforcement | SGLang or TensorRT-LLM (with grammar) | Native constrained decoding |
| Single-user lowest-latency on H100 | TensorRT-LLM | Hand-tuned kernels, FP8 |
| Multi-model endpoint | Triton (with vLLM/TRT-LLM backend) | Built for this |
| CPU / Mac / on-device | llama.cpp | Only realistic option |
| Long-context (>32k) | vLLM or SGLang with paged + chunked-prefill | Both handle KV growth well |

### 9.2 Quantization Decision

| Goal | Use | Notes |
|------|-----|-------|
| Easy 2× memory | INT8 weight-only | Almost always free quality-wise |
| 70B on 1×H100 | INT4 AWQ or GPTQ | ~1–3% quality hit; calibrate properly |
| Frontier H100/B200 perf | FP8 (E4M3 weights + activations) | Native tensor cores; calibrate scale factors |
| Long-context KV pressure | KV-cache INT8 or FP8 | Stack with weight quant |
| QLoRA training | BnB NF4 | Training only — switch to AWQ/FP8 for serving |

### 9.3 Speculative Decoding On/Off

| Condition | Spec on? |
|-----------|----------|
| Batch size ≥ 32, MBU > 80% | **No** — no free FLOPs |
| Batch size 1–4, decode-dominant | **Yes** — biggest win |
| Code/structured output workload | **Yes** — high acceptance |
| Creative writing / high temperature | Maybe — acceptance often <40% |
| You don't have a good draft model | Try self-speculation (Medusa/EAGLE) |

---

## 10. Performance-Tuning Playbook

When TTFT or TPOT breaches an SLO, walk this list **in order**:

1. **Measure first**. p50/p95/p99 for TTFT, TPOT, e2e. Plot against input length and output length. Don't optimize until you see the distribution.
2. **Verify the engine has continuous batching + PagedAttention + FlashAttention**. If not, switch engine. (1-day fix, 5–20× win.)
3. **Enable prefix caching** if your traffic has overlapping prefixes (system prompts, multi-turn, RAG). Measure cache hit rate. (Often 30–70% cost reduction.)
4. **Quantize**. INT8 weight-only first. Then INT4 AWQ if memory-bound. FP8 on H100 if you have it. Validate quality on your eval set.
5. **Right-size the GPU**. KV cache + weights must fit in HBM at your target batch size. Compute: `weights_bytes + 2 × n_layers × n_kv_heads × head_dim × seq_len × batch × kv_dtype_bytes`.
6. **Tune `max_num_seqs` / `max_num_batched_tokens`** in vLLM. Increase until TPOT degrades or HBM fills.
7. **Add speculative decoding** if batch is small or workload is structured. Measure acceptance rate; tune K.
8. **Disaggregate prefill and decode** if TPOT spikes are correlated with long-prompt requests, AND you have multiple GPU nodes.
9. **Tensor-parallelize** if a single GPU is too small. NVLink-connected only.
10. **Replicate** for capacity. Replica parallelism is the easiest scale-out — multiple full copies behind a load balancer.

---

## 11. Real Numbers — What Healthy Looks Like

Rough order-of-magnitude on **H100 80GB SXM** (3 TB/s HBM, 989 TF32 TFLOP/s, 1979 BF16 TFLOP/s) running **vLLM**:

| Model + Quant | Batch | TTFT (1k input) | TPOT | tok/s decode | MBU |
|---------------|-------|-----------------|------|--------------|-----|
| Llama-3-8B FP16 | 1 | ~50ms | ~10ms | ~100 | ~50% |
| Llama-3-8B FP16 | 64 | ~120ms | ~12ms | ~5000 | very high (compute-bound) |
| Llama-3-70B INT4 (AWQ) | 1 | ~250ms | ~12ms | ~80 | ~93% |
| Llama-3-70B INT4 (AWQ) | 32, 2×H100 TP | ~400ms | ~18ms | ~1500 | high |
| Llama-3-70B FP8, 2×H100 TP | 64 | ~350ms | ~15ms | ~3000 | high |

If your numbers are **substantially worse** (e.g., MBU < 30% at batch=1, TPOT > 50ms on a 7B), something is wrong: engine not using FlashAttention, weights spilling to CPU, KV-cache fragmentation, or wrong dtype.

---

## 12. Anti-Patterns

- **Optimizing for nvidia-smi GPU%**. It's "fraction of time the GPU is doing *something*", not "fraction of FLOPs used". Track MFU/MBU instead.
- **Reporting average latency.** Always p50/p95/p99. Means lie under heavy tails.
- **Static batching in production.** Switch to a continuous-batching engine.
- **Spec decoding with a saturated engine.** If MBU is already 90% at high batch, spec gives nothing.
- **KV cache offload to CPU on the hot path.** 30–60× slowdown. Prevent it; size HBM correctly.
- **Tensor parallelism over PCIe.** AllReduce on a 70B at every layer over 64 GB/s PCIe is a death sentence. Need NVLink.
- **"Quantization is free"**. It usually isn't free on math/code/long-tail tasks. Always re-eval.
- **Tuning a benchmark instead of your traffic**. p99 TTFT on a fixed-length 1k-input synthetic dataset is meaningless if your real traffic is bimodal short-chat + long-RAG.
- **Self-hosting before measuring the managed-API cost-equivalence.** Below ~10M tokens/day, managed APIs (with prompt caching) are usually cheaper than self-hosting an H100. Self-host once your bill is the kind of number that justifies an SRE on-call rotation.

---

## See Also

- [`../../ml-architectures/quantization/`](../../ml-architectures/quantization/) — INT8 / INT4 / FP8 / AWQ / GPTQ / BnB-NF4 in depth.
- [`../../ml-architectures/attention/`](../../ml-architectures/attention/) — FlashAttention 1/2/3, PagedAttention, GQA, MLA internals.
- [`../../ml-architectures/sampling-strategies/`](../../ml-architectures/sampling-strategies/) — temperature, top-p, beam search, structured/constrained generation.
- [`../../ml-architectures/transformer/`](../../ml-architectures/transformer/) — KV cache mechanics, prefill/decode at the math layer.
- [`../../ml-architectures/llm/`](../../ml-architectures/llm/) — model architectures, distillation patterns.
- [`../../ml-libraries/vllm/`](../../ml-libraries/vllm/) — vLLM-specific configuration, deployment patterns.
- [`../../ml-libraries/sglang/`](../../ml-libraries/sglang/) — SGLang-specific configuration, RadixAttention details.
- [`../../ml-libraries/triton-inference-server/`](../../ml-libraries/triton-inference-server/) — multi-model serving, ensemble, BLS.
- [`../online-experimentation/`](../online-experimentation/) — A/B test the new engine before promoting.
- [`../llm-evaluation/`](../llm-evaluation/) — re-eval after quantization to catch quality regressions.

---

## References

- vLLM documentation: https://docs.vllm.ai/en/latest/
- SGLang documentation: https://docs.sglang.ai/
- TensorRT-LLM documentation: https://nvidia.github.io/TensorRT-LLM/
- HuggingFace TGI: https://huggingface.co/docs/text-generation-inference/index
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM, 2023): https://arxiv.org/abs/2309.06180
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022): https://arxiv.org/abs/2205.14135
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023): https://arxiv.org/abs/2307.08691
- Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (2024): https://arxiv.org/abs/2407.08608
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2022): https://arxiv.org/abs/2211.17192
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (DeepMind, 2023): https://arxiv.org/abs/2302.01318
- Yang et al., "Inference with Reference: Lossless Acceleration of Large Language Models" (LLMA, 2023): https://arxiv.org/abs/2304.04487
- Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving" (2024): https://arxiv.org/abs/2401.09670
- Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" (2024): https://arxiv.org/abs/2311.18677
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI '22, continuous batching): https://www.usenix.org/conference/osdi22/presentation/yu
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (2024): https://arxiv.org/abs/2401.10774
- Medusa code: https://github.com/FasterDecoding/Medusa
- Williams et al., "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (2009): https://dl.acm.org/doi/10.1145/1498765.1498785
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023): https://arxiv.org/abs/2305.13245
- Chip Huyen, *AI Engineering*, Chapter 9 — Inference Optimization (O'Reilly, 2024).
