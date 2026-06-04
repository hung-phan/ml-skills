---
name: attention
description: All attention mechanisms for LLMs — variants (MHA, MQA, GQA, MLA, sliding window, sparse, linear, cross), efficient implementations (FlashAttention, PagedAttention, RadixAttention, Ring Attention), KV-cache math, and serving patterns. Use when designing transformer architectures, optimizing inference memory, extending context length, choosing attention for a new model, or debugging KV-cache OOM.
---

# Attention Mechanisms for LLMs

## Why This Exists

Self-attention computes pairwise interactions between all tokens: O(N²) time and memory
in sequence length N. For a 128K context, the attention matrix alone would be 128K × 128K
× 2 bytes = 32 GB — impossible to materialize.

Different attention variants trade off:
- **Quality** — full bidirectional attention vs approximations
- **Memory** — KV-cache size during inference dominates GPU RAM
- **Speed** — FLOP count and memory bandwidth utilization
- **Context length** — what's the maximum feasible sequence

The implementation layer (FlashAttention, PagedAttention) is orthogonal to the variant —
you can combine GQA with FlashAttention + PagedAttention for maximum efficiency.

---

## Attention Variants (What's Being Computed)

### Multi-Head Attention (MHA)

**Problem it solves:** Allows the model to jointly attend to information from different
representation subspaces at different positions.

**Math:** Given input X ∈ R^{N×d_model}, for each head h:
```
Q_h = X @ W_Q_h    # (N, d_k)
K_h = X @ W_K_h    # (N, d_k)
V_h = X @ W_V_h    # (N, d_v)
Attn_h = softmax(Q_h @ K_h^T / √d_k) @ V_h
Output = Concat(Attn_1, ..., Attn_H) @ W_O
```

**KV-cache impact:** Each layer stores `2 × n_heads × d_k × seq_len` values.
For 32 heads × 128 dim × 8192 tokens × 2 (K+V) × 2 bytes = 128 MB per layer.
At 32 layers = 4 GB per sequence.

**Models:** GPT-2, GPT-3, BERT, original Transformer.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # Independent Q, K, V projections per head
        self.W_q = nn.Linear(d_model, d_model)  # n_heads * d_k
        self.W_k = nn.Linear(d_model, d_model)  # n_heads * d_k
        self.W_v = nn.Linear(d_model, d_model)  # n_heads * d_k
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        B, N, _ = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        if kv_cache is not None:
            K = torch.cat([kv_cache[0], K], dim=2)
            V = torch.cat([kv_cache[1], V], dim=2)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, -1)
        return self.W_o(out), (K, V)
```

---

### Multi-Query Attention (MQA)

**Problem it solves:** KV-cache is the bottleneck for large batch inference. MQA
reduces KV-cache by n_heads× by sharing a single K,V head across all query heads.

**Math:** All query heads project independently, but K and V use a single shared projection:
```
Q_h = X @ W_Q_h    # per head (N, d_k)
K   = X @ W_K      # shared  (N, d_k)  — ONE set
V   = X @ W_V      # shared  (N, d_v)  — ONE set
```

**KV-cache impact:** Reduced from `n_heads × d_k` to `1 × d_k` per layer.
32× reduction vs MHA. Enables much larger batch sizes.

**Models:** PaLM, PaLM-2, Falcon-40B, StarCoder.

**Trade-off:** Slight quality degradation (~0.1-0.3% on benchmarks) for massive
inference speedup. Quality loss increases with model scale.

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)      # n_heads * d_k
        self.W_k = nn.Linear(d_model, self.d_k)     # SINGLE head
        self.W_v = nn.Linear(d_model, self.d_k)     # SINGLE head
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        B, N, _ = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, 1, self.d_k).transpose(1, 2)  # (B,1,N,d_k)
        V = self.W_v(x).view(B, N, 1, self.d_k).transpose(1, 2)

        if kv_cache is not None:
            K = torch.cat([kv_cache[0], K], dim=2)
            V = torch.cat([kv_cache[1], V], dim=2)

        # K,V broadcast across all query heads
        K = K.expand(-1, self.n_heads, -1, -1)
        V = V.expand(-1, self.n_heads, -1, -1)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, -1)
        return self.W_o(out), (K[:, :1], V[:, :1])  # cache only 1 head
```

---

### Grouped-Query Attention (GQA)

**Problem it solves:** MQA is too aggressive — quality loss is noticeable at scale.
GQA interpolates between MHA and MQA: group query heads to share K,V within each group.

**Math:** With G groups (G=1 → MQA, G=n_heads → MHA):
```
Q_h = X @ W_Q_h           # per head
K_g = X @ W_K_g           # per group (n_heads/G heads share this)
V_g = X @ W_V_g           # per group
```

**KV-cache impact:** Reduced by factor of `n_heads / G`. Typical: 32 heads, 8 groups →
4× reduction vs MHA while retaining near-MHA quality.

**Models:** Llama 2 70B, Llama 3, Mistral 7B, Mixtral, Gemma, CodeLlama.

**Why it won:** Llama 2 paper showed GQA-8 matches MHA quality while approaching
MQA speed. Now the de facto standard for all large models.

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model=4096, n_heads=32, n_kv_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # heads per KV group
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)  # fewer
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)  # fewer
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        B, N, _ = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_kv_heads, self.d_k).transpose(1, 2)

        if kv_cache is not None:
            K = torch.cat([kv_cache[0], K], dim=2)
            V = torch.cat([kv_cache[1], V], dim=2)

        # Repeat K,V to match query head count
        K = K.repeat_interleave(self.n_rep, dim=1)  # (B, n_heads, S, d_k)
        V = V.repeat_interleave(self.n_rep, dim=1)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, -1)
        return self.W_o(out), (K[:, ::self.n_rep], V[:, ::self.n_rep])
```

---

### Multi-Latent Attention (MLA)

**Problem it solves:** Even GQA's KV-cache is too large for extremely long contexts
(1M+ tokens). MLA compresses KV into a low-rank latent representation, decoupling
cache size from head count entirely.

**Math:** Instead of caching K,V per head:
```
# Compression (during encoding):
c_KV = X @ W_DKV          # (N, d_c) where d_c << n_heads * d_k
# Decompression (during attention):
K_h = c_KV @ W_UK_h       # restore per-head K from compressed latent
V_h = c_KV @ W_UV_h       # restore per-head V from compressed latent
```
Cache only c_KV (small) instead of full K,V (large).

**KV-cache impact:** Cache stores d_c per token (e.g., 512) instead of
n_kv_heads × d_k (e.g., 8 × 128 = 1024). ~2× reduction over GQA, with
quality matching or exceeding MHA through the joint compression.

**Models:** DeepSeek-V2, DeepSeek-V3, DeepSeek-R1.

**Key insight:** The up-projection W_UK can be absorbed into Q projection during
inference, so decompression is free at decode time.

```python
class MultiLatentAttention(nn.Module):
    def __init__(self, d_model=4096, n_heads=32, d_compress=512, d_k=128):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_compress = d_compress

        self.W_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        # Down-project to compressed latent
        self.W_dkv = nn.Linear(d_model, d_compress, bias=False)
        # Up-project from latent to K,V per head
        self.W_uk = nn.Linear(d_compress, n_heads * d_k, bias=False)
        self.W_uv = nn.Linear(d_compress, n_heads * d_k, bias=False)
        self.W_o = nn.Linear(n_heads * d_k, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        B, N, _ = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # Compress KV into low-rank latent
        c_kv = self.W_dkv(x)  # (B, N, d_compress) — THIS is cached

        if kv_cache is not None:
            c_kv = torch.cat([kv_cache, c_kv], dim=1)

        # Decompress to full K, V
        K = self.W_uk(c_kv).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_uv(c_kv).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, -1)
        return self.W_o(out), c_kv  # cache compressed latent only
```

---

### Sliding Window Attention

**Problem it solves:** Full attention over long contexts is wasteful — most useful
information is local. Sliding window limits attention to the last W tokens, giving
O(N×W) complexity and bounded KV-cache.

**Math:**
```
Attn(i,j) = softmax(Q_i @ K_j / √d_k)  only if |i - j| <= W
```

**KV-cache impact:** Fixed at W tokens regardless of total sequence length.
With W=4096, cache never grows beyond 4096 entries per layer.

**Context reach:** Through stacking L layers with window W, information propagates
up to L×W tokens — a 32-layer model with W=4096 has effective reach of 131K tokens.

**Models:** Mistral 7B (W=4096), Gemma (W=8192), Longformer (combines with global tokens).

**Limitation:** Cannot directly attend to distant tokens in a single layer. Works
because deep networks propagate information layer by layer.

```python
class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model=4096, n_heads=32, window_size=4096):
        super().__init__()
        self.window_size = window_size
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.W_qkv(x).reshape(B, N, 3, self.n_heads, self.d_k)
        Q, K, V = qkv.permute(2, 0, 3, 1, 4)  # each (B, H, N, d_k)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Mask: only attend to tokens within window
        positions = torch.arange(N, device=x.device)
        mask = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs() > self.window_size
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, -1)
        return self.W_o(out)
```

---

### Sparse Attention

**Problem it solves:** Full O(N²) attention is prohibitive for very long sequences
(10K+ tokens). Sparse patterns attend to only O(N√N) or O(N log N) positions using
structured sparsity.

**Patterns:**
- **Local + Global (Longformer):** Each token attends to a local window + designated
  global tokens (e.g., [CLS]) attend everywhere
- **Block Sparse (BigBird):** Random + window + global blocks. Provably Turing complete
- **Strided (Sparse Transformer):** Attend every k-th token for long-range + local window

**KV-cache impact:** Only store KV for positions in the sparsity pattern.
Reduces memory proportionally to sparsity ratio.

**Models:** Longformer, BigBird, Sparse Transformer, GPT-3 (internal sparse layers).

**Trade-off:** Requires specialized kernels (Triton block-sparse). Quality matches
dense attention on most benchmarks but can miss rare long-range dependencies.

---

### Linear Attention

**Problem it solves:** Replace softmax(QK^T)V with a kernel trick that avoids
materializing the N×N matrix, achieving true O(N) time and memory.

**Math:** Replace softmax with kernel φ:
```
Standard: Attn = softmax(QK^T / √d) @ V       → O(N²d)
Linear:   Attn = φ(Q) @ (φ(K)^T @ V)          → O(Nd²)
```
By changing association order: compute φ(K)^T @ V first (d×d matrix), then
multiply by each φ(Q) row. Works when d << N.

**Variants:**
- **Performers:** Random feature maps approximating softmax kernel
- **RWKV:** Linear attention with exponential decay (recurrent formulation)
- **RetNet:** Multi-scale exponential decay with chunk-wise parallel training
- **Mamba/S4:** State space models — linear attention generalized with HiPPO theory

**KV-cache impact:** Constant — store a d×d running state instead of growing KV.
No per-token cache. Enables infinite context in theory.

**Models:** RWKV-4/5/6, RetNet, Mamba, Mamba-2.

**Trade-off:** Approximate attention. Quality gap vs softmax attention is
narrowing but still measurable on recall-intensive tasks. Best for streaming
and infinite-context use cases.

---

### Cross Attention

**Problem it solves:** Attend to a different sequence (encoder output, image embeddings,
retrieved context) rather than self-attending within one sequence.

**Math:**
```
Q = decoder_hidden @ W_Q     # from current sequence
K = encoder_output @ W_K     # from separate source
V = encoder_output @ W_V     # from separate source
Attn = softmax(QK^T / √d_k) @ V
```

**KV-cache impact:** Encoder KV is computed once during prefill and reused
across all decode steps. No growth during generation.

**Models:** T5, BART, Whisper, Flamingo (vision→language), Stable Diffusion
(text→image U-Net cross attention).

**Key property:** Encoder KV is static — never changes during autoregressive decode.
This makes cross-attention very cache-friendly.

---

## Efficient Implementations (How It's Computed Fast)

These are orthogonal to the attention variant — they optimize the computation itself.

### FlashAttention 1/2/3

**Problem:** Standard attention materializes the N×N score matrix, causing O(N²) HBM
reads/writes that bottleneck memory bandwidth.

**Solution:** Tile Q, K, V into SRAM blocks. Compute attention per tile using
**online softmax** (track running max and sum). Never materialize full N×N matrix.

| Version | Key Advance | Paper |
|---------|-------------|-------|
| Flash-1 | Tiled IO-aware attention, 2-4× speedup | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| Flash-2 | Better work partitioning, causal masking, ~2× over Flash-1 | [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) |
| Flash-3 | H100 warp specialization, FP8, 1.5-2× over Flash-2 | [arXiv:2407.08691](https://arxiv.org/abs/2407.08608) |

**Result:** Exact attention (no approximation) in O(N²) FLOPs but O(N) memory.
Enables 16× longer contexts on the same hardware.

**Usage:** `torch.nn.functional.scaled_dot_product_attention` with `attn_implementation="flash_attention_2"` in HuggingFace, or `flash_attn` package directly.

---

### PagedAttention

**Problem:** KV-cache allocation wastes memory due to fragmentation. Pre-allocating
max_seq_len per request leaves most memory unused. Batch size limited by worst-case
allocation.

**Solution:** Inspired by OS virtual memory — store KV in non-contiguous **pages**
(blocks of fixed token count, e.g., 16 tokens). Page table maps logical positions
to physical blocks. Allocate pages on demand.

**Impact:** Eliminates fragmentation. Enables near-zero waste and 2-4× higher
batch sizes. Enables memory sharing between beam search candidates.

**Paper:** [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

**Implementation:** vLLM (reference), TensorRT-LLM, SGLang all use this.

---

### RadixAttention

**Problem:** Many requests share common prefixes (system prompt, few-shot examples).
Each request re-computes and stores duplicate KV entries for the shared prefix.

**Solution:** Store KV-cache in a **radix tree** (prefix tree). When a new request
arrives, find the longest matching prefix and reuse its KV-cache directly.

**Impact:** For chat workloads with shared system prompts, saves 50-80% of KV-cache
compute and memory. Eliminates redundant prefill.

**Paper:** [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)

**Implementation:** SGLang's core scheduler feature.

---

### Ring Attention

**Problem:** Single-GPU memory limits maximum context length. Even with FlashAttention,
KV for 1M tokens doesn't fit on one device.

**Solution:** Distribute the sequence across GPUs in a ring topology. Each GPU holds
a chunk of Q and circulates K,V blocks around the ring. Overlaps communication with
FlashAttention computation per block.

**Impact:** Scales context length linearly with number of GPUs. 8 GPUs with 128K
each → 1M context. Communication is hidden behind compute.

**Paper:** [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)

---

### Cascade Attention

**Problem:** In multi-turn chat, the system prompt + conversation history (shared prefix)
is reprocessed for every response token. With long system prompts (10K+ tokens),
this dominates latency.

**Solution:** Hierarchical attention where the shared prefix's KV is computed once
and stored separately. New tokens attend first to prefix KV, then to the local
suffix KV. Merge results with proper softmax correction.

**Impact:** Eliminates redundant prefix computation in decode. Combines naturally
with RadixAttention for cross-request sharing.

---

## Serving Patterns

### Prefill vs Decode

| Phase | Characteristic | Bottleneck |
|-------|---------------|------------|
| **Prefill** | Process all input tokens at once | Compute-bound (matrix multiply) |
| **Decode** | Generate one token at a time | Memory-bandwidth-bound (KV read) |

Prefill is parallelizable (all tokens independent). Decode is sequential and
reads the entire KV-cache for each new token. This is why KV-cache size directly
impacts decode latency.

**Disaggregated serving:** Separate prefill and decode to different GPU pools
(DistServe, Splitwise). Prefill nodes use compute-optimized GPUs; decode nodes
use memory-bandwidth-optimized GPUs.

### POD-Attention (Prefill-Only Disaggregated)

Prefill-only nodes compute KV-cache and transfer it to decode nodes over NVLink/IB.
Eliminates the decode-node prefill cost entirely. Used in production at scale
(Anthropic, Google).

### Speculative Verify Attention

In speculative decoding, a draft model generates N candidate tokens. The target
model verifies all N tokens in one forward pass using a modified attention mask
that allows parallel verification. This is essentially a prefill-like operation
over the draft tokens — the verify attention reads existing KV + new draft KV
in one batched operation.

---

## KV-Cache Memory Formula

```
KV_cache_bytes = layers × kv_heads × head_dim × seq_len × 2 × dtype_bytes
                 ↑         ↑           ↑          ↑        ↑     ↑
              depth   KV head count  per-head   context  K+V  fp16=2, fp8=1
```

### Common Models

| Model | Layers | KV Heads | Head Dim | KV per Token (FP16) | 4K ctx | 128K ctx |
|-------|--------|----------|----------|---------------------|--------|----------|
| Llama 3 8B | 32 | 8 (GQA) | 128 | 128 KB | 512 MB | 16 GB |
| Llama 3 70B | 80 | 8 (GQA) | 128 | 320 KB | 1.25 GB | 40 GB |
| Mistral 7B | 32 | 8 (GQA) | 128 | 128 KB | 512 MB | 16 GB |
| GPT-3 175B | 96 | 96 (MHA) | 128 | 4.7 MB | 18.8 GB | 602 GB |
| DeepSeek-V3 | 61 | — (MLA) | d_c=512 | 62 KB | 248 MB | 7.9 GB |

**Per-token formula:**
```python
def kv_cache_per_token(layers, kv_heads, head_dim, dtype_bytes=2):
    """Returns bytes per token in KV-cache."""
    return layers * kv_heads * head_dim * 2 * dtype_bytes

# Llama 3 8B: 32 * 8 * 128 * 2 * 2 = 131,072 bytes = 128 KB/token
# At 4096 tokens: 128 KB * 4096 = 512 MB
```

**Batch memory:** Total KV = per_token_bytes × seq_len × batch_size.
A batch of 32 requests at 4K context on Llama 3 8B: 512 MB × 32 = 16 GB just for KV.

---

## Decision Table

| Scenario | Recommended Variant | Why |
|----------|-------------------|-----|
| New model training (general purpose) | **GQA** | Best quality/memory trade-off, proven |
| Maximum batch size for serving | **MQA** | Minimum KV-cache, slight quality cost acceptable |
| Ultra-long context (1M+) | **MLA + Ring Attention** | Compressed cache + distributed memory |
| Fixed-context deployment (e.g., 4K) | **GQA + FlashAttention-2** | Standard, well-optimized |
| Streaming/infinite context | **Linear (RWKV/Mamba)** | Constant memory, no growing cache |
| High-throughput serving | **GQA + PagedAttention** | Eliminate fragmentation, maximize batch |
| Shared prefix workloads (chat) | **GQA + RadixAttention** | Amortize system prompt across requests |
| Encoder-decoder tasks (translation) | **Cross Attention** | Attend to separate source sequence |
| Resource-constrained (edge, mobile) | **Sliding Window + GQA** | Bounded memory, still good quality |
| Research/maximum quality | **MHA** | No approximation, full expressiveness |

### Implementation Stack Combinations

| Inference Engine | Attention Variant Support | Implementation |
|-----------------|--------------------------|----------------|
| vLLM | GQA, MQA, MHA, MLA | FlashAttention-2 + PagedAttention |
| SGLang | GQA, MQA, MHA | FlashInfer + RadixAttention |
| TensorRT-LLM | GQA, MQA, MHA | Custom CUDA + PagedAttention |
| llama.cpp | GQA, MQA | Custom GGML kernels |

---

## Gotchas

1. **FlashAttention requires head_dim ≤ 256** — larger head dims need splitting or fallback
2. **MQA → GQA conversion is free** — duplicate the single KV head G times to create GQA checkpoint (no retraining needed for inference)
3. **PagedAttention page size matters** — too small = overhead, too large = waste. 16 tokens is standard
4. **Sliding window + RoPE interaction** — RoPE positions are absolute even with windowed attention; don't reset position IDs
5. **KV quantization (FP8/INT4) stacks with GQA** — 4× from GQA × 2× from FP8 = 8× reduction vs MHA FP16
6. **MLA requires fused kernels** — naive implementation is slower than GQA due to decompression overhead; needs custom Triton kernel to absorb W_UK into Q projection

---

## References

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017. Original transformer and MHA.
2. [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — Shazeer, 2019. MQA.
3. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023.
4. [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) — DeepSeek AI, 2024. MLA.
5. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao et al., 2022.
6. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao, 2023.
7. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Shah et al., 2024.
8. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., 2023. vLLM.
9. [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) — Zheng et al., 2023. RadixAttention.
10. [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) — Liu et al., 2023.
11. [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — Beltagy et al., 2020. Sparse attention.
12. [BigBird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — Zaheer et al., 2020.
13. [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) — Peng et al., 2023.
14. [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005) — Ye et al., 2025.
