---
name: transformer
description: Complete Transformer architecture reference — self-attention, multi-head attention, positional encodings (sinusoidal, learned, RoPE, ALiBi), FFN variants, layer normalization, KV-cache, and Flash Attention. Use when implementing a Transformer from scratch, debugging attention internals, or choosing positional encoding and normalization for a custom model.
---

## Why This Exists

**Problem**: RNNs process sequences serially (can't parallelize) and struggle with long-range dependencies despite LSTM/GRU gating. CNNs parallelize but have limited receptive fields requiring many layers to see the full sequence.

**Key insight**: Self-attention lets every token directly attend to every other token in O(1) layers — no serial bottleneck, no limited receptive field. The model learns which tokens are relevant to each other regardless of distance, and processes all positions in parallel during training.

**Reach for this when**: You're working with any sequence-to-sequence task (translation, summarization), need a general-purpose backbone for NLP or vision, or want to leverage pretrained models (BERT, GPT, ViT). The default architecture for most sequence modeling tasks since 2018. Trade-off: O(N²) attention cost for sequence length N.


# Transformer Architecture

Complete reference for the Transformer architecture family — from vanilla "Attention Is All You Need" through modern LLM variants.

---

## 1. Self-Attention (Scaled Dot-Product)

Core operation: each token attends to all other tokens via learned projections.

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, heads, seq_len, d_k)
    K: (batch, heads, seq_len, d_k)
    V: (batch, heads, seq_len, d_v)
    mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
    Returns: (batch, heads, seq_len, d_v)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)
```

**Why scale by √d_k?** Without scaling, dot products grow with dimension, pushing softmax into saturation (tiny gradients).

---

## 2. Multi-Head Attention

Split representation into `h` heads — each learns different attention patterns.

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, T, _ = query.shape
        Q = self.W_q(query).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)
```

---

## 3. Positional Encoding

### 3.1 Sinusoidal (Original, Fixed)

```python
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

### 3.2 Learned Positional Embeddings (GPT, BERT)

```python
class LearnedPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.embedding(positions)
```

### 3.3 RoPE (Rotary Position Embedding — LLaMA, Mistral)

Encodes position by rotating Q/K vectors in 2D subspaces. Relative position naturally emerges in the dot product.

```python
def precompute_freqs_cis(d_k: int, max_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
    t = torch.arange(max_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(x, freqs_cis):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_rotated = x_complex * freqs_cis[None, None, :x.size(2), :]
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)
```

### 3.4 ALiBi (Attention with Linear Biases — BLOOM)

No positional embedding at all. Add a linear bias to attention scores based on distance. Best length extrapolation.

---

## 4. Feed-Forward Network (FFN)

### Standard FFN

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))
```

### SwiGLU FFN (LLaMA, Mistral, PaLM)

```python
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

---

## 5. Layer Normalization

### RMSNorm (LLaMA, Mistral)

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

**Pre-norm** (modern standard): normalize BEFORE sublayer, add residual AFTER. Stable training without warmup.

---

## 6. Architecture Variants

| Type | Examples | Use |
|------|----------|-----|
| Encoder-Decoder | T5, BART | Seq2seq (translation, summarization) |
| Decoder-Only | GPT, LLaMA, Mistral | Text generation (dominant for LLMs) |
| Encoder-Only | BERT, RoBERTa | Classification, NER, retrieval |

---

## 7. KV-Cache for Inference

```python
class CachedAttention(nn.Module):
    def forward(self, x, kv_cache=None):
        Q = self.W_q(x).view(B, 1, self.num_heads, self.d_k).transpose(1, 2)
        K_new = self.W_k(x).view(B, 1, self.num_heads, self.d_k).transpose(1, 2)
        V_new = self.W_v(x).view(B, 1, self.num_heads, self.d_k).transpose(1, 2)

        if kv_cache is not None:
            K = torch.cat([kv_cache[0], K_new], dim=2)
            V = torch.cat([kv_cache[1], V_new], dim=2)
        else:
            K, V = K_new, V_new

        out = scaled_dot_product_attention(Q, K, V)
        return self.W_o(out.transpose(1, 2).contiguous().view(B, 1, -1)), (K, V)
```

**Memory cost:** `2 * num_layers * batch * seq_len * num_heads * d_k` (in fp16).

---

## 8. Flash Attention

```python
from torch.nn.functional import scaled_dot_product_attention as sdpa

# Automatically uses Flash Attention when available (CUDA, bf16/fp16)
out = sdpa(Q, K, V, attn_mask=mask, is_causal=True, dropout_p=0.0)
```

**Key properties:** O(N) memory vs O(N²), 2-4× faster, exact (not approximate).

---

## 9. Common Architectures Reference

| Model | Type | d_model | heads | layers | Context |
|-------|------|---------|-------|--------|---------|
| BERT-base | Enc | 768 | 12 | 12 | 512 |
| GPT-2 | Dec | 1600 | 25 | 48 | 1024 |
| LLaMA 2 7B | Dec | 4096 | 32 | 32 | 4096 |
| LLaMA 2 70B | Dec | 8192 | 64 | 80 | 4096 |
| Mistral 7B | Dec | 4096 | 32 | 32 | 32K |
| LLaMA 3 8B | Dec | 4096 | 32 | 32 | 128K |

---

## 10. Full Decoder-Only Transformer (GPT-style)

```python
class GPTBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x, mask=None):
        h = self.ln1(x)
        h = self.attn(h, h, h, mask)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 768, num_heads: int = 12,
                 num_layers: int = 12, d_ff: int = 3072, max_len: int = 2048):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying

    def forward(self, idx):
        B, T = idx.shape
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x)
```

---

## 11. Training Tips

- **Warmup + Cosine Decay**: Linear warmup 2000 steps, cosine to min_lr = 0.1 × max_lr
- **Gradient Accumulation**: effective_batch = micro_batch × accumulation_steps
- **bf16 over fp16**: Same range as fp32, no loss scaling needed (Ampere+)
- **Gradient clipping**: max_norm=1.0
- **Weight decay**: 0.1 (applied to all except biases and LayerNorm)
- **AdamW β₁=0.9, β₂=0.95**

---

## 12. Shape Annotation Summary

```
Input tokens:      (B, T)          integers
Token embedding:   (B, T, D)       after embedding lookup
After QKV proj:    (B, H, T, K)    reshaped per-head
Attention scores:  (B, H, T, T)    softmax over last dim
Attention output:  (B, H, T, K)    weighted sum of V
After concat+Wo:   (B, T, D)       heads concatenated
FFN intermediate:  (B, T, F)       expanded (F = 4D typically)
FFN output:        (B, T, D)       contracted back
Final logits:      (B, T, V)       over vocabulary
```

---

## 13. Common Gotchas

1. **Forgetting causal mask** → decoder attends to future tokens → information leak
2. **Wrong scaling** → using `1/d_model` instead of `1/√d_k` per head
3. **KV-cache shape mismatch** → forgetting to account for GQA head groups
4. **Pre-norm final layer** → need an extra LayerNorm before the output head
5. **Weight tying** → output projection shares weights with token embedding (saves params, improves quality)
6. **Attention sink** → first token accumulates attention; some models add a dedicated sink token
7. **RoPE theta** → LLaMA 3 uses θ=500000 (vs 10000 in LLaMA 2) for longer context extrapolation

## When to Use

| ✅ Use Transformer | ❌ Don't Use |
|---|---|
| Long-range dependencies in sequences | Very short sequences (<50 tokens, MLP suffices) |
| Large datasets and compute budget available | Tiny datasets (<1K samples) |
| Parallelized training needed (vs sequential RNN) | Edge devices with <1GB RAM (use RNN/Mamba) |
| Multi-modal fusion (text + image + audio) | Infinite-length streaming (use Mamba/RNN) |
| Transfer learning from pretrained models | When O(n²) attention is too expensive |

**Typical domains**: NLP (GPT, BERT), vision (ViT, DINO), code (Codex), multimodal (CLIP, GPT-4V), protein folding, music generation.

**Decision rule**: Default for sequences >100 tokens when compute allows. If >8K tokens and need linear scaling → Mamba. If model must be tiny → RNN.

---

## References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — Original transformer architecture
- [RoPE (Su et al., 2021)](https://arxiv.org/abs/2104.09864) — Rotary Position Embedding
