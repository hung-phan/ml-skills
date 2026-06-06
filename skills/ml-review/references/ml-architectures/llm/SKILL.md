---
name: llm
description: Large language model architectures (GPT, BERT, T5), modern optimizations (RoPE, GQA, SwiGLU, RMSNorm), tokenization, KV-cache, Flash Attention, LoRA/QLoRA fine-tuning, RLHF/DPO alignment, and inference optimization. Use when designing or fine-tuning an LLM, choosing between encoder/decoder/encoder-decoder, or planning serving with vLLM.
---

## Why This Exists

**Problem**: Natural language is infinitely compositional, ambiguous, and requires world knowledge — earlier NLP required task-specific architectures for each problem (NER, QA, summarization, translation) with expensive labeled datasets for each.

**Key insight**: Scale a simple autoregressive Transformer on massive text corpora and it emerges with general-purpose language understanding and generation. Fine-tuning or prompting this single pretrained model handles virtually any language task without task-specific architecture changes.

**Reach for this when**: You need text generation, understanding, summarization, translation, code completion, reasoning, or any language task. Use encoder-only (BERT) for classification/embeddings, decoder-only (GPT) for generation, encoder-decoder (T5) for seq2seq. The default starting point for any NLP task in 2024+.


# LLM Architectures & Techniques

## Core Architectures

### GPT (Decoder-Only, Causal)
Autoregressive: each token attends only to previous tokens via causal mask.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Causal mask: lower triangular
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
                             .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # Scaled dot-product with causal mask
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = self.dropout(F.softmax(att, dim=-1))
        return self.proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))
```

### BERT (Encoder-Only, Bidirectional)
Masked Language Modeling (MLM) + Next Sentence Prediction. Full bidirectional attention.

```python
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS] token for classification
```

### T5 (Encoder-Decoder)
Text-to-text framework. Encoder uses bidirectional attention; decoder uses causal + cross-attention.

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
inputs = tokenizer("translate English to French: Hello", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Modern LLM Optimizations (LLaMA/Mistral)

### RoPE (Rotary Position Embeddings)
Encodes relative position by rotating Q/K vectors. Extrapolates to longer sequences than training.

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary embeddings to query and key tensors."""
    cos = cos.unsqueeze(1)  # [seq_len, 1, dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

### GQA (Grouped-Query Attention)
Shares K/V heads across multiple Q heads. Reduces KV-cache memory by `n_heads / n_kv_heads` factor.

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # repetition factor
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        # Repeat KV heads to match Q heads
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)
        # ... standard attention computation
```

**Model configs:**
| Model | n_heads | n_kv_heads | Ratio |
|-------|---------|------------|-------|
| LLaMA-2 70B | 64 | 8 | 8:1 |
| Mistral 7B | 32 | 8 | 4:1 |
| LLaMA-3 8B | 32 | 8 | 4:1 |

### SwiGLU Activation
Replaces ReLU/GELU in FFN. `SwiGLU(x) = Swish(xW₁) ⊙ (xV)` — gated linear unit with Swish.

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

### RMSNorm
Cheaper than LayerNorm — removes mean subtraction. `RMSNorm(x) = x / RMS(x) * γ`

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight
```

---

## Tokenization

### BPE (Byte-Pair Encoding)
GPT-2/3/4, LLaMA. Iteratively merges most frequent byte pairs.

```python
from transformers import AutoTokenizer

# GPT-2 BPE
tok = AutoTokenizer.from_pretrained("gpt2")
tokens = tok.encode("Hello world!")  # [15496, 995, 0]
tok.decode(tokens)  # "Hello world!"

# LLaMA SentencePiece (BPE variant)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tok.vocab_size  # 32000
```

### Key Differences
| Tokenizer | Algorithm | Vocab Size | Used By |
|-----------|-----------|-----------|---------|
| GPT-2 BPE | Byte-level BPE | 50,257 | GPT-2/3 |
| tiktoken (cl100k) | BPE | 100,256 | GPT-4, Claude |
| SentencePiece | Unigram/BPE | 32,000 | LLaMA, T5 |
| Mistral tokenizer | BPE | 32,768 | Mistral |

---

## KV-Cache

Stores previously computed K/V tensors to avoid recomputation during autoregressive generation.

```python
class CachedAttention(nn.Module):
    def forward(self, x, kv_cache=None, use_cache=True):
        q, k, v = self.qkv_proj(x)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=-2)  # append new K
            v = torch.cat([past_v, v], dim=-2)  # append new V
        new_cache = (k, v) if use_cache else None
        # q attends to full k, v (past + current)
        attn = scaled_dot_product_attention(q, k, v)
        return attn, new_cache
```

**Memory formula:** `KV_cache_bytes = 2 × n_layers × n_kv_heads × head_dim × seq_len × batch × dtype_bytes`

Example: LLaMA-2 70B, seq=4096, batch=1, fp16:
`2 × 80 × 8 × 128 × 4096 × 1 × 2 = ~1.3 GB`

---

## Flash Attention

IO-aware exact attention — computes attention in SRAM tiles without materializing the full N×N matrix.

```python
# Using PyTorch 2.0+ native SDPA
from torch.nn.functional import scaled_dot_product_attention

output = scaled_dot_product_attention(q, k, v, is_causal=True)  # auto-selects FlashAttention

# Using flash-attn library directly
from flash_attn import flash_attn_func

# q, k, v: (batch, seqlen, nheads, headdim)
output = flash_attn_func(q, k, v, causal=True, softmax_scale=None)

# Variable-length sequences (packed batch)
from flash_attn import flash_attn_varlen_func

output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens_q,  # cumulative sequence lengths
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_len_q,
    max_seqlen_k=max_len_k,
    causal=True
)
```

**Key properties:**
- O(N) memory vs O(N²) for standard attention
- ~2-4x faster on A100/H100
- Exact (not approximate) — numerically equivalent to standard attention
- Sliding window variant for Mistral: `window_size=(window_left, window_right)`

---

## LoRA / QLoRA Fine-Tuning

### LoRA (Low-Rank Adaptation)
Freezes base model, trains low-rank decomposition matrices A (d×r) and B (r×d).

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # rank (8-64 typical)
    lora_alpha=32,           # scaling factor (usually 2×r)
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable: 0.1-0.5% of total params
```

### QLoRA (4-bit Quantized Base + LoRA)
Quantizes base model to 4-bit NF4, trains LoRA in fp16/bf16.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # Normal Float 4-bit
    bnb_4bit_use_double_quant=True,       # nested quantization
    bnb_4bit_compute_dtype=torch.bfloat16 # compute in bf16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
```

**LoRA rank selection:**
| Use Case | Rank | Trainable % |
|----------|------|-------------|
| Simple classification | 8 | ~0.05% |
| Instruction tuning | 16-32 | 0.1-0.3% |
| Domain adaptation | 64-128 | 0.5-1% |
| Near full fine-tune | 256+ | 2%+ |

---

## RLHF / DPO Alignment

### DPO (Direct Preference Optimization)
Eliminates reward model — optimizes policy directly from preference pairs.

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-sft-model")
ref_model = AutoModelForCausalLM.from_pretrained("your-sft-model")
tokenizer = AutoTokenizer.from_pretrained("your-sft-model")
tokenizer.pad_token = tokenizer.eos_token

# Dataset format: {"prompt": str, "chosen": str, "rejected": str}
training_args = DPOConfig(
    output_dir="dpo-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    beta=0.1,              # KL penalty strength (0.1-0.5)
    loss_type="sigmoid",   # or "hinge", "ipo"
    max_length=512,
    max_prompt_length=256,
    num_train_epochs=1,
    bf16=True,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

### RLHF Pipeline (PPO)
SFT → Reward Model → PPO optimization. More complex but more flexible.

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# 1. SFT model (already fine-tuned on instructions)
# 2. Reward model (trained on preference data)
# 3. PPO optimization
ppo_config = PPOConfig(
    batch_size=16,
    learning_rate=1e-5,
    ppo_epochs=4,
    mini_batch_size=4,
)
model = AutoModelForCausalLMWithValueHead.from_pretrained("sft-model")
# Training loop: generate -> score with reward model -> PPO update
```

### GRPO (Group Relative Policy Optimization)
No reward model needed. Samples multiple responses, uses relative ranking within group as reward signal.

---

## Inference Optimization

### Quantization
```python
# GPTQ (post-training, 4-bit, GPU-optimized)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-GPTQ", device_map="auto")

# AWQ (activation-aware, better quality than GPTQ at same bits)
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-AWQ", device_map="auto")

# GGUF (CPU-friendly, llama.cpp format)
# Use with llama-cpp-python or ollama
```

**Quantization comparison:**
| Method | Bits | Speed | Quality | Hardware |
|--------|------|-------|---------|----------|
| FP16 | 16 | 1x | Baseline | GPU |
| GPTQ | 4 | ~1.5x | -0.5% perplexity | GPU |
| AWQ | 4 | ~1.5x | -0.3% perplexity | GPU |
| GGUF Q4_K_M | 4.8 | CPU-fast | -1% perplexity | CPU/GPU |
| FP8 | 8 | ~1.3x | ~lossless | H100 |

### Speculative Decoding
Draft model proposes N tokens; target model verifies all N in one forward pass.

```python
# vLLM speculative decoding
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    speculative_model="meta-llama/Llama-2-7b-hf",  # draft
    num_speculative_tokens=5,  # propose 5 tokens per step
    tensor_parallel_size=4
)
# Achieves 2-3x speedup with identical output quality
```

**How it works:**
1. Small draft model generates N candidate tokens autoregressively
2. Large target model scores all N candidates in single forward pass
3. Accept prefix of correct tokens, reject from first mismatch
4. Expected speedup: ~α×N where α = acceptance rate (typically 70-90%)

### vLLM Serving
PagedAttention-based serving with continuous batching.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=2,       # multi-GPU
    gpu_memory_utilization=0.9,   # KV-cache memory fraction
    max_model_len=8192,
    enforce_eager=False,          # use CUDA graphs
    quantization="awq",           # optional quantization
)

params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
outputs = llm.generate(["Explain quantum computing"], params)
```

**vLLM key features:**
- **PagedAttention**: KV-cache stored in non-contiguous pages (like OS virtual memory). Eliminates fragmentation, enables ~3x more concurrent sequences.
- **Continuous batching**: New requests join mid-batch without waiting for others to finish.
- **Prefix caching**: Shares KV-cache for common prompt prefixes.
- **Chunked prefill**: Splits long prefills across iterations to maintain low latency.

### OpenAI-compatible server
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000
```

---

## Model Selection Guide

| Task | Recommended | Why |
|------|-------------|-----|
| Text generation | LLaMA-3/Qwen-2.5 | Best open-weight quality |
| Classification/NER | BERT/DeBERTa | Bidirectional context |
| Summarization | T5/BART | Encoder-decoder for seq2seq |
| Code generation | CodeLlama/DeepSeek-Coder | Code-specialized training |
| Embedding/retrieval | GTE/E5/BGE | Contrastive-trained encoders |
| Local/edge | Phi-3/Gemma-2 | Small but capable (1-4B) |
| Maximum quality | LLaMA-3.1 405B/Qwen-2.5 72B | Frontier open models |

### Fine-tuning decision tree
```
Need to adapt a model?
├── < 1000 examples → Few-shot prompting (no training)
├── 1K-10K examples → LoRA (rank 16-32)
├── 10K-100K examples → QLoRA (rank 64) or full fine-tune small model
├── Custom domain vocabulary → Continue pretraining then LoRA
└── Alignment/safety → DPO on preference pairs (simpler) or RLHF/PPO (more control)
```

### Hardware requirements (inference)
| Model Size | FP16 VRAM | 4-bit VRAM | Min GPU |
|-----------|-----------|-----------|---------|
| 7B | 14 GB | 4 GB | RTX 3090 / L4 |
| 13B | 26 GB | 8 GB | A10G / RTX 4090 |
| 34B | 68 GB | 20 GB | A100 40GB |
| 70B | 140 GB | 40 GB | 2×A100 80GB |
| 405B | 810 GB | 230 GB | 8×A100/H100 |

---

## Training Tips

- **Learning rate**: 1e-5 to 5e-5 for fine-tuning, 1e-4 to 3e-4 for pretraining
- **Gradient checkpointing**: Trades ~30% slowdown for ~60% memory reduction
- **bf16 over fp16**: More stable training, same memory, requires Ampere+ GPU
- **Warmup**: 3-10% of total steps for fine-tuning
- **Cosine schedule**: Standard for LLM training with min_lr = 0.1 × max_lr
- **Pack sequences**: Concatenate short examples to fill max_seq_len, separated by EOS
- **NEFTune**: Add noise to embeddings during training for 2-5% quality improvement

---

## References

- [Llama 2 (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288) — Open foundation model training and RLHF
- [GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165) — Language models as few-shot learners
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — Model hub and inference library
