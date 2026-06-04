---
name: quantization
description: Model quantization methods for shrinking LLM memory footprint and accelerating inference — PTQ, QAT, GPTQ, AWQ, bitsandbytes, FP8, and GGUF. Use when serving models on smaller GPUs, reducing inference cost, or choosing a quantization scheme for vLLM/llama.cpp deployment.
---

# Quantization

## Why This Exists

LLMs are too large for available GPU memory. A 70B parameter model at FP16 requires **140 GB** of VRAM -- more than any single consumer GPU and even most datacenter cards. Quantization trades numerical precision for memory savings and inference speed, enabling:

- Running 70B models on a single 48GB GPU (4-bit)
- 2-4x faster inference from reduced memory bandwidth
- Training with QLoRA on consumer hardware (24GB)
- Deploying multiple models per node in production

The core tradeoff: fewer bits per weight = less memory, but potential quality degradation. Modern methods minimize this loss through calibration and activation-aware strategies.

## Quantization Types

### Post-Training Quantization (PTQ)

Quantize a pre-trained model without additional training. Fast, requires only a small calibration dataset (128-512 samples).

```python
# General PTQ flow
# 1. Load full-precision model
# 2. Run calibration data through model to collect activation statistics
# 3. Determine optimal quantization parameters (scale, zero-point)
# 4. Convert weights to lower precision
# 5. Save quantized model
```

**Pros:** No training compute, minutes to quantize, works on any model  
**Cons:** Quality loss at aggressive quantization (2-3 bit), no recovery mechanism

### Quantization-Aware Training (QAT)

Simulate quantization during training/fine-tuning. The model learns to be robust to reduced precision through fake quantization nodes in the forward pass.

```python
# QAT inserts fake-quant ops during training:
# forward: x_q = fake_quantize(x)  # simulates precision loss
# backward: straight-through estimator (STE) passes gradients unchanged
```

**Pros:** Higher quality at same bit-width vs PTQ, recovers lost accuracy  
**Cons:** Requires training compute, more complex pipeline

## Methods

### GPTQ (3/4-bit, PTQ)

Layer-wise quantization using approximate second-order information (Hessian). Calibration-based -- processes weights one layer at a time to minimize output error.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B")
quantization_config = GPTQConfig(
    bits=4,
    dataset="c4",            # calibration dataset
    group_size=128,          # quantize in groups of 128 weights
    desc_act=True,           # order by activation magnitude (slower but better)
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    quantization_config=quantization_config,
    device_map="auto",
)
model.save_pretrained("Llama-3.1-70B-GPTQ-4bit")
```

**Characteristics:**
- 4-bit: ~0.5 perplexity increase over FP16
- 3-bit: ~1-2 perplexity increase
- Quantization takes 1-4 hours for 70B on A100
- Excellent inference speed with optimized kernels (exllama, marlin)

### AWQ (Activation-Aware Weight Quantization, 4-bit, PTQ)

Identifies and preserves "salient" weights (those corresponding to large activations) by scaling them up before quantization, then scaling activations down to compensate.

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-3.1-70B"
quant_path = "Llama-3.1-70B-AWQ"

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",      # GEMM for batch>1, GEMV for batch=1
}

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
```

**Characteristics:**
- Slightly better quality than GPTQ at 4-bit (preserves salient channels)
- Faster quantization (~30 min for 70B)
- Excellent vLLM support with fused kernels
- No 3-bit variant (4-bit only)

### bitsandbytes (QLoRA, 4-bit NF4)

Dynamic quantization using NormalFloat4 (NF4) data type optimized for normally-distributed weights. Primary use: memory-efficient fine-tuning with QLoRA.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 > FP4 for normal distributions
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16
    bnb_4bit_use_double_quant=True,      # quantize the quantization constants
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    quantization_config=bnb_config,
    device_map="auto",
)

# QLoRA: freeze quantized base, train LoRA adapters
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
# Trainable params: ~160M on a 70B model (0.2%)
```

**Characteristics:**
- NF4 is information-theoretically optimal for normal distributions
- Double quantization saves additional ~0.4 bits/param
- Dequantizes on-the-fly for compute (slower inference than GPTQ/AWQ)
- Primary value: enables fine-tuning, not optimized inference serving

### FP8 (Hopper/Ada Native)

Hardware-native 8-bit floating point on NVIDIA H100/H200/L4/RTX 4090. Two formats: E4M3 (range for weights) and E5M2 (range for gradients).

```python
# vLLM FP8 serving (requires H100/L4)
# vllm serve meta-llama/Llama-3.1-70B --quantization fp8

# Manual FP8 quantization with llm-compressor
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",    # per-tensor dynamic scaling
    ignore=["lm_head"],      # keep output layer in fp16
)

oneshot(
    model="meta-llama/Llama-3.1-70B",
    recipe=recipe,
    output_dir="Llama-3.1-70B-FP8",
    max_seq_length=4096,
    num_calibration_samples=512,
)
```

**Characteristics:**
- Near-lossless (<0.1 perplexity increase)
- 2x memory reduction vs FP16
- Hardware tensor core acceleration (no software dequant overhead)
- Requires Hopper (H100/H200) or Ada (L4/RTX 4090) architecture

### INT8 (LLM.int8())

Mixed-precision decomposition: most weights in INT8, outlier features (>6σ) in FP16. Two schemes:

```python
# Absmax quantization (symmetric)
# scale = max(|W|) / 127
# W_int8 = round(W / scale)

# Zero-point quantization (asymmetric)
# scale = (max(W) - min(W)) / 255
# zero_point = round(-min(W) / scale)
# W_uint8 = round(W / scale) + zero_point

# HuggingFace usage
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    quantization_config=config,
    device_map="auto",
)
# ~70GB -> ~70GB (INT8) + outlier overhead ≈ 72GB total
```

**Characteristics:**
- Negligible quality loss for most models
- ~1.5-2x slower than FP16 due to decomposition overhead
- Universal hardware support (any GPU with INT8 tensor cores)
- Superseded by FP8 on Hopper, still useful on Ampere (A100/A10)

### GGUF Quantization Levels

GGUF (GPT-Generated Unified Format) is the standard for CPU/hybrid inference via llama.cpp, Ollama, and LM Studio. Multiple quantization levels with K-quant (importance-weighted) variants:

```bash
# Convert and quantize with llama.cpp
python convert_hf_to_gguf.py meta-llama/Llama-3.1-70B --outfile llama-70b-f16.gguf
./llama-quantize llama-70b-f16.gguf llama-70b-Q4_K_M.gguf Q4_K_M
```

| Level | Bits/Weight | 7B Size | 70B Size | Quality |
|-------|-------------|---------|----------|---------|
| Q2_K | 2.6 | 2.7 GB | 26 GB | Poor -- noticeable degradation |
| Q3_K_M | 3.4 | 3.3 GB | 33 GB | Acceptable for simple tasks |
| Q4_K_M | 4.8 | 4.4 GB | 43 GB | **Sweet spot** -- minimal loss |
| Q5_K_M | 5.7 | 5.1 GB | 50 GB | Near-lossless |
| Q6_K | 6.6 | 5.9 GB | 58 GB | Negligible loss |
| Q8_0 | 8.5 | 7.2 GB | 70 GB | Lossless for practical purposes |

K-quant variants (K_S, K_M, K_L) allocate more bits to important layers (attention) and fewer to less sensitive ones (FFN).

## When to Use Each

| Method | Bits | Memory Saving | Speed vs FP16 | Quality Loss | Hardware | Best For |
|--------|------|---------------|---------------|--------------|----------|----------|
| FP8 | 8 | 2x | 1.5-2x faster | Negligible | H100/L4 | Production serving (Hopper) |
| INT8 | 8 | 2x | 0.7-1x | Negligible | Any GPU | Legacy/Ampere serving |
| AWQ | 4 | 4x | 2-3x faster | Very low | Any GPU | **Production serving (vLLM)** |
| GPTQ | 4/3 | 4-5x | 2-3x faster | Low (4-bit) | Any GPU | Production serving, 3-bit option |
| bitsandbytes | 4 | 4x | 0.5-0.7x | Low | Any GPU | **Fine-tuning (QLoRA)** |
| GGUF Q4_K_M | ~4.8 | 3.3x | CPU-dependent | Low | CPU/hybrid | Local/edge deployment |
| GGUF Q8_0 | 8.5 | 1.9x | CPU-dependent | Negligible | CPU/hybrid | Quality-sensitive local |

**Decision flow:**
1. **Fine-tuning?** → bitsandbytes NF4 + QLoRA
2. **GPU serving on Hopper?** → FP8 (best quality/speed)
3. **GPU serving on Ampere/older?** → AWQ 4-bit (vLLM) or GPTQ
4. **CPU/laptop/edge?** → GGUF Q4_K_M (balance) or Q5_K_M (quality)
5. **Maximum compression?** → GPTQ 3-bit or GGUF Q3_K_M

## Integration

### vLLM

```bash
# AWQ model
vllm serve TheBloke/Llama-2-70B-AWQ --quantization awq --dtype float16

# GPTQ model
vllm serve TheBloke/Llama-2-70B-GPTQ --quantization gptq --dtype float16

# FP8 dynamic (H100/L4)
vllm serve meta-llama/Llama-3.1-70B --quantization fp8

# FP8 static (pre-quantized)
vllm serve neuralmagic/Llama-3.1-70B-FP8 --quantization fp8
```

### Unsloth (QLoRA Training)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-70B",
    max_seq_length=4096,
    load_in_4bit=True,       # NF4 quantization
    dtype=None,              # auto-detect (bf16 on Ampere+)
)

model = FastLanguageModel.get_peft_model(
    model, r=64, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
```

### HuggingFace BitsAndBytesConfig

```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit for fine-tuning
config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 8-bit for inference
config_8bit = BitsAndBytesConfig(load_in_8bit=True)
```

### GGUF with Ollama / llama.cpp

```bash
# Ollama (simplest)
ollama run llama3.1:70b-instruct-q4_K_M

# llama.cpp server
./llama-server -m llama-70b-Q4_K_M.gguf \
  --n-gpu-layers 40 \    # offload 40 layers to GPU (hybrid)
  --ctx-size 8192 \
  --port 8080
```

## Memory Math

**Formula:** `Memory (GB) = Parameters × Bytes_per_param / 1e9`

| Model | FP32 | FP16/BF16 | INT8/FP8 | 4-bit |
|-------|------|-----------|----------|-------|
| 7B | 28 GB | 14 GB | 7 GB | 3.5 GB |
| 13B | 52 GB | 26 GB | 13 GB | 6.5 GB |
| 34B | 136 GB | 68 GB | 34 GB | 17 GB |
| 70B | 280 GB | 140 GB | 70 GB | 35 GB |
| 405B | 1620 GB | 810 GB | 405 GB | 203 GB |

**Additional overhead:**
- KV cache: `2 × layers × heads × head_dim × seq_len × batch × bytes_per_param`
- Activations during inference: ~10-20% of model weights
- CUDA context: ~0.5-1 GB fixed

**Example:** Llama 3.1 70B with AWQ 4-bit + 4096 context + batch=32:
- Weights: 35 GB
- KV cache: 2 × 80 × 8 × 128 × 4096 × 32 × 2 bytes ≈ 17 GB (FP16 KV)
- Total: ~54 GB → fits on 2× A10G (48 GB each with tensor parallelism)

## Gotchas

- **GPTQ desc_act=True** gives better quality but breaks exllama kernel compatibility. Use `desc_act=False` for maximum inference speed
- **AWQ GEMM vs GEMV:** GEMM kernel is faster for batch>1 (serving), GEMV for batch=1 (interactive)
- **bitsandbytes is slow for inference** -- it dequantizes on every forward pass. Use only for training; convert to GPTQ/AWQ for serving
- **FP8 requires calibration** for static quantization; dynamic FP8 is easier but ~5% slower
- **GGUF Q4_K_M ≠ GPTQ 4-bit:** K-quant uses mixed precision across layers, often better quality than uniform 4-bit
- **Quantized models can't be further fine-tuned** with full-parameter methods -- only LoRA/QLoRA on the frozen quantized base
- **Group size matters:** smaller group_size (64 vs 128) = better quality but more overhead. 128 is the standard tradeoff

## References

1. [GPTQ Paper -- Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
2. [AWQ Paper -- Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
3. [QLoRA Paper -- Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
4. [bitsandbytes GitHub](https://github.com/bitsandbytes-foundation/bitsandbytes)
5. [LLM.int8() Paper -- 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
6. [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
7. [vLLM Quantization Docs](https://docs.vllm.ai/en/stable/quantization/index.html)
8. [llm-compressor (FP8)](https://github.com/vllm-project/llm-compressor)
9. [AutoAWQ GitHub](https://github.com/casper-hansen/AutoAWQ)
10. [Unsloth 4-bit Loading](https://docs.unsloth.ai/get-started/all-our-models)
