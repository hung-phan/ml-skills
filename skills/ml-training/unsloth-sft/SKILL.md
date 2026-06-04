---
name: unsloth-sft
description: Fine-tune LLMs with Unsloth for 2x speed and 60% less VRAM using LoRA/QLoRA. Covers Alpaca, ShareGPT, and chat template datasets on single GPU. Use when fine-tuning Llama/Qwen/Gemma/Mistral/Phi models, need fast iteration, or working with consumer GPUs (RTX 3090/4090, T4, L4, A100).
---

# Unsloth SFT (Supervised Fine-Tuning)

Single-GPU fine-tuning with 2x speed and 70% less VRAM via custom Triton kernels. No accuracy loss.

- **Repo**: https://github.com/unslothai/unsloth
- **Docs**: https://docs.unsloth.ai
- **Notebooks**: https://github.com/unslothai/notebooks
- **HuggingFace models**: https://huggingface.co/unsloth
- **License**: Apache 2.0

## Why This Exists

**Problem**: Standard HuggingFace fine-tuning with QLoRA is slow and VRAM-hungry — a single training step on a 7B model can exhaust a 24GB GPU with a small batch, making iteration slow and consumer-GPU fine-tuning impractical.

**Key insight**: Replacing attention and cross-entropy kernels with hand-written Triton kernels eliminates redundant memory reads/writes, giving 2x throughput and 60–70% less VRAM with no accuracy trade-off.

**Reach for this when**: Fine-tuning any Llama/Qwen/Gemma/Mistral/Phi model on a single GPU (T4, L4, RTX 3090/4090, A100). Switch to `ray-distributed-sft` when you need multi-GPU/multi-node, or `distributed-grpo` when you need RL alignment at scale.

## When to Use

| Scenario | Use Unsloth? |
|----------|-------------|
| Single-GPU fine-tuning (T4, L4, RTX 3090/4090, A100) | ✅ Yes |
| LoRA/QLoRA parameter-efficient training | ✅ Yes |
| Quick prototyping before scaling | ✅ Yes |
| Multi-GPU / multi-node distributed | ❌ Use Ray Train or OpenRLHF |
| FSDP sharding across GPUs | ❌ Triton kernels incompatible |

## Installation

```bash
pip install unsloth
# or bleeding edge:
pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

Docs: https://docs.unsloth.ai/get-started/installing-unsloth

## Supported Models (500+)

| Family | Versions | Pre-quantized slug example |
|--------|----------|---------------------------|
| Llama | 3, 3.1, 3.2, 3.3, 4 | `unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit` |
| Qwen | 3, 3.5, 3.6 | `unsloth/Qwen3-8B-unsloth-bnb-4bit` |
| Gemma | 3, 4 | `unsloth/gemma-3-4b-it-unsloth-bnb-4bit` |
| Mistral/Ministral | v0.3, Medium 3.5 | `unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit` |
| Phi | 4 | `unsloth/Phi-4-unsloth-bnb-4bit` |
| DeepSeek | V3, R1 (MoE) | `unsloth/DeepSeek-R1-0528-Qwen3-8B-bnb-4bit` |

Full list: https://huggingface.co/unsloth

## Core API Pattern

```python
from unsloth import FastLanguageModel
import torch

# 1. Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit",
    max_seq_length=2048,
    dtype=None,           # auto-detect (float16 on T4, bfloat16 on Ampere+)
    load_in_4bit=True,    # QLoRA
)

# 2. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,                # 0 is optimized by Unsloth
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% less VRAM
    random_state=3407,
)
```

## LoRA Parameter Guide

| Parameter | Range | Recommendation |
|-----------|-------|---------------|
| `r` (rank) | 8–128 | 16 for general, 64 for complex, 128 for math/code |
| `lora_alpha` | = r | Keep equal to r for stability |
| `lora_dropout` | 0 | Always 0 with Unsloth (optimized kernel) |
| `target_modules` | list | All attention + MLP projections |
| `use_gradient_checkpointing` | `"unsloth"` | Always — 30% VRAM saved, no speed loss |
| `use_rslora` | False | True for rank-stabilized (scales by √r) |

## Dataset Formats

### Alpaca (single-turn instruction)
```json
{"instruction": "Summarize this.", "input": "The article...", "output": "Summary..."}
```

### ShareGPT (multi-turn chat)
```json
{"conversations": [
  {"from": "human", "value": "Help me?"},
  {"from": "gpt", "value": "Sure, here's..."}
]}
```

### ChatML (OpenAI-style)
```json
{"messages": [
  {"role": "user", "content": "What is 1+1?"},
  {"role": "assistant", "content": "2"}
]}
```

**Size guidelines**: 500–2,000 high-quality examples is the sweet spot. Quality > quantity.

## Complete Example: Alpaca SFT

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit",
    max_seq_length=2048, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        texts.append(alpaca_prompt.format(instruction, inp, output) + tokenizer.eos_token)
    return {"text": texts}

dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=2048,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        warmup_steps=5, num_train_epochs=1, learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="linear",
        seed=3407, output_dir="outputs", logging_steps=1,
    ),
)
trainer.train()
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```

## Complete Example: Chat Format with Response Masking

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit",
    max_seq_length=2048, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth")

tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = standardize_sharegpt(dataset)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
             for c in convos]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=2048,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=1, learning_rate=2e-4, optim="adamw_8bit",
        output_dir="outputs", logging_steps=1, seed=3407,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
    ),
)

# Mask user turns — only train on assistant responses (+1% accuracy)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)
trainer.train()
```

## Export

```python
# LoRA adapter only (small)
model.save_pretrained("lora_model")

# Merged 16-bit (for vLLM serving)
model.save_pretrained_merged("model_merged", tokenizer, save_method="merged_16bit")

# GGUF for Ollama/llama.cpp
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method="q4_k_m")
# Options: q4_k_m, q5_k_m, q8_0, f16

# Push to HuggingFace Hub
model.push_to_hub_merged("your-username/model", tokenizer, token="hf_...")
```

## Inference

```python
FastLanguageModel.for_inference(model)  # 2x faster inference mode

messages = [{"role": "user", "content": "Explain quantum computing simply."}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True,
    add_generation_prompt=True, return_tensors="pt").to("cuda")

from transformers import TextStreamer
model.generate(input_ids=inputs, streamer=TextStreamer(tokenizer, skip_prompt=True),
               max_new_tokens=256, use_cache=True)
```

## VRAM Requirements

| Model Size | 4-bit QLoRA | 16-bit LoRA |
|-----------|-------------|-------------|
| 1-3B | ~3 GB | ~8 GB |
| 7-8B | ~6 GB | ~18 GB |
| 13B | ~10 GB | ~30 GB |
| 70B | ~36 GB | ~140 GB |

## Hyperparameter Recommendations

| Task | LR | Epochs | Eff. Batch | Rank |
|------|-----|--------|-----------|------|
| General SFT | 2e-4 | 1-3 | 8-16 | 16 |
| Chat/Instruction | 2e-4 | 1-2 | 8 | 16 |
| Code | 1e-4 | 2-3 | 4 | 64 |
| Math/Reasoning | 5e-5 | 3-5 | 4 | 128 |

## Notebooks

| Model | Colab |
|-------|-------|
| Llama 3.1 8B Alpaca | https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb |
| Llama 3.2 1B+3B Chat | https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb |
| Gemma 3 4B | https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb |
| Qwen3 14B Reasoning | https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb |
| Phi-4 Chat | https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb |

## Key Limitation

**Single GPU only.** Unsloth's Triton kernels don't support DDP/FSDP. For multi-GPU, see the `ray-distributed-sft` and `distributed-grpo` skills.

## References

- GitHub: https://github.com/unslothai/unsloth
- Official docs: https://docs.unsloth.ai
- HuggingFace models: https://huggingface.co/unsloth
- Notebooks: https://github.com/unslothai/notebooks
