---
name: unsloth-advanced
description: Advanced Unsloth techniques beyond basic SFT — GRPO reinforcement learning, DPO/ORPO alignment, vision model fine-tuning, and continued pretraining for domain adaptation. Use when building reasoning models, aligning with preferences, fine-tuning vision-language models, or teaching new knowledge/languages.
---

# Unsloth Advanced Techniques

Beyond basic SFT: alignment, reinforcement learning, vision, and pretraining.

- **GRPO docs**: https://docs.unsloth.ai/basics/reasoning-grpo-and-rl
- **DPO notebook**: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-DPO.ipynb
- **Vision docs**: https://docs.unsloth.ai/basics/vision-fine-tuning
- **CPT docs**: https://docs.unsloth.ai/basics/continued-pretraining

## Why This Exists

**Problem**: Basic SFT teaches format and behavior but cannot teach a model to reason more deeply, align with human preferences, understand images, or absorb new domain knowledge — each of those goals requires a different training objective and data contract.

**Key insight**: The same Unsloth speed/VRAM advantages that apply to SFT extend to GRPO (no reward model needed, just Python reward functions), DPO/ORPO (preference alignment without PPO overhead), vision fine-tuning, and continued pretraining — so you can run the full alignment pipeline on a single consumer GPU.

**Reach for this when**: You've done basic SFT with `unsloth-sft` and now need to improve reasoning (GRPO), align to preferences (DPO/ORPO), add image understanding (Vision), or adapt to a new domain/language (CPT). For multi-GPU GRPO at scale, graduate to `distributed-grpo`.

---

## 1. GRPO (Group Relative Policy Optimization)

Based on DeepSeek's approach. **No value model, no reward model** — uses custom Python reward functions + group statistics. 90% less VRAM than PPO.

### Code

```python
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    max_seq_length=2048, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=64, lora_alpha=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth")

# Custom reward functions — no trained reward model needed
def correctness_reward(prompts, completions, answer, **kwargs):
    """Reward exact answer match."""
    responses = [c[0]["content"] for c in completions]
    return [3.0 if extract_answer(r) == a else -3.0
            for r, a in zip(responses, answer)]

def format_reward(prompts, completions, **kwargs):
    """Reward proper XML formatting."""
    responses = [c[0]["content"] for c in completions]
    return [1.0 if "<answer>" in r and "</answer>" in r else -1.0 for r in responses]

trainer = GRPOTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=dataset,
    reward_funcs=[correctness_reward, format_reward],
    args=GRPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=8,        # Group size (G samples per prompt)
        max_completion_length=256,
        num_train_epochs=1,
        learning_rate=5e-6,
        output_dir="grpo_output",
    ),
)
trainer.train()
```

### GRPO Tips
- Minimum **300 steps** before reward increase visible
- Need ≥500 data rows
- Model should be ≥1.5B for reasoning tokens to emerge
- `num_generations=8` is the group size — higher = more stable but slower
- Supports variants: `loss_type='gspo'` (GSPO), `loss_type='dr_grpo'` (DR-GRPO)
- Use `fast_inference=True` + vLLM for faster generation during training

### GRPO Notebook
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(8B)-GRPO.ipynb

---

## 2. DPO (Direct Preference Optimization)

Uses paired preference data (chosen/rejected). No reward model.

```python
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()  # Must call before importing DPOTrainer
from trl import DPOTrainer, DPOConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/zephyr-sft-bnb-4bit", max_seq_length=2048, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=64, lora_alpha=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth")

# Dataset format: {"prompt": "...", "chosen": "...", "rejected": "..."}
dpo_trainer = DPOTrainer(
    model=model, ref_model=None,  # None = implicit reference (saves VRAM)
    args=DPOConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1, num_train_epochs=3,
        optim="adamw_8bit", output_dir="dpo_output",
    ),
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_length=1024, max_prompt_length=512,
)
dpo_trainer.train()
```

### DPO Data Format
```json
{
  "prompt": "Explain gravity to a 5-year-old",
  "chosen": "Gravity is like an invisible hand pulling everything down...",
  "rejected": "Gravity is described by Einstein's field equations..."
}
```

---

## 3. ORPO (Odds Ratio Preference Optimization)

Single-stage alignment — combines SFT and preference in one pass. No separate SFT step, no reference model. Same data format as DPO.

Notebook: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-ORPO.ipynb

---

## 4. Continued Pretraining (CPT)

Teach new knowledge or language. Uses `UnslothTrainer` with dual learning rates.

```python
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3-8b-bnb-4bit", max_seq_length=2048, load_in_4bit=True,
)

# CRITICAL: include lm_head and embed_tokens for vocabulary learning
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",
                    "lm_head", "embed_tokens"])

trainer = UnslothTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    args=UnslothTrainingArguments(
        learning_rate=5e-5,
        embedding_learning_rate=5e-6,  # 2-10x SMALLER for embeddings
        output_dir="cpt_output",
        per_device_train_batch_size=2,
        num_train_epochs=1,
    ),
)
trainer.train()
```

### CPT Rules
- **Two learning rates**: main for attention/MLP, 2-10x smaller for embeddings
- Include `lm_head` + `embed_tokens` in target_modules
- Dataset = raw text (no instruction formatting)
- Use cases: domain adaptation (medical, legal, finance), new languages

CPT docs: https://docs.unsloth.ai/basics/continued-pretraining

---

## 5. Vision Model Fine-Tuning

1.5-2x faster, 70% less memory than Flash Attention 2.

### Supported: Qwen3-VL, Gemma 3/4, Llama 3.2 Vision, Ministral 3

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct", load_in_4bit=True,
)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_tokens"],
)
```

### Vision Dataset Format
```python
[
  {"role": "user", "content": [
    {"type": "text", "text": "Describe this image."},
    {"type": "image", "image": pil_image}
  ]},
  {"role": "assistant", "content": [
    {"type": "text", "text": "A cat on a windowsill..."}
  ]},
]
```

### Tips
- Use `UnslothVisionDataCollator` for image preprocessing
- Image resize: `resize="min"` or explicit `(w, h)`, recommended 300-1000px
- Can combine GRPO with VLMs (Gemma 3/4, Qwen3-VL)

Vision docs: https://docs.unsloth.ai/basics/vision-fine-tuning

---

## 6. Method Selection

| Goal | Method | Data Needed |
|------|--------|-------------|
| Teach new behavior/format | SFT | instruction/response pairs |
| Improve reasoning | GRPO | prompts + reward functions |
| Align with preferences | DPO | prompt/chosen/rejected triples |
| Single-stage alignment | ORPO | prompt/chosen/rejected triples |
| New knowledge/language | CPT | raw text corpus |
| Image understanding | Vision SFT | image+text conversations |
| Combined | SFT → then GRPO or DPO | both |

## Typical Pipeline

```
1. SFT (teach format/behavior)  →  GRPO (improve reasoning)
   or
1. SFT (teach format/behavior)  →  DPO (align with human preferences)
```

## References

- GitHub: https://github.com/unslothai/unsloth
- GRPO docs: https://docs.unsloth.ai/basics/reasoning-grpo-and-rl
- DPO blog post (TRL): https://huggingface.co/blog/dpo-trl
- ORPO paper: https://arxiv.org/abs/2309.07124
