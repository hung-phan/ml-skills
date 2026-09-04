---
name: unsloth-advanced
description: Advanced Unsloth techniques beyond basic SFT — GRPO reinforcement learning, DPO/ORPO alignment, vision model fine-tuning, continued pretraining for domain adaptation, and knowledge distillation (response-based, logit KD, on-policy/GKD, reasoning/CoT). Use when building reasoning models, aligning with preferences, fine-tuning vision-language models, teaching new knowledge/languages, or distilling a large teacher into a small student.
---

# Unsloth Advanced Techniques

Beyond basic SFT: alignment, reinforcement learning, vision, and pretraining.

- **GRPO docs**: https://docs.unsloth.ai/basics/reasoning-grpo-and-rl
- **DPO notebook**: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb
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
    model=model, processing_class=tokenizer,
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
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb

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
        beta=0.1,
        max_length=1024, max_prompt_length=512,
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
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

## 6. Knowledge Distillation

Transfer the capability of a large **teacher** into a small **student**. The four variants differ on two axes: *what signal you match* (the teacher's final text vs. its token-level probability distribution) and *where the training data comes from* (teacher-generated / off-policy vs. student-generated / on-policy).

**What Unsloth actually ships**: only the response-based path is first-class — `synthetic-data-kit` (Meta's tool, wrapped as `unsloth.dataprep.SyntheticDataKit`) to generate data, then ordinary SFT (see `unsloth-sft`). There is **no** `FastDistillationTrainer` or native logit-KD trainer — do not reach for one. Logit-level and on-policy KD run through TRL's `GKDTrainer` with an Unsloth-loaded student.

- **Synthetic data (Unsloth docs)**: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide
- **synthetic-data-kit**: https://github.com/meta-llama/synthetic-data-kit
- **TRL GKDTrainer**: https://huggingface.co/docs/trl/main/en/gkd_trainer

### (a) Response-based (sequence-level) distillation

**Why this variant**: The default and cheapest. You never touch logits — the teacher just *writes answers*, and you SFT the student on that text. Works with **any** teacher, including API-only models (GPT-4o, Claude Sonnet 4.5), because you only need generated strings. Reach for it first; it covers 80% of practical distillation and is the only path when teacher and student have different tokenizers.

Open-weights teacher (served locally via vLLM):

```python
# pip install unsloth vllm synthetic-data-kit
from unsloth.dataprep import SyntheticDataKit

generator = SyntheticDataKit.from_pretrained(
    model_name="unsloth/Llama-3.3-70B-Instruct",  # or Qwen2.5-72B-Instruct
    max_seq_length=2048,
)
generator.prepare_qa_generation(
    output_folder="data", temperature=0.7, top_p=0.95,
    max_generation_tokens=512,
)  # writes synthetic_data_kit_config.yaml + boots a vLLM server
```

```bash
synthetic-data-kit -c synthetic_data_kit_config.yaml ingest "docs/report.pdf"
synthetic-data-kit -c synthetic_data_kit_config.yaml create data/output/report.txt \
    --num-pairs 100 --type qa
synthetic-data-kit -c synthetic_data_kit_config.yaml curate \
    data/generated/report_qa_pairs.json --threshold 1.0   # Llama-as-a-judge filter
synthetic-data-kit -c synthetic_data_kit_config.yaml save-as \
    data/curated/report.json -f ft            # -f ft -> chat/messages JSON
```

Then feed the `ft` output straight into `SFTTrainer` (see `unsloth-sft`).

API teacher (GPT-4o / Claude Sonnet 4.5) — `synthetic-data-kit`'s `api-endpoint` provider is OpenAI-compatible, so point it at the vendor endpoint (GPT) or a **LiteLLM proxy** that presents one unified `/v1` for Anthropic and everything else:

```yaml
llm:
  provider: "api-endpoint"
api-endpoint:
  api_base: "http://localhost:4000/v1"   # LiteLLM proxy fronting the teacher
  api_key: "sk-..."
  model: "claude-sonnet-4-5"             # or "gpt-4o"
```

**Data format** (`save-as -f ft`):

```json
{"messages": [
  {"role": "user", "content": "<question grounded in the source doc>"},
  {"role": "assistant", "content": "<teacher's answer>"}
]}
```

**Pitfalls**
- Teacher mistakes are copied verbatim — you SFT on hallucinations unless you `curate` (LLM-judge threshold) or check answers against a source.
- Low diversity: one teacher at low temperature yields repetitive rows. Vary prompts/temperature and dedup.
- Licensing: some API vendors' terms restrict training competing models on their outputs — verify current terms before using GPT-4o/Claude output commercially.
- Using the same base for teacher and student (as the demo notebook does with Llama-3.2-3B) teaches format, not capability — pick a genuinely stronger teacher.

Notebook: https://github.com/unslothai/notebooks/blob/main/nb/Meta_Synthetic_Data_Llama3_2_(3B).ipynb

### (b) Logit-based KD (KL on token distributions)

**Why this variant**: Matching the teacher's full next-token distribution transfers far more signal per token than a single hard label — the "dark knowledge" in the soft probabilities. Wins when you have an **open-weights teacher whose logits you can read at train time** and teacher+student **share a tokenizer** (e.g. Qwen2.5-7B → Qwen2.5-1.5B). Requires more compute than (a) but converges on fewer tokens.

Unsloth has no dedicated logit-KD trainer. Two supported routes:

Hand-rolled Hinton loss (temperature-scaled KL + hard-label CE) inside a custom training loop:

```python
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)                                    # T^2 keeps gradient scale
    hard = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1), ignore_index=-100,
    )
    return alpha * soft + (1 - alpha) * hard        # alpha weights toward the teacher
```

TRL `GKDTrainer` in the offline (supervised) regime — `lmbda=0.0` reduces the loss to token-level distillation against the teacher; `beta=0.0` makes it forward KL, `beta=1.0` reverse KL (MiniLLM). Note it moved to `trl.experimental` in recent TRL (older versions: `from trl import GKDTrainer, GKDConfig`):

```python
from unsloth import FastLanguageModel
from trl.experimental.gkd import GKDConfig, GKDTrainer

student, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-1.5B-Instruct", max_seq_length=2048, load_in_4bit=True)
student = FastLanguageModel.get_peft_model(student, r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth")

trainer = GKDTrainer(
    model=student,
    teacher_model="Qwen/Qwen2.5-7B-Instruct",   # same tokenizer family REQUIRED
    args=GKDConfig(lmbda=0.0, beta=0.0, temperature=1.0,   # offline forward-KL
                   learning_rate=1e-5, num_train_epochs=2, output_dir="kd_out"),
    processing_class=tok, train_dataset=dataset,
)
trainer.train()
```

**Data format**: same `messages` list as SFT — GKD computes teacher logits on the fly from the same prompts/responses, so you do not pre-store logits.

**Pitfalls**
- Vocabulary mismatch is fatal: token-level KL needs identical tokenizers. Cross-family KD (Llama teacher → Qwen student) requires logit alignment/projection — don't attempt casually; use (a) instead.
- Memory: teacher logits span the full vocab (~150k for Qwen) at every position and every step. Keep the teacher in 4/8-bit and chunk, or the teacher forward pass dominates VRAM.
- Forward KL is mode-covering — small students hedge and blur. Reverse KL (`beta→1.0`, the MiniLLM insight) sharpens generation but can mode-collapse; tune per task.

### (c) On-policy distillation (student samples, teacher scores)

**Why this variant**: Off-policy KD (a/b) trains the student on sequences it would never generate, so errors compound at inference (the train-inference distribution mismatch GKD/MiniLLM were built to fix). On-policy KD has the **student generate rollouts** and the **teacher score them** (its logits become the target), so the student gets corrected exactly where it actually errs. It also composes cleanly with GRPO — same generate-then-score loop, teacher logits as an extra reward/KL term. Use it when off-policy SFT plateaus below the teacher and you can afford the cost.

Same `GKDTrainer`, now on-policy via `lmbda`:

```python
args = GKDConfig(
    lmbda=1.0,          # 1.0 = fully on-policy; 0<lmbda<1 mixes per batch
    beta=0.5,           # JSD interpolation; tune per task (0=fwd KL, 1=rev KL)
    temperature=0.9,    # student sampling temperature
    max_new_tokens=256, # student generates this each step -> the cost driver
    seq_kd=False,       # True = SFT on teacher-generated sequences instead
    learning_rate=1e-6, num_train_epochs=1, output_dir="onpolicy_kd",
)
```

**Data format**: `messages` list — but only the prompt/user turn drives training; the student generates the completion and the teacher scores it, so reference assistant turns matter less than in SFT.

**Off-policy vs. on-policy / cost**: Start off-policy (`lmbda=0.0`, cheap, stable) to get the student into the right neighborhood, then raise `lmbda` to close the last gap. On-policy is expensive: `student.generate()` runs every step *and* the teacher must be resident in memory for scoring (TRL GKD has no vLLM-served-teacher path). Budget roughly RL-training cost, not SFT cost.

**Pitfalls**
- Needs the teacher **available at train time** — impossible with an API-only teacher (you can't read its logits). API teachers force you back to (a).
- Very slow; keep the teacher small (or heavily quantized) and `max_new_tokens` modest.
- Gemma-2 teachers/students need `attn_implementation="kernels-community/flash-attn2"` or the logits go NaN (soft-capping) — per the TRL warning.
- Flipping `lmbda` to 1.0 immediately is unstable; warm up from 0.

### (d) Reasoning / chain-of-thought distillation

**Why this variant**: When the teacher is a *reasoning* model (DeepSeek-R1, QwQ), the valuable signal is the **full reasoning trace**, not just the final answer. DeepSeek showed that SFT'ing dense students on ~800k R1-curated samples beats running RL on the small model directly — distillation transfers reasoning patterns the small model can't discover on its own. This is response-based distillation (variant a) specialized for long CoT, optionally followed by GRPO to push past the teacher. Typical size ratios: 671B/70B teacher → 8B/14B/32B student.

```python
# 1. Generate/collect traces from a reasoning teacher (served via vLLM), or reuse
#    an existing R1-distill dataset. synthetic-data-kit supports CoT generation:
#    synthetic-data-kit create <src> --type cot
# 2. CURATE: keep only rows whose FINAL answer matches ground truth (rejection sampling).
# 3. SFT the student on the traces (unsloth-sft), high max_seq_length for long CoT:
from unsloth import FastLanguageModel
student, tok = FastLanguageModel.from_pretrained(
    "unsloth/DeepSeek-R1-Distill-Qwen-14B", max_seq_length=8192, load_in_4bit=True)
# ... standard SFTTrainer on the reasoning dataset ...
# 4. (optional) GRPO on held-out problems with a correctness reward (see section 1).
```

**Data format** — reasoning trace embedded in the assistant turn, using the teacher's own delimiter (R1 uses `<think>`):

```json
{"messages": [
  {"role": "user", "content": "If 3x + 7 = 22, find x."},
  {"role": "assistant", "content": "<think>\n3x = 22 - 7 = 15, so x = 5.\n</think>\n\nx = 5"}
]}
```

**Pitfalls**
- Skipping the correct-answer filter distills *wrong* reasoning — rejection sampling on the final answer is mandatory, not optional.
- Truncation: traces run 2k–8k tokens; too-small `max_seq_length` cuts off the answer and teaches the model to never conclude.
- Template drift: keep the `<think>` delimiter consistent between training and inference, or the reasoning behavior collapses.
- Eval on reasoning benches (AIME, MATH-500, GPQA) — gains often don't surface on MMLU-style MCQ, so chat/knowledge benchmarks will mislead you.
- Don't expect RL from scratch on a tiny model to match distillation; DeepSeek found distillation >> small-model RL for the same compute.

Model card (SFT-only distill, 800k samples): https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

### Distillation hyperparameters

| Variant | LR | Temperature | KD weight | Epochs | Notes |
|---------|-----|-------------|-----------|--------|-------|
| (a) Response-based SFT | 2e-4 (LoRA) | 0.7 (teacher gen) | n/a — pure SFT | 1–3 | any teacher, incl. API |
| (b) Logit KD (offline / GKD lmbda=0) | 1e-5–5e-5 | T=2 hand-rolled; ~1 GKD | alpha≈0.5; beta=0 (fwd KL) | 1–3 | teacher+student share vocab |
| (c) On-policy KD (GKD lmbda>0) | 1e-6–1e-5 | 0.9 (student sampling) | beta≈0.5 (task-tuned) | 1–2 | student gen each step; slow |
| (d) Reasoning/CoT SFT (+GRPO) | 5e-5–1e-4 | 0.6 (teacher gen) | n/a — pure SFT | 2–5 | filter for correct answers |

### Distillation References

- Hinton et al. 2015, *Distilling the Knowledge in a Neural Network* (KD + temperature): https://arxiv.org/abs/1503.02531
- Gu et al. 2023, *MiniLLM: On-Policy Distillation* (reverse-KL for LLMs): https://arxiv.org/abs/2306.08543
- Agarwal et al. 2023, *GKD: On-Policy Distillation from Self-Generated Mistakes*: https://arxiv.org/abs/2306.13649
- DeepSeek-AI 2025, *DeepSeek-R1* (reasoning distillation to dense models): https://arxiv.org/abs/2501.12948
- LiteLLM proxy (unified OpenAI-compatible endpoint for API teachers): https://docs.litellm.ai/docs/simple_proxy

---

## 7. Method Selection

| Goal | Method | Data Needed |
|------|--------|-------------|
| Teach new behavior/format | SFT | instruction/response pairs |
| Improve reasoning | GRPO | prompts + reward functions |
| Align with preferences | DPO | prompt/chosen/rejected triples |
| Single-stage alignment | ORPO | prompt/chosen/rejected triples |
| New knowledge/language | CPT | raw text corpus |
| Image understanding | Vision SFT | image+text conversations |
| Combined | SFT → then GRPO or DPO | both |
| Shrink a large model into a small one | Distillation (response-based SFT, or logit KD) | teacher-generated outputs (or teacher logits + shared tokenizer) |
| Transfer reasoning from an R1-class teacher | CoT/reasoning distillation (SFT ± GRPO) | curated teacher reasoning traces, filtered for correct final answers |

## Typical Pipeline

```
1. SFT (teach format/behavior)  →  GRPO (improve reasoning)
   or
1. SFT (teach format/behavior)  →  DPO (align with human preferences)
   or (distillation — bootstrap from a stronger teacher)
1. Teacher generates data (synthetic-data-kit, or R1-style CoT traces)
     →  SFT student on teacher outputs (response-based / reasoning distillation)
     →  (optional) logit or on-policy KD (TRL GKDTrainer) to close the gap
     →  (optional) GRPO on reasoning to push the student past the teacher
```

## References

- GitHub: https://github.com/unslothai/unsloth
- GRPO docs: https://docs.unsloth.ai/basics/reasoning-grpo-and-rl
- DPO blog post (TRL): https://huggingface.co/blog/dpo-trl
- ORPO paper: https://arxiv.org/abs/2403.07691
- Unsloth docs — Datasets Guide (synthetic data generation): https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide
- synthetic-data-kit (Meta): https://github.com/meta-llama/synthetic-data-kit
- TRL GKDTrainer (Generalized Knowledge Distillation): https://huggingface.co/docs/trl/main/en/gkd_trainer
- Unsloth notebook — Meta Synthetic Data (Llama 3.2 3B): https://github.com/unslothai/notebooks/blob/main/nb/Meta_Synthetic_Data_Llama3_2_(3B).ipynb
- Hinton et al. 2015 — Distilling the Knowledge in a Neural Network: https://arxiv.org/abs/1503.02531
- Gu et al. 2023 — MiniLLM (On-Policy Distillation): https://arxiv.org/abs/2306.08543
- Agarwal et al. 2023 — GKD (On-Policy Distillation from Self-Generated Mistakes): https://arxiv.org/abs/2306.13649
- DeepSeek-AI 2025 — DeepSeek-R1 (reasoning distillation to dense models): https://arxiv.org/abs/2501.12948
- DeepSeek-R1-Distill-Qwen-32B model card: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
- LiteLLM proxy (unified OpenAI-compatible endpoint): https://docs.litellm.ai/docs/simple_proxy
