---
name: huggingface
description: HuggingFace ecosystem — transformers (AutoModel/AutoTokenizer/Trainer), datasets, tokenizers, PEFT (LoRA/QLoRA), Accelerate, and Hub. Use when loading pretrained models, fine-tuning with Trainer, or working with HF datasets and tokenizers.
  - model hub
  - push_to_hub
  - fine-tune
  - pretrained model
---

# HuggingFace Ecosystem

## Why This Exists

**Problem**: Pretrained models are expensive to train, vary in format and API, and lack a standard way to download, fine-tune, share, and deploy — gluing together model weights, tokenizers, training loops, and parameter-efficient adapters from scratch requires hundreds of lines of error-prone boilerplate.

**Key insight**: A unified `Auto*` class system plus a central Hub means any of 200K+ public models can be loaded, fine-tuned with LoRA/QLoRA, and published back to the Hub with a single consistent API regardless of architecture.

**Reach for this when**: You need to load or fine-tune any pretrained transformer (LLM, vision, audio, multimodal); use PEFT/LoRA/QLoRA to adapt large models on limited VRAM; stream large datasets; or distribute training across GPUs without rewriting your training loop. Use raw PyTorch instead only when you need full control over the forward pass or custom architectures not supported by `Auto*` classes.

## 1 — Why HuggingFace Exists

Problem: Pretrained models are expensive to train, vary in format/API, and lack a standard way to load, fine-tune, share, and deploy. HuggingFace provides:

- Unified API for 200K+ models across NLP, vision, audio, multimodal
- Standard format (safetensors) and config-driven architecture resolution
- One-line loading of any public/private model with automatic weight download
- Built-in training loops, dataset streaming, and parameter-efficient fine-tuning
- A Hub for publishing models, datasets, and demo Spaces

---

## 2 — Transformers Library

Core abstraction: `Auto*` classes resolve architecture from config.

### Loading Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",  # shard across available GPUs
)
```

### Pipeline (High-Level Inference)

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("HuggingFace makes NLP easy!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

generator = pipeline("text-generation", model=model_name, tokenizer=tokenizer)
output = generator("The future of AI is", max_new_tokens=50)
```

### Trainer (Fine-Tuning)

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)
trainer.train()
trainer.push_to_hub("my-org/my-finetuned-model")
```

### Key Auto Classes

| Task | Class |
|------|-------|
| Causal LM | `AutoModelForCausalLM` |
| Seq2Seq | `AutoModelForSeq2SeqLM` |
| Classification | `AutoModelForSequenceClassification` |
| Token Classification | `AutoModelForTokenClassification` |
| QA | `AutoModelForQuestionAnswering` |
| Vision | `AutoModelForImageClassification` |
| Speech | `AutoModelForSpeechSeq2Seq` |

---

## 3 — Datasets Library

### Loading

```python
from datasets import load_dataset

# From Hub
ds = load_dataset("imdb")  # dict with 'train', 'test' splits
ds = load_dataset("json", data_files="data.jsonl")
ds = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

# Streaming (no download, iterate on-the-fly)
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
for example in ds:
    process(example)
    break
```

### Transforms

```python
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized = ds.map(tokenize, batched=True, num_proc=4, remove_columns=["text"])
tokenized.set_format("torch")  # returns torch tensors from __getitem__
```

### Filtering and Selecting

```python
filtered = ds.filter(lambda x: len(x["text"]) > 100)
subset = ds.select(range(1000))
shuffled = ds.shuffle(seed=42)
split = ds.train_test_split(test_size=0.1)
```

### Push to Hub

```python
ds.push_to_hub("my-org/my-dataset", private=True)
```

---

## 4 — Tokenizers

HuggingFace tokenizers are Rust-backed (fast) with Python bindings.

### Algorithms

| Algorithm | Used By | Description |
|-----------|---------|-------------|
| BPE | GPT-2, LLaMA, Falcon | Byte-Pair Encoding — iteratively merges most frequent pairs |
| WordPiece | BERT, DistilBERT | Similar to BPE but uses likelihood-based merging |
| Unigram | T5, ALBERT | Starts with large vocab, prunes by loss |
| SentencePiece | LLaMA, T5 | Language-agnostic, treats input as raw bytes |

### Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Encode
encoded = tokenizer("Hello world", return_tensors="pt")
# {'input_ids': tensor([[...]]), 'attention_mask': tensor([[...]])}

# Decode
text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)

# Batch with padding
batch = tokenizer(["short", "a longer sentence here"], padding=True, truncation=True, return_tensors="pt")

# Special tokens
tokenizer.pad_token = tokenizer.eos_token  # common for causal LMs
```

### Training a Custom Tokenizer

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=32000, special_tokens=["<pad>", "<eos>", "<bos>"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)
tokenizer.save("my-tokenizer.json")
```

---

## 5 — PEFT (Parameter-Efficient Fine-Tuning)

Train <1% of parameters while achieving 90-99% of full fine-tune quality.

### LoRA

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                   # rank — higher = more capacity, more params
    lora_alpha=32,          # scaling factor (effective lr = alpha/r)
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 13M || all params: 8B || trainable%: 0.16%
```

### QLoRA (4-bit Quantized + LoRA)

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Then apply LoRA on top
model = get_peft_model(model, config)
```

### Merging and Saving

```python
# Save adapter only (~50MB vs 16GB)
model.save_pretrained("./lora-adapter")

# Merge adapter into base for deployment
merged = model.merge_and_unload()
merged.save_pretrained("./merged-model")

# Load adapter later
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base, "./lora-adapter")
```

### When to Use What

| Method | VRAM | Quality | Speed | Use Case |
|--------|------|---------|-------|----------|
| Full fine-tune | 4x model size | Best | Slow | Unlimited budget |
| LoRA | ~1.2x model size | 95-99% | Fast | Most fine-tuning |
| QLoRA | ~0.5x model size | 90-97% | Medium | Large models on consumer GPUs |

---

## 5b — SFTTrainer (TRL — Preferred for LLM Fine-Tuning)

`Trainer` is general-purpose. For LLM instruction fine-tuning, use `SFTTrainer` from TRL — it handles chat templates, packing, and LoRA integration without boilerplate.

```python
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./sft-output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True,
        max_seq_length=2048,
        packing=True,  # pack short examples into one context window
    ),
    train_dataset=dataset,
    processing_class=tokenizer,  # replaces deprecated tokenizer= arg (TRL ≥0.13)
    peft_config=lora_config,
)
trainer.train()
trainer.save_model("./sft-lora-adapter")
```

**Dataset format** — pass a `messages` column and let SFTTrainer apply the chat template:

```python
# messages column: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
# SFTTrainer applies tokenizer.apply_chat_template() automatically
```

> **API note**: `SFTTrainer(tokenizer=...)` was deprecated in TRL 0.13. Use `processing_class=tokenizer`. The base `Trainer` made the same rename.

| | `Trainer` | `SFTTrainer` |
|--|-----------|-------------|
| Chat template handling | Manual | Built-in |
| Sequence packing | Manual | `packing=True` |
| LoRA integration | Manual `get_peft_model()` | `peft_config=` arg |
| Best for | Classification, general tasks | Causal LM instruction tuning |

---

## 6 — Accelerate (Distributed Training)

Zero-code-change distributed training launcher.

### Setup

```bash
accelerate config  # interactive wizard -> saves ~/.cache/huggingface/accelerate/default_config.yaml
accelerate launch train.py  # launches with configured parallelism
```

### In Code

```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

### Multi-GPU Launch

```bash
# 4 GPUs, single node
accelerate launch --num_processes=4 train.py

# Multi-node (2 nodes × 4 GPUs)
accelerate launch --num_processes=8 --num_machines=2 --machine_rank=0 --main_process_ip=10.0.0.1 train.py
```

### DeepSpeed Integration

```yaml
# accelerate config -> deepspeed
compute_environment: LOCAL_MACHINE
deepspeed_config:
  zero_stage: 3
  offload_optimizer_device: cpu
  offload_param_device: cpu
mixed_precision: bf16
```

### Accelerate + Trainer

Trainer uses Accelerate under the hood. For custom loops, use Accelerator directly.

---

## 7 — Hub (Model Cards, Spaces, Sharing)

### Push Model to Hub

```python
model.push_to_hub("my-org/my-model", private=True)
tokenizer.push_to_hub("my-org/my-model")
```

### Model Cards

Create a `README.md` in the model repo with YAML frontmatter:

```yaml
---
license: apache-2.0
language: en
tags:
  - text-generation
  - llama
datasets:
  - my-org/my-dataset
metrics:
  - perplexity
model-index:
  - name: my-model
    results:
      - task:
          type: text-generation
        metrics:
          - name: Perplexity
            type: perplexity
            value: 5.2
---
```

### Download Files Programmatically

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Single file
path = hf_hub_download("meta-llama/Llama-3.1-8B-Instruct", "config.json")

# Entire repo
snapshot_download("my-org/my-model", local_dir="./model-files")
```

### Spaces (Demo Apps)

```python
# app.py for Gradio Space
import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", model="my-org/my-model")

def generate(prompt):
    return pipe(prompt, max_new_tokens=100)[0]["generated_text"]

gr.Interface(fn=generate, inputs="text", outputs="text").launch()
```

Deploy with `huggingface-cli repo create --type space --space_sdk gradio`.

---

## 8 — Common Patterns

### Chat Template (Instruct Models)

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing briefly."},
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

### Gradient Checkpointing (Save VRAM)

```python
model.gradient_checkpointing_enable()
# or in TrainingArguments:
training_args = TrainingArguments(..., gradient_checkpointing=True)
```

### Flash Attention 2

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
```

---

## 9 — Gotchas

| Issue | Fix |
|-------|-----|
| OOM on large models | Use `device_map="auto"` + `torch_dtype=torch.bfloat16` or QLoRA |
| Tokenizer has no pad token | `tokenizer.pad_token = tokenizer.eos_token` |
| `push_to_hub` auth error | Run `huggingface-cli login` or set `HF_TOKEN` env var |
| Slow tokenization | Ensure you're using fast tokenizer (default); check `tokenizer.is_fast` |
| LoRA target modules wrong | Check `model.named_modules()` for actual linear layer names |
| Streaming dataset can't shuffle | Use `.shuffle(buffer_size=10000)` on IterableDataset |
| `Trainer` ignores columns | Dataset columns not in model's `forward()` signature are auto-removed |
| Model outputs gibberish after fine-tune | Check chat template; ensure `labels` are shifted correctly |

---

## 10 — Reference Links

- Transformers docs: https://huggingface.co/docs/transformers
- Datasets docs: https://huggingface.co/docs/datasets
- PEFT docs: https://huggingface.co/docs/peft
- Accelerate docs: https://huggingface.co/docs/accelerate
- Tokenizers docs: https://huggingface.co/docs/tokenizers
- Hub Python client: https://huggingface.co/docs/huggingface_hub
- GitHub: https://github.com/huggingface
- Model Hub: https://huggingface.co/models
- Spaces: https://huggingface.co/spaces

## References

- Official docs (Transformers): https://huggingface.co/docs/transformers
- Official docs (Datasets): https://huggingface.co/docs/datasets
- Official docs (PEFT): https://huggingface.co/docs/peft
- Official docs (Accelerate): https://huggingface.co/docs/accelerate
- GitHub: https://github.com/huggingface/transformers
