---
name: dataset-curation
description: Dataset engineering for foundation models — SFT, preference, CoT, tool-use, multi-turn data formats; data quality vs quantity vs coverage; synthesis (Self-Instruct, Evol-Instruct, distillation); filtering, deduplication, contamination control; annotation ops; data versioning. Use when curating instruction-tuning data, generating synthetic data, building pretraining or RAG corpora, debugging data-driven quality issues, or vetting open datasets.
---

# Dataset Engineering for Foundation Models

Curate, synthesize, filter, and version data for SFT, preference tuning, continued pretraining, and RAG indexing. The FM-data counterpart of `feature-engineering` (which targets classical tabular ML).

---

## Why This Exists

**Problem.** Once you fix a base model, dataset quality is the dominant lever on output quality. Data-centric AI flips the old benchmark setup: instead of fixing data and optimizing models, you fix the model and optimize the dataset (Andrew Ng's 2021 challenge, DataComp 2023). Three failure modes show up in practice:

1. **Format mismatch** — SFT data is in the wrong chat template, the model trains on `### Response:` literals, and inference prompts that don't include the exact same scaffolding produce garbage. A pristine dataset in the wrong format trains a worse model than a noisy dataset in the right format.
2. **Silent collapse from AI-generated data** — naive Self-Instruct loops over-represent probable events and forget rare ones. The Curse of Recursion (Shumailov et al. 2023) shows irreversible defects when training on recursively generated data.
3. **Coverage holes** — quality is measurable but coverage is not, and most teams over-index on quality and ship models that fail on long-tail user inputs.

**Key insight.** For the same model, the dataset choice now drives most of the performance gap. LIMA (Zhou et al. 2023) finetuned LLaMA-65B on **1,000** carefully curated prompts and matched or beat GPT-4 in 43% of head-to-head pairs. Yi authors found 10K curated instructions beat hundreds of thousands of noisy ones. Llama 3's gains over Llama 2 came primarily from "improvements in data quality and diversity" — not architecture.

**Reach for this when** you are:
- Curating SFT or DPO data and unsure how much you need or what "high quality" means operationally.
- Generating synthetic instruction data and worried about model collapse, license contamination, or eval leakage.
- Filtering noisy web text for continued pretraining (FineWeb-style pipelines).
- Building a RAG corpus where chunk quality dictates retrieval ceiling.
- Debugging "we trained but quality didn't move" — almost always a data problem.
- Evaluating an open dataset before adopting it.

**Not this skill** — for classical tabular feature engineering (one-hot, target encoding, TF-IDF, Box-Cox), see `../feature-engineering/`. For schema/distribution validation, see `../data-validation/`.

---

## 1. Data Format by Training Phase

The training objective dictates the data shape. Get this wrong and nothing else matters.

| Phase | Format | Quantity unit | Typical size | What it teaches |
|---|---|---|---|---|
| **Continued pretraining / self-supervised** | Raw token sequences (no labels) | Tokens | 1B – 10T tokens | Knowledge, distribution shift to new domain or language |
| **Supervised finetuning (SFT)** | `(instruction, response)` pairs (single- or multi-turn) | Examples | 50 – 1M examples | Task format, instruction following, style |
| **Preference tuning (DPO, ORPO, KTO, GRPO)** | `(instruction, chosen, rejected)` triples | Pairs | 1K – 1M pairs | Calibration, preferences, refusal behavior |
| **Reward model training** | `(instruction, response, score)` or pairwise | Pairs / scored examples | 10K – 1M | Score function for RLHF/PPO |
| **Tool use / function calling** | Multi-message format with tool calls + results | Episodes | 1K – 100K | When/how to call tools, parse results |
| **CoT / reasoning** | `(instruction, rationale + answer)` | Examples | 1K – 1M (expensive) | Step-by-step solving |
| **RAG indexing** | Chunked docs + metadata + (optional) Q→chunk mappings | Chunks | depends on corpus | Retrieval quality (not model weights) |

### Single-turn vs multi-turn

Single-turn `(user, assistant)` is cheaper to collect and easier to filter, but real tasks are multi-turn — clarification, correction, follow-up. UltraChat is the canonical multi-turn synthetic example.

### Tool-use needs a richer format

In a typical chat turn, the assistant emits **one** message. For tool use, the assistant emits **several** messages per turn (one to a code interpreter, one to the user). Llama 3 (Dubey et al. 2024) introduced a multi-message chat format with explicit source/destination headers and termination tokens. If you train on standard `(user, assistant)` pairs, you cannot teach the model to interleave tool calls and user-facing speech.

### Chat template alignment is non-negotiable

Each model expects a specific tokenizer and chat template. Train with the wrong template and the model bakes in `<|user|>` literals that won't match inference prompts. Use the tokenizer's `apply_chat_template` instead of hand-rolling the format:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is the boiling point of nitrogen?"},
    {"role": "assistant", "content": "Nitrogen boils at -195.79 C (-320.4 F) at 1 atm."},
]

# tokenize=False returns the formatted string so you can sanity-check it
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(text)
# Returns the full prompt with the model's specific role markers and EOS tokens.
```

For SFT training masks, you typically only train on assistant tokens. TRL's `SFTTrainer`, Unsloth, and Axolotl all support `train_on_responses_only` or equivalent — use it.

---

## 2. The Three Pillars (Huyen)

Quality, coverage, and quantity. Treat them as a Pareto frontier — pushing one too hard at the expense of the others is the most common failure.

### 2.1 Data Quality

A high-quality example is **relevant, aligned with task requirements, consistent, correctly formatted, sufficiently unique, and compliant**.

- **Relevant** — 19th-century legal text is irrelevant for a 2025 legal-Q&A bot, but perfect for "explain 19th-century US case law".
- **Aligned** — note the word: not "correct" but aligned. If the task is to generate creative copy, factually-perfect-but-boring is a low-quality response. If the task is concise answers, verbose responses are low quality even if accurate.
- **Consistent** — two annotators on the same example should give similar annotations. If essays-of-equal-quality get different scores, the model can't learn the score function.
- **Correctly formatted** — strip HTML, normalize whitespace, fix casing. Databricks reported 20% accuracy gain and 60% input-token reduction from removing extraneous Markdown/HTML.
- **Sufficiently unique** — duplicates skew distribution and contaminate test splits. Anthropic found that duplicating 0.1% of the data 100× degraded an 800M model to the level of a 400M model (Hernandez et al. 2022).
- **Compliant** — no PII, no copyrighted material you can't redistribute, no license violations from teacher-model outputs.

**LIMA / Yi finding to internalize.** A small, hand-curated SFT set (~1K examples) routinely beats a 100K noisy mixture. For SFT specifically, pick quality over quantity until you've maxed quality, then add diversity, then scale.

### 2.2 Data Coverage / Diversity

Coverage = your training distribution should match your inference distribution. Dimensions to check:
- **Domain** (math, code, English knowledge, multilingual)
- **Format** (JSON, prose, bullet lists, yes/no)
- **Length** (short responses vs long-form)
- **Language / locale**
- **Task type** (summarization, QA, classification, transformation)
- **User-input quirks** (typos, ALL CAPS, code-switching)

Llama 3's domain mix differs sharply across phases — anchor on this when designing your own:

| Domain | Pretraining | SFT | Preference |
|---|---|---|---|
| General knowledge (English) | 50% | 52.66% | 81.99% |
| Math and reasoning | 25% | 21.19% | 5.89% |
| Coding | 17% | 14.89% | 6.93% |
| Multilingual | 8% | 3.01% | 5.19% |
| Exam-like | — | 8.14% | — |
| Long context | — | 0.11% | — |

Source: Dubey et al. 2024 (Llama 3). Note how preference tuning collapses to mostly general knowledge — preference data should reflect *real user preferences*, not the breadth needed during pretraining.

Chung et al. (2022) showed performance on held-out tasks scales with the **number of finetuning tasks**: 9 → 282 yields large gains, 282 → 1,836 yields diminishing but positive returns. Diversify across tasks first, examples-per-task second.

### 2.3 Data Quantity

Approximate rules of thumb:

| Setup | Min viable | Sweet spot | Notes |
|---|---|---|---|
| Sanity-check feasibility | 50 examples | — | If 50–100 doesn't move the needle, more data rarely will |
| PEFT (LoRA, QLoRA) on strong base | a few hundred | 1K – 10K | LIMA-style |
| Full finetuning | 10K | 100K – 1M | Justify only if PEFT plateaus |
| Pretraining from scratch | — | 1T – 16T tokens | Llama 2 = 2T, Llama 3 = 16T |
| Preference tuning (DPO/ORPO) | 1K pairs | 10K – 100K pairs | UltraFeedback ~64K pairs |
| Reward model | 10K | 100K+ | Pairwise more sample-efficient than scored |
| Continued pretraining | 100M tokens | 1B+ tokens | Watch for catastrophic forgetting |

**Ossification warning** — a small base model trained on too much SFT data can have its pretrained weights frozen so hard that further adaptation fails (Hernandez et al. 2021). Smaller models suffer worse. If you have millions of examples and a small base, evaluate training-from-scratch or stepping up to a larger base.

**Performance-gain curves.** Train on 25%, 50%, 100% of your data and plot the metric. A steep slope means doubling will help. A plateau means you should fix quality or coverage instead of adding more data.

---

## 3. Decision Tables

### 3.1 Annotation method

| Method | Best when | Quality | Cost | Risk |
|---|---|---|---|---|
| **Human (in-house experts)** | Sensitive domain (medical, legal); novel task | Highest if guidelines are good | Highest ($1 – $50/example) | Annotator fatigue, drift, IAA gaps |
| **Human (crowdsourced)** | Volume on well-defined tasks | Medium; needs aggregation | Medium ($0.10 – $2) | Quality variance |
| **AI-assisted (human reviews AI)** | Need scale + quality | High when guidelines tight | Low–medium | Anchoring bias on AI output |
| **AI-only with verification** | Functionally checkable (code, math) | High if verifier is strong | Lowest | Collapse, lineage opacity |
| **AI-only no verification** | Pretraining augmentation only | Variable | Lowest | Compounding errors |
| **Synthetic from templates** | Structured outputs (JSON, SQL) | Deterministic | Negligible | Low diversity |

Llama 3 chose AI-assisted over pure human for nuanced safety annotations because human-only data was *more* prone to errors and inconsistency.

### 3.2 Filtering aggressiveness

| Stage | Aggressiveness | Why |
|---|---|---|
| **Pretraining** | High at the document level (FastText quality classifier, dedup, perplexity), low at the token level | Volume matters; you can throw away 90% of CommonCrawl and still have trillions of tokens |
| **Continued pretraining** | High; keep only domain-relevant, high-quality | Smaller corpus; every doc shifts the model |
| **SFT** | Very high; manually inspect samples; LIMA discipline | 1K great > 100K noisy |
| **Preference** | High on chosen/rejected disagreement margin | Tiny margin = noise |
| **RAG indexing** | High on chunk coherence + metadata; low on volume | Retrieval quality bounded by chunk quality |
| **Eval set** | Maximum; manually inspect every example | A bad eval invalidates everything downstream |

### 3.3 Data format by training phase (compact)

| Phase | Use this format | Library helpers |
|---|---|---|
| Continued pretraining | List of `{"text": "..."}` | HF `datasets`, Mosaic streaming |
| SFT (single-turn) | `{"messages": [{"role": "user"}, {"role": "assistant"}]}` (chat template applied later) | TRL `SFTTrainer`, Unsloth |
| SFT (multi-turn) | Same, more messages | Same |
| DPO / ORPO | `{"prompt", "chosen", "rejected"}` | TRL `DPOTrainer` |
| KTO | `{"prompt", "completion", "label": bool}` | TRL `KTOTrainer` |
| Reward model | `{"prompt", "chosen", "rejected"}` (pairwise) | TRL `RewardTrainer` |
| Tool use | Multi-message with `tool_calls` and `tool` role | OpenAI/Anthropic chat-tool spec |
| CoT | `{"messages": [..., {"role":"assistant","content":"<reasoning>...</reasoning><answer>...</answer>"}]}` | Custom |

---

## 4. Loading and Inspecting with HuggingFace `datasets`

```python
from datasets import load_dataset, Dataset
import pandas as pd

# Load a real SFT mixture
ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
print(ds)
# Dataset({features: ['id', 'messages', 'source'], num_rows: ~939k})

# Always look at examples FIRST. Brockman: "Manual inspection of data has
# probably the highest value-to-prestige ratio of any activity in ML."
for ex in ds.shuffle(seed=0).select(range(5)):
    print(ex["source"], "::", ex["messages"][0]["content"][:120])

# Distribution checks
df = ds.select_columns(["source"]).to_pandas()
print(df["source"].value_counts())  # source-mix sanity check

# Length distribution — flags outliers and helps set max_seq_len
lens = ds.map(
    lambda ex: {"n_chars": sum(len(m["content"]) for m in ex["messages"])},
    num_proc=8,
)
print(pd.Series(lens["n_chars"]).describe(percentiles=[0.5, 0.9, 0.99]))

# Filter: drop too-short, too-long, empty
clean = ds.filter(
    lambda ex: 16 <= sum(len(m["content"]) for m in ex["messages"]) <= 32_000,
    num_proc=8,
)

# Train/val/test split (stratified by source if you care about coverage)
split = clean.train_test_split(test_size=0.02, seed=42)
```

---

## 5. Deduplication

**Three duplication levels matter** — exact, near-duplicate, and semantic. Pick based on cost vs. recall:

```python
# 5a. Exact dedup on a hash of the full content
import hashlib
def content_hash(ex):
    text = "\n".join(m["content"] for m in ex["messages"])
    return {"hash": hashlib.sha256(text.encode()).hexdigest()}

ds_hashed = ds.map(content_hash, num_proc=8)
seen = set()
def keep(ex):
    if ex["hash"] in seen:
        return False
    seen.add(ex["hash"])
    return True
ds_exact_dedup = ds_hashed.filter(keep)
```

```python
# 5b. Near-duplicate dedup with MinHash + LSH (datasketch)
# Catches paraphrases, whitespace differences, small edits.
from datasketch import MinHash, MinHashLSH

def shingles(text, k=5):
    text = " ".join(text.lower().split())  # normalize whitespace
    return {text[i:i+k] for i in range(max(1, len(text) - k + 1))}

lsh = MinHashLSH(threshold=0.85, num_perm=128)
minhashes = {}

for i, ex in enumerate(ds_exact_dedup):
    text = "\n".join(m["content"] for m in ex["messages"])
    m = MinHash(num_perm=128)
    for sh in shingles(text):
        m.update(sh.encode())
    # Keep only if no near-duplicate already in index
    if not lsh.query(m):
        lsh.insert(str(i), m)
        minhashes[i] = m

dedup_indices = sorted(minhashes.keys())
ds_near_dedup = ds_exact_dedup.select(dedup_indices)
print(f"Kept {len(ds_near_dedup)} of {len(ds)} after MinHash dedup")
```

```python
# 5c. Semantic dedup — embed and cluster
# Use when paraphrase-level dedup is needed (Self-Instruct does this).
# Expensive: O(N) embedding + O(N log N) ANN.
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
texts = ["\n".join(m["content"] for m in ex["messages"]) for ex in ds_near_dedup]
embs = model.encode(texts, normalize_embeddings=True, batch_size=64,
                    show_progress_bar=True).astype("float32")

index = faiss.IndexFlatIP(embs.shape[1])  # cosine via inner product on normalized
index.add(embs)
D, I = index.search(embs, k=2)            # nearest neighbor (excluding self)
# Drop examples whose nearest neighbor similarity > 0.95
keep_mask = D[:, 1] < 0.95
ds_sem_dedup = ds_near_dedup.select(np.where(keep_mask)[0])
```

**MinHash thresholds.** 0.7 = aggressive (catches paraphrases, may over-prune); 0.85 = balanced; 0.95 = conservative (only catches near-exact). FineWeb uses ~0.85 with 5-shingles.

---

## 6. Quality Filtering

### 6.1 Rule-based heuristics (cheap, run first)

```python
import re
from langdetect import detect_langs

def low_quality(ex, min_chars=32, max_chars=32000):
    text = "\n".join(m["content"] for m in ex["messages"])
    n = len(text)
    if n < min_chars or n > max_chars:
        return True
    # Repetition rate: top-token frequency
    tokens = text.split()
    if len(tokens) > 20:
        top = max(tokens.count(t) for t in set(tokens))
        if top / len(tokens) > 0.3:
            return True
    # Special-char ratio
    special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special / max(1, n) > 0.4:
        return True
    # Language check (drop non-target language for monolingual training)
    try:
        if detect_langs(text[:500])[0].lang != "en":
            return True
    except Exception:
        return True
    return False

ds_quality = ds_sem_dedup.filter(lambda ex: not low_quality(ex), num_proc=8)
```

### 6.2 Classifier-based filtering

For pretraining-scale, train a fastText classifier on (high-quality, low-quality) seeds — this is what CCNet, FineWeb, and FineWeb-Edu do.

```python
# Pseudocode for the FineWeb-Edu pattern:
#   1. Use a strong LLM to score 500K web docs on educational value (0-5).
#   2. Train a small BERT regressor on (doc, score).
#   3. Apply the regressor to the entire CommonCrawl dump — keep score >= 3.
# Result: FineWeb-Edu (~1.3T tokens) outperforms FineWeb (~15T) on benchmarks.
# https://arxiv.org/abs/2406.17557
```

For SFT-scale (10K – 1M examples), use an LLM-as-judge with a fixed rubric and average over 3 prompts:

```python
import json
from openai import OpenAI

client = OpenAI()
RUBRIC = """Score the following (instruction, response) pair on a 1-5 scale:
1 = harmful/wrong, 2 = poor, 3 = acceptable, 4 = good, 5 = excellent.
Consider: relevance, factual accuracy, completeness, formatting, safety.
Return ONLY a JSON object: {"score": <int>, "reason": "<one sentence>"}"""

def score_example(instruction, response):
    out = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"Instruction: {instruction}\n\nResponse: {response}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(out.choices[0].message.content)["score"]
```

Watch for first-position bias when using LLM judges for pairwise preference — NVIDIA's Nemotron pipeline asks each judgment **twice with the order swapped** and only keeps the pair if both judgments agree.

### 6.3 PII and toxicity

```python
# Microsoft Presidio — PII detection + redaction
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def scrub_pii(text):
    results = analyzer.analyze(text=text, language="en",
                                entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                                          "CREDIT_CARD", "US_SSN", "IP_ADDRESS"])
    return anonymizer.anonymize(text=text, analyzer_results=results).text

# Toxicity — use Detoxify or a small classifier; drop or flag
# from detoxify import Detoxify
# scores = Detoxify("original").predict(text)
# if scores["toxicity"] > 0.8: drop
```

### 6.4 Eval contamination check

**Critical.** If your training data overlaps with the eval set, all your numbers are fake. Use n-gram overlap (Llama 2 / GPT-4 use 13-grams):

```python
def ngrams(text, n=13):
    toks = text.lower().split()
    return {tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)}

eval_ngrams = set()
for ex in eval_dataset:
    eval_ngrams |= ngrams(ex["question"] + " " + ex["answer"])

def contaminated(ex):
    text = "\n".join(m["content"] for m in ex["messages"])
    train_ng = ngrams(text)
    if not train_ng:
        return False
    overlap = len(train_ng & eval_ngrams) / len(train_ng)
    return overlap > 0.05  # tune; 5% n-gram overlap is the common cutoff

ds_clean = ds_quality.filter(lambda ex: not contaminated(ex), num_proc=8)
```

If your benchmark shows a sudden +20pt jump, suspect contamination before celebrating.

---

## 7. Synthesis and Augmentation

### 7.1 Self-Instruct (Wang et al. 2022)

Bootstrap an instruction dataset from a small seed set + a strong LLM. Alpaca used this with 175 seeds → 52K instructions.

```python
import random, json
from openai import OpenAI

client = OpenAI()

# Seed: 175 hand-written diverse (instruction, response) pairs
seeds = json.load(open("seeds.jsonl"))

EXPAND_PROMPT = """Below are examples of diverse task instructions. Generate 8
new instructions that are different from the examples in topic, format, and
style. Ensure they are concrete, solvable tasks.

Examples:
{exemplars}

New instructions (numbered 1-8):"""

def expand_instructions():
    sample = random.sample(seeds, 6)
    exemplars = "\n".join(f"- {s['instruction']}" for s in sample)
    out = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": EXPAND_PROMPT.format(exemplars=exemplars)}],
        temperature=1.0,
    )
    # Parse the numbered list
    return [line.split(". ", 1)[1] for line in out.choices[0].message.content.split("\n")
            if line and line[0].isdigit()]

def respond(instruction):
    out = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": instruction}],
        temperature=0.7,
    )
    return out.choices[0].message.content

# Self-Instruct quality filters (from Wang et al.):
def keep_generated(instr, resp, all_instrs):
    if len(instr) < 10 or len(instr) > 500: return False
    if len(resp) < 5: return False
    # Drop near-duplicates of existing instructions (use MinHash in production)
    if any(instr.lower() in x.lower() or x.lower() in instr.lower()
           for x in all_instrs): return False
    # Drop if response is just an echo of the input
    if instr.lower() in resp.lower()[:len(instr)]: return False
    return True

dataset = list(seeds)
for _ in range(1000):  # ~8K new examples
    new_instrs = expand_instructions()
    for instr in new_instrs:
        resp = respond(instr)
        if keep_generated(instr, resp, [d["instruction"] for d in dataset]):
            dataset.append({"instruction": instr, "response": resp})
```

**Heuristics from the original Self-Instruct paper to keep:** drop repetitive examples, instructions too long/short, examples where instruction and response disagree, and outputs that are repetitions of the input.

### 7.2 Evol-Instruct (WizardLM, Xu et al. 2023)

Instead of generating new instructions from scratch, **evolve** existing ones into more complex variants. Five operations: add constraints, deepen, concretize, increase reasoning steps, and introduce complications.

```python
EVOL_PROMPTS = {
    "constraints": "Rewrite the instruction adding one new constraint or requirement.",
    "deepen":      "Rewrite the instruction making it more specific and demanding.",
    "concretize":  "Replace abstract concepts with more concrete, specific ones.",
    "reasoning":   "Rewrite the instruction so it requires multi-step reasoning.",
    "breadth":     "Create a new instruction in the same domain but on a different aspect.",
}

def evolve(instr, op):
    prompt = f"{EVOL_PROMPTS[op]}\n\nOriginal: {instr}\n\nRewritten:"
    out = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return out.choices[0].message.content.strip()

# Multi-round evolution; prune failed evolutions (too short, off-topic, refusal)
evolved = []
for ex in seeds:
    cur = ex["instruction"]
    for _ in range(3):
        cur = evolve(cur, random.choice(list(EVOL_PROMPTS)))
        if len(cur) < 20 or "I cannot" in cur:
            break
        evolved.append({"instruction": cur, "response": respond(cur)})
```

### 7.3 OSS-Instruct / Magicoder (Wei et al. 2023)

Seed instruction generation from **real open-source code snippets** instead of random topics — yields more grounded, diverse code instructions than Self-Instruct on synthetic seeds. The Magicoder pattern: sample 1–10 lines of OSS code → ask the model to invent a coding problem that uses similar patterns → generate the solution → filter via execution.

### 7.4 Distillation: Alpaca-style and Orca-style

- **Alpaca-style** — student learns to mimic teacher outputs directly. Cheap and effective for surface style. Risk: superficial imitation (Gudibande et al. 2023) — the model learns to *sound like* the teacher without acquiring the underlying capability.
- **Orca-style** — teacher emits not just the answer but the **reasoning trace** (system messages like "Explain your reasoning step by step"). Student learns the chain of thought, not just the conclusion. Materially better for reasoning tasks.

```python
ORCA_SYSTEM = """You are a careful expert. For each answer:
1. State the question's core requirement.
2. Reason step by step using only stated facts.
3. State the final answer prefixed with FINAL:.
Be explicit; do not skip steps."""

def orca_response(instruction):
    out = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": ORCA_SYSTEM},
                  {"role": "user",   "content": instruction}],
        temperature=0.3,
    )
    return out.choices[0].message.content
```

### 7.5 Constitutional AI / RLAIF for preference data

When you can't afford human pairwise annotation, generate two responses and use an AI judge (with a written constitution) to pick the winner. This is how Anthropic's Constitutional AI bootstraps preference signal. Always run the judge **twice with order swapped** (Section 6.2) to neutralize position bias.

```python
def make_pair(instruction):
    a = respond(instruction)  # temperature 0.7
    b = respond(instruction)
    # Judge twice with swap
    j1 = judge(instruction, a, b)  # returns "A" or "B"
    j2 = judge(instruction, b, a)
    if j1 == "A" and j2 == "B":
        chosen, rejected = a, b
    elif j1 == "B" and j2 == "A":
        chosen, rejected = b, a
    else:
        return None  # disagreement → drop
    return {"prompt": instruction, "chosen": chosen, "rejected": rejected}
```

### 7.6 Backtranslation (Köksal et al. 2023, Li et al. 2023)

Reverse the generation direction: take **existing high-quality long-form content** (textbooks, papers, Wikipedia) and ask the model to invent a prompt that would have elicited it. This sidesteps the "AI can't write long, high-quality responses" problem because the responses are real human prose.

```python
BACKTRANSLATE = """Read the following passage. Write a single user instruction
that would have produced this passage as a response. The instruction should be
specific enough that this passage is a natural answer.

Passage:
{passage}

Instruction:"""
```

### 7.7 Caveats — read before running any of the above

1. **License contamination.** OpenAI's ToS forbids using its outputs to train competing models. Llama license restricts redistribution of derivative outputs. Gemini/Anthropic similarly. If you ship a model trained on undisclosed teacher outputs, you carry legal risk.
2. **Model collapse.** Recursive training on AI-generated data has been shown to cause irreversible defects (Shumailov et al. 2023). Mitigation: always mix synthetic with real human data, never train successive generations purely on the previous generation's outputs (Gerstgrasser et al. 2024).
3. **Lineage opacity.** If your synthetic data was generated by a model trained on benchmark X, your evaluation on X is contaminated and you don't know it. Track teacher model version, prompt, generation date, and seed for every synthetic example.
4. **Superficial imitation.** Student learns the surface form (length, headers, hedge phrases) without the underlying capability. Gudibande et al.: "improvements come from improving the base model, not imitation." Don't expect distillation alone to make a weak base model strong.
5. **Bias amplification.** Training on prior model outputs can amplify existing biases (Taori & Hashimoto 2023).

---

## 8. Annotation Operations

### 8.1 Annotation guidelines (the actual hardest part)

LinkedIn and others have reported that **writing good annotation guidelines is the single hardest part** of the AI engineering pipeline. Without them, IAA (inter-annotator agreement) collapses and the model has no consistent target to learn.

A guideline must answer:
- What does each label or score mean? Give 3+ examples per label.
- Edge cases — what about empty inputs, ambiguous queries, refusals?
- Tie-breaking rules — when does relevance beat correctness?
- Scoring rubrics — what's the difference between a 3 and a 4?

Run a **pilot round**: 50 examples × 3 annotators. Compute Cohen's kappa or Krippendorff's alpha. If <0.7, the guidelines are not yet specific enough — rewrite, don't proceed.

```python
from sklearn.metrics import cohen_kappa_score

# Two annotators, scores 1-5
ann_a = [4, 3, 5, 2, 4, 5, 3]
ann_b = [4, 4, 5, 2, 3, 5, 3]
print("Kappa:", cohen_kappa_score(ann_a, ann_b, weights="quadratic"))
# >= 0.7 = acceptable; < 0.5 = guidelines too vague
```

### 8.2 Active learning / human-in-the-loop

When annotation is expensive, label only the most informative examples. Two cheap strategies:

```python
# Uncertainty sampling — label examples the current model is least sure about
import torch.nn.functional as F

def margin_uncertainty(model, tokenizer, prompts, top_k=200):
    scores = []
    for p in prompts:
        # For a classifier: 1 - margin between top-2 logits
        logits = model(**tokenizer(p, return_tensors="pt")).logits[0]
        probs = F.softmax(logits, dim=-1).sort(descending=True).values
        scores.append(1.0 - (probs[0] - probs[1]).item())
    return sorted(zip(scores, prompts), reverse=True)[:top_k]

# Diversity sampling — embed unlabeled pool, k-means cluster, label one per cluster
```

### 8.3 AI-assisted annotation

Llama 3's pattern: AI generates an initial annotation, human reviews and corrects. Faster than human-only, more reliable than AI-only — but watch for **anchoring bias**: humans rubber-stamp AI proposals. Build in disagreement-rate monitoring; if humans agree with AI >95%, sample-audit a fresh batch with the AI label hidden.

### 8.4 Annotation tools

| Tool | Strengths | Use when |
|---|---|---|
| **Argilla** | Built for LLM data; native HF Datasets integration; AI-assisted | SFT/preference curation in HF stack |
| **Label Studio** | General-purpose; rich UI types; self-hosted | Heterogeneous tasks (NER, classification, segmentation) |
| **Prodigy** | Active learning out of the box; commercial | Small expert team, NLP focus |
| **Lilac** | Dataset exploration + clustering for SFT data | Pre-annotation dataset auditing |
| **Doccano** | OSS, lightweight | Simple text classification/NER |

---

## 9. Data Versioning and Lineage

Without versioning you can't reproduce your training run, debug regressions, or pass an audit.

| Tool | What it versions | Use when |
|---|---|---|
| **DVC** | Data + models + pipelines, Git-coupled | Want Git-style branching for data |
| **lakeFS** | Object storage; Git-like over S3 | Petabyte-scale, multiple teams |
| **HuggingFace Datasets revisions** | HF Hub-hosted | Public or org-internal sharing |
| **Weights & Biases artifacts** | Tied to experiments | Already using W&B for tracking |
| **Custom + content hashing** | Anything | You need full control |

```bash
# DVC quickstart — version a dataset
dvc init
dvc add data/sft_v3.jsonl                   # creates data/sft_v3.jsonl.dvc
git add data/sft_v3.jsonl.dvc .gitignore
git commit -m "sft v3: +12K math examples, dedup pass 2"

# Later, reproduce exact training data
git checkout <commit>
dvc pull
```

**Always emit a dataset card** (HuggingFace dataset card spec). Minimum:
- Source(s) and license per source
- Row counts, token counts, language distribution
- Generation method (human / AI / mixed; teacher model if AI)
- Filter pipeline (which filters, in what order, with what thresholds)
- Known issues, biases, limitations
- Eval contamination check status

Without these, your model card cannot honestly describe its training data.

---

## 10. Notable Open Datasets (Shortcuts)

Use these as starting points, ablation baselines, or augmentation. Always verify license against your use case; commercial use is not always allowed even when "open".

### Pretraining

| Dataset | Size | License | Notes |
|---|---|---|---|
| **FineWeb** | ~15T tokens | ODC-By 1.0 | High-quality CommonCrawl rebuild; replaces RefinedWeb in many setups |
| **FineWeb-Edu** | ~1.3T tokens | ODC-By 1.0 | Educational-value-filtered subset of FineWeb; punches above its weight |
| **RedPajama-V2** | ~30T tokens | Apache 2.0 | Multilingual, with quality signals |
| **Dolma** | ~3T tokens | ODC-By 1.0 | AI2's open pretraining mix; full lineage |
| **The Pile** | ~825GB | Mixed; some restricted | Older but well-studied; some sources retracted |
| **C4** | ~750GB | ODC-By | Cleaned CommonCrawl, used for T5 |

### SFT / instruction

| Dataset | Size | License | Notes |
|---|---|---|---|
| **FLAN Collection** | ~15M examples | Apache 2.0 | Task diversity (1,800+ tasks); the original instruction-tuning corpus |
| **OpenAssistant (oasst1/oasst2)** | ~160K convos | Apache 2.0 | Crowdsourced human conversations |
| **UltraChat** | ~1.5M multi-turn dialogues | MIT | GPT-3.5 synthetic; widely used |
| **OpenHermes 2.5** | ~1M examples | Mixed | Curated mixture; commercial-use varies |
| **Tulu-3 SFT mixture** | ~939K examples | ODC-By + per-source | AI2's current best open SFT mix |
| **Alpaca / Alpaca-cleaned** | ~52K examples | CC-BY-NC 4.0 | Historical; OpenAI-output-derived = noncommercial |

### Preference

| Dataset | Size | License | Notes |
|---|---|---|---|
| **HH-RLHF** | ~169K pairs | MIT | Anthropic's helpful/harmless pairs |
| **UltraFeedback** | ~64K examples × 4 responses | MIT | Multi-aspect preference (helpfulness, honesty, instruction-following, truthfulness) |
| **Nectar** | ~183K prompts × 7 ranked responses | Apache 2.0 | Pairwise extractable |

### Code

| Dataset | Size | License | Notes |
|---|---|---|---|
| **The Stack v2** | ~67TB / 658B tokens | Per-source (mostly permissive) | Filtered GitHub; the standard code-pretraining corpus |
| **CodeFeedback** | ~150K examples | Apache 2.0 | Code with execution feedback |
| **Magicoder OSS-Instruct** | ~75K examples | MIT | OSS-Instruct seeded code data |

---

## 11. RAG Indexing Data

RAG quality is bounded by chunk quality. The retrieval ceiling is set during indexing, not at query time.

```python
# Chunk with semantic awareness, not blind length cuts
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,           # tokens (approx via chars)
    chunk_overlap=100,        # to preserve cross-boundary context
    separators=["\n\n", "\n", ". ", " ", ""],  # break on natural boundaries first
)

chunks = []
for doc in corpus:
    for i, text in enumerate(splitter.split_text(doc["text"])):
        chunks.append({
            "id": f"{doc['id']}::{i}",
            "text": text,
            "source": doc["source"],
            "title": doc.get("title"),
            "url": doc.get("url"),
            "created_at": doc.get("created_at"),  # for time-aware retrieval
            "section": doc.get("section"),
        })
```

Quality checks specific to RAG:
- **Chunk coherence** — does each chunk stand alone, or does it start mid-sentence? Sample 50 chunks and read.
- **Metadata completeness** — every chunk needs `source`, ideally `title`, `url`, `created_at`, `section`. Without these, citations are impossible.
- **Deduplication at chunk level** — the same boilerplate (footers, navbars) appears in thousands of pages.
- **Embedding consistency** — re-embed everything when you change the embedding model. Mixed-embedding indexes silently degrade.

For RAG, see `../../ml-architectures/rag/` for indexing strategy and retrieval evaluation.

---

## 12. Common Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Model trains, loss drops, eval flat | Format mismatch (training format ≠ inference format) | Audit chat template; verify `apply_chat_template` round-trips |
| Eval skyrockets, real users complain | Eval contamination | Re-run 13-gram overlap check; rebuild eval set |
| Quality plateaus quickly | Coverage hole, not quantity | Plot length/topic histograms; add diverse buckets |
| First-finetune-iteration great, second worse | Model collapse or distribution narrowing | Stop chaining synthetic; mix in real data |
| Refuses too much or hedges | Over-aggressive safety filtering or RLHF data skew | Audit refusal-heavy preference pairs |
| Generates hallucinated reasoning | Trained on Alpaca-style answers without rationales | Switch to Orca-style generation; verify CoT is grounded |
| Output style drifts toward teacher quirks | Distillation imitation | Reduce teacher mix, improve base model |

---

## See Also

- `../feature-engineering/` — classical-ML counterpart (numeric/categorical encoding, TF-IDF)
- `../data-validation/` — schema, drift, and distribution checks for tabular and ML inputs
- `../../ml-training/training-workflow/` — end-to-end training process
- `../../ml-training/unsloth-sft/` — efficient SFT with Unsloth
- `../../ml-training/unsloth-advanced/` — DPO/ORPO/KTO with Unsloth
- `../../ml-libraries/huggingface/` — `datasets`, `transformers`, `trl` reference
- `../../ml-architectures/llm/` — FM architectures and what data each phase needs
- `../../ml-architectures/rag/` — retrieval pipelines and chunk-quality evaluation

---

## References

- HuggingFace Datasets — https://huggingface.co/docs/datasets/index
- HuggingFace Datasets repo — https://github.com/huggingface/datasets
- LIMA: Less Is More for Alignment (Zhou et al. 2023) — https://arxiv.org/abs/2305.11206
- Self-Instruct (Wang et al. 2022) — https://arxiv.org/abs/2212.10560
- Self-Instruct repo — https://github.com/yizhongw/self-instruct
- Evol-Instruct / WizardLM (Xu et al. 2023) — https://arxiv.org/abs/2304.12244
- Magicoder / OSS-Instruct (Wei et al. 2023) — https://arxiv.org/abs/2310.20689
- FineWeb (Penedo et al. 2024) — https://arxiv.org/abs/2406.17557
- FineWeb dataset — https://huggingface.co/datasets/HuggingFaceFW/fineweb
- RedPajama-V2 — https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2
- RefinedWeb (Penedo et al. 2023) — https://arxiv.org/abs/2306.01116
- Dolma (AI2) — https://github.com/allenai/dolma
- Tulu-3 SFT mixture — https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
- UltraChat 200K — https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- UltraFeedback — https://huggingface.co/datasets/openbmb/UltraFeedback
- Anthropic HH-RLHF — https://huggingface.co/datasets/Anthropic/hh-rlhf
- FLAN Collection — https://github.com/google-research/FLAN
- Argilla — https://github.com/argilla-io/argilla
- Label Studio — https://github.com/HumanSignal/label-studio
- DVC — https://github.com/iterative/dvc
- datasketch (MinHash, LSH) — https://github.com/ekzhu/datasketch
- Microsoft Presidio (PII) — https://github.com/microsoft/presidio
- Textbooks Are All You Need (phi / synthetic data quality) — https://arxiv.org/abs/2306.11644
- Databricks Dolly — https://github.com/databrickslabs/dolly
