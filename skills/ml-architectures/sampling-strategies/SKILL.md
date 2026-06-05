---
name: sampling-strategies
description: How autoregressive LMs pick next tokens — temperature, top-k/top-p/min-p, beam search, repetition penalties, logprobs, structured generation, self-consistency, test-time compute. Use when tuning generation quality, debugging hallucinations or repetition, extracting confidence scores, or choosing between deterministic and creative output modes.
---

# Sampling Strategies

## Why This Exists

**Problem**: Defaults are wrong for most use cases. `temperature=1.0, top_p=1.0` (the OpenAI default) produces hallucinations on factual queries because every token in the long tail keeps non-zero probability mass. Greedy decoding (`temperature=0`) mode-collapses creative tasks into bland repetition. Frameworks expose 8+ knobs (temperature, top-k, top-p, min-p, repetition_penalty, presence_penalty, frequency_penalty, no_repeat_ngram_size, typical_p, mirostat...) and engineers either leave them at defaults and blame "the model" for hallucinations, or tune them blindly without understanding the interactions.

**Key insight**: Sampling is a free, interpretable lever. The model's logits already encode its uncertainty — sampling chooses how to project that distribution into one token. Changing temperature 1.0 → 0.3 on a factual QA pipeline costs nothing, ships in one config line, and is often more impactful than a model upgrade. Conversely, the *same* model at `T=0` gives you a deterministic classifier and at `T=1.0, top_p=0.95` gives you a creative writer.

**Reach for this when**:
- Debugging hallucinations or factual errors that look "made up" → lower temperature, lower top-p, or use logprobs to detect low-confidence tokens
- Reducing variance in evaluation runs → `T=0` or `seed=...` for reproducibility
- Enforcing JSON / regex / grammar output → constrained sampling (outlines, xgrammar, vLLM `guided_decoding`)
- Getting better creative writing or brainstorming → `T=0.8–1.0` with `top_p=0.95`, possibly with repetition penalty
- Improving reasoning accuracy → self-consistency (sample N CoTs, majority-vote)
- Building a classifier from a generative LM → read logprobs of the answer tokens, no fine-tune needed
- The model "never stops" or terminates mid-JSON → stop tokens / EOS handling, not sampling

## Sampling Fundamentals

```
input → transformer → logits ∈ ℝ^V → /T → softmax → P(token) → sample → token
```

Every sampling strategy is a **transformation on the logit vector** before the categorical sample. They compose: temperature first, then truncation (top-k / top-p / min-p), then optional penalties, then sample.

```python
import torch
import torch.nn.functional as F

def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None, min_p=None):
    """logits: shape [vocab_size]. Returns sampled token id."""
    # 1. Temperature
    if temperature == 0:
        return logits.argmax().item()  # greedy; T=0 is a special case
    logits = logits / temperature

    # 2. Top-k truncation (mask everything below the k-th largest logit)
    if top_k is not None:
        kth_value = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < kth_value, torch.full_like(logits, -float("inf")), logits)

    # 3. Top-p (nucleus) truncation
    if top_p is not None:
        sorted_logits, sorted_idx = logits.sort(descending=True)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Keep tokens with cumulative prob ≤ top_p (always keep at least one)
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits[mask] = -float("inf")
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)

    # 4. Min-p: keep tokens whose prob ≥ min_p × p_max
    if min_p is not None:
        probs = logits.softmax(dim=-1)
        threshold = min_p * probs.max()
        logits = torch.where(probs < threshold, torch.full_like(logits, -float("inf")), logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
```

**Numerical stability**: Always work in log-space. With `vocab_size=128k`, the smallest probabilities underflow FP16 / BF16. `log_softmax` keeps the gradients well-defined and matches what HF and vLLM return when you ask for `logprobs`.

### Temperature: the master knob

Temperature `T` divides logits before softmax: `softmax(logits / T)`.

| T | Effect | Use |
|---|--------|-----|
| 0 | argmax (greedy) — fully deterministic | Classification, evals, structured extraction |
| 0.1–0.3 | Sharp distribution — picks top-1 ~95% of the time | Factual QA, code, math, function calling |
| 0.5–0.7 | Balanced | Chat, summarization, instruction following |
| 0.8–1.0 | Diverse — long tail gets real mass | Creative writing, brainstorming, synthetic data |
| >1.0 | Very flat — incoherent for most models | Rarely useful; some MCTS / RL setups |

`T → 0` is a softmax limit you can't compute directly (division by zero). Implementations special-case it as `argmax`.

`T → ∞` gives a uniform distribution; this is purely theoretical — at `T=2` most production models already incoherent.

## Sampling Strategies

### Greedy (T=0)
Always pick `argmax(logits)`. Deterministic given identical hardware, but two failure modes for free-form generation:

1. **Mode collapse**: long generations repeat — `"... and the cat and the cat and the cat ..."` because greedy is locally optimal but globally bad.
2. **Inconsistency under hardware drift**: Different GPUs / kernels / batch sizes round logits differently; identical input + `T=0` can still produce different outputs across hardware. Set `seed` and pin hardware if you need bitwise reproducibility.

Use for: classification, evals, function-call argument extraction.

### Beam search
Maintain top-k partial sequences; expand each, keep top-k by cumulative log-prob. Optimal for tasks with a single best output (translation, constrained decoding, summarization with strict ROUGE optimization).

**Failure mode for open-ended generation**: beam search produces *boring*, generic outputs. Holtzman et al. (2019) showed that human text has *lower* per-token probability than beam-search outputs — the most probable continuation isn't the most natural one. Don't beam-search for chat or creative work.

```python
out = model.generate(input_ids, num_beams=4, length_penalty=1.0, early_stopping=True)
```

`length_penalty > 1` favors longer outputs; `< 1` favors shorter. With `length_penalty=0` you get raw cumulative log-prob, which biases toward short outputs (sum of negative numbers).

### Top-k
Keep only the top-k logits, renormalize, sample. `k=50` is a common default.

**Failure mode**: `k` is fixed regardless of context. For "Yes or no?" you'd want `k=2`; for "What's a creative metaphor for time?" you'd want `k=200`. Top-k clips the distribution either too early or too late depending on context.

### Top-p (nucleus sampling)
Sort tokens by probability, keep the smallest set whose cumulative probability ≥ `p`. Holtzman et al. (2019). Default `p=0.9–0.95` for chat.

**Why it works**: dynamically adapts the candidate set to context confidence. Sharp distribution → tiny nucleus → near-greedy. Flat distribution → wide nucleus → diverse. This matches how human writing is high-entropy in some places (creative phrasing) and low-entropy in others (function words, punctuation).

**Failure mode**: at very high `p` (0.99) you re-include the long tail and get hallucinations. At very low `p` (0.5) you re-introduce mode collapse. Combine with temperature, not as a replacement.

### Min-p
Keep tokens whose probability ≥ `min_p × p_max`. Nguyen et al. (2024). Default `min_p=0.05–0.1`.

**Why it's better than top-p in some setups**: top-p with high temperature still admits low-probability garbage when the distribution is flat (the cumulative sum just walks further). Min-p anchors to the *peak* — if the top token has 50% prob and `min_p=0.1`, only tokens with ≥ 5% prob survive. Particularly useful for high-temperature creative sampling where you want diversity without nonsense.

```python
# vLLM, HuggingFace, llama.cpp, and SGLang all support min_p directly.
SamplingParams(temperature=1.0, top_p=1.0, min_p=0.05)
```

### Typical / locally-typical sampling
Meister et al. (2022). Instead of "most probable tokens," sample tokens whose log-prob is *close to the entropy of the distribution* — i.e., tokens that are surprisingly probable but not the most probable. Implements an information-theoretic notion of "typical" output.

In practice: rarely used in production, but available in HF as `typical_p`. Sometimes helps in story generation; usually min-p or top-p does the same job with one fewer knob.

### Mirostat
Adaptive: targets a fixed perplexity by adjusting `top_k` per step. Used in llama.cpp / Kobold AI for long-form RP. Niche; if you don't already know you need it, you don't.

## Repetition Control

LLMs repeat for two reasons: (1) the loss function rewards confident predictions, so once a phrase appears it's the most-confident continuation; (2) attention can lock onto a recurring pattern.

| Knob | Formula | When to use |
|------|---------|-------------|
| `repetition_penalty` (CTRL paper) | divide logit of seen tokens by α (>1 penalizes) | General chat; α=1.05–1.15 |
| `frequency_penalty` (OpenAI) | subtract α × token_count from logit | Discourage *frequent* repeats; α=0.1–0.5 |
| `presence_penalty` (OpenAI) | subtract α from logit if token appeared *at all* | Discourage *any* reuse — encourages new vocabulary; α=0.1–0.5 |
| `no_repeat_ngram_size` | hard ban on repeating any n-gram of size n | Beam search summaries; n=3 typical |

**Anti-pattern**: applying repetition penalties to structured tasks. JSON has `{`, `}`, `,`, `:` which legitimately repeat — repetition_penalty=1.2 will produce malformed JSON. Same for code (every function returns, every Python line might start with whitespace).

**Anti-pattern**: stacking all four. They interact unpredictably. Pick one.

```python
# Chat: repetition_penalty alone usually sufficient
out = model.generate(input_ids, do_sample=True, temperature=0.7, top_p=0.9,
                     repetition_penalty=1.1)

# JSON / code: NO repetition penalties
out = model.generate(input_ids, do_sample=False, temperature=0)
```

## Stop Sequences and EOS

A common production bug: the model "never stops" and burns tokens until `max_new_tokens`. Causes:

1. **Wrong chat template**: the model was trained with `<|eot_id|>` but you're decoding without that token in `eos_token_id`. Fix: pass `eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]`.
2. **Stop tokens too short**: stop on `"\n"` truncates valid multi-line responses. Stop on `"\n\n"` is usually safer.
3. **Constrained sampling with no EOS in the grammar**: if your JSON schema doesn't allow EOS, the model literally cannot stop. Always include the implicit "end of document" transition.
4. **Streaming clients ignoring stop**: you set `stop=["</answer>"]` but the client buffers and shows the stop token. The server *did* stop; the client renders incorrectly.

```python
# vLLM: stop_token_ids and stop strings are both supported
SamplingParams(stop=["</s>", "<|eot_id|>"], stop_token_ids=[128009], max_tokens=512)
```

## Logprobs and Self-Evaluation

`logprobs` = `log(P(token))`. Production model APIs increasingly expose them (OpenAI: top-20; Anthropic: not exposed; vLLM: arbitrary; HF: arbitrary).

### Extracting logprobs (HuggingFace)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                                             device_map="auto", torch_dtype="bfloat16")

inputs = tok("The capital of France is", return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=10,
    do_sample=False,
    output_scores=True,            # list of [batch, vocab] logits per step
    return_dict_in_generate=True,
)

# Convert raw scores to log-probs
import torch
generated = out.sequences[0, inputs.input_ids.shape[1]:]
for step, score in enumerate(out.scores):
    lp = torch.log_softmax(score[0], dim=-1)
    tok_id = generated[step].item()
    print(f"{tok.decode([tok_id])!r}\t logprob={lp[tok_id].item():.3f}")
```

### Classifier via logprobs

When you need to classify text but only have a generative model: prompt for the answer tokens, compare their logprobs.

```python
def classify(prompt, labels=("positive", "negative", "neutral")):
    """Pick the label whose first token has the highest logprob."""
    label_token_ids = [tok(f" {label}", add_special_tokens=False).input_ids[0]
                       for label in labels]
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]   # last position, [vocab]
    label_logprobs = torch.log_softmax(logits, dim=-1)[label_token_ids]
    return labels[label_logprobs.argmax().item()], label_logprobs.softmax(-1).tolist()

label, probs = classify("Review: 'best film of the year' Sentiment:", )
# → ('positive', [0.92, 0.05, 0.03])
```

This is faster than generating + parsing and gives you a calibrated probability. Use for: sentiment, topic classification, multiple-choice eval (MMLU, ARC), guardrail decisions.

### Confidence-aware generation
Average logprob per generated token approximates how "sure" the model is.

```python
mean_lp = sum(torch.log_softmax(s[0], -1)[g].item()
              for s, g in zip(out.scores, generated)) / len(generated)
# mean_lp ≈ -0.2 → confident, ≈ -1.5 → uncertain (often hallucinated)
```

Threshold this for a cheap hallucination detector — abstain or escalate to a stronger model when `mean_lp` is low.

## Self-Consistency: Sample N, Vote

Wang et al. (2022). For tasks with a discrete answer (math, multiple-choice, extraction), sample N independent CoT reasoning chains and majority-vote on the final answer. Robust to the brittleness of any single sample.

```python
from collections import Counter
import re

def self_consistency(prompt, n=10, temperature=0.7):
    answers = []
    for _ in range(n):
        out = model.generate(
            **tok(prompt, return_tensors="pt").to(model.device),
            max_new_tokens=512, do_sample=True, temperature=temperature, top_p=0.95,
        )
        text = tok.decode(out[0], skip_special_tokens=True)
        # Extract final answer — task-specific (here: "The answer is X")
        m = re.search(r"answer is\s+([A-D]|\-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if m:
            answers.append(m.group(1))
    if not answers:
        return None, 0.0
    top, count = Counter(answers).most_common(1)[0]
    return top, count / len(answers)   # answer + agreement ratio
```

**Why diversity matters**: identical samples don't vote; you need temperature > 0. Wang reports `T=0.7, top_p=0.95, N=40` for GSM8K. Diminishing returns past N=20–40 in practice.

**Cost**: N× tokens. Combine with prefix caching (vLLM auto-handles this for shared prompts) to keep cost ≈ 1× prefill + N× decode.

## Structured / Constrained Generation

The mechanism: at each step, mask logits for tokens that would violate the grammar, then sample from the survivors.

```
logits → grammar.allowed(tokens) → mask → softmax → sample
```

Three levels of strictness:

1. **JSON mode** (OpenAI, Anthropic, vLLM): guarantees parseable JSON, not schema-conformant.
2. **JSON schema / Pydantic**: enforces field names, types, enums.
3. **Regex / CFG**: arbitrary formats — phone numbers, SQL, custom DSLs.

### vLLM with guided decoding

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "email": {"type": "string", "format": "email"},
    },
    "required": ["name", "age"],
}

params = SamplingParams(
    temperature=0.0,
    max_tokens=200,
    guided_decoding=GuidedDecodingParams(json=schema),
)
out = llm.generate(["Extract user info: 'Alice, 30, alice@example.com'"], params)
print(out[0].outputs[0].text)  # always valid against schema
```

vLLM supports `GuidedDecodingParams(json=..., regex=..., choice=[...], grammar=...)` and uses xgrammar / outlines / lm-format-enforcer as backends. See the `../../ml-libraries/vllm/` skill for engine config.

### When constrained sampling is **lossy**

The grammar masks logits *after* the model picked them. If the model's preferred next token isn't in the grammar, you force it onto a less-probable path — and that bad first choice contaminates everything downstream (snowballing, per Zhang et al. 2023).

Example: schema requires `{"answer": <int>}` but the model wants to answer `"unknown"`. Constrained sampling forces it to invent a number. You shipped a hallucination because the schema forbade "I don't know."

**Mitigation**: include an escape hatch in the schema (`answer: int | "unknown"`), or use a two-stage approach (first ask "Do you know?" → if yes, generate JSON).

For grammars (CFG) and engine choice details, see `../../ml-libraries/sglang/` and `../../ml-libraries/vllm/`.

## Test-Time Compute

Sample more, win more — at a cost. Three strategies, increasing in cleverness:

| Strategy | How | Compute multiplier | When |
|----------|-----|---|------|
| Best-of-N | Sample N independently, pick best by reward / verifier / logprob | N× | Reward model available, e.g. coding |
| Self-consistency | Sample N CoTs, majority-vote final answer | N× | Discrete answer (math, classification) |
| Beam search w/ process reward | Score *each step* of CoT with a PRM, prune bad branches | ~k×depth | Math, planning (Snell et al. 2024) |
| Tree search (MCTS) | Expand promising branches more | variable | Research; o1-style reasoning |

OpenAI's o1 / DeepSeek-R1 / Qwen-QwQ are roughly "huge amount of CoT samples internally + RL'd to terminate when confident." From the API surface they look like one slow generation, but inside they're doing lots of test-time compute.

Brown et al. (2024, "Large Language Monkeys") showed coverage scales **log-linearly** with N up to N=10,000 for code+math — i.e., a smaller model with massive sampling can match a much larger one. This is the empirical foundation for the "compute-optimal inference" framing.

**Cost reality**: production rarely samples N=400 or 10,000. Reasonable budgets:
- Self-consistency for math/eval: N=5–32
- Best-of-N for code with tests: N=4–16
- Tree search: only when you have a fast verifier (compiler, theorem prover)

## Speculative Sampling

A draft model proposes K tokens; the target model verifies them in one parallel forward pass; accepted tokens advance the prefix, rejected ones are resampled from a corrected distribution. **The output distribution is mathematically identical to sampling from the target model alone** — pure speedup, no quality loss.

This is a serving-time optimization, not a sampling-strategy choice. See `../../ml-training/inference-optimization/` for vLLM speculative decoding setup, EAGLE / Medusa / draft model selection.

## Decision Table

| Task | temperature | top_p | top_k | min_p | repetition_penalty | extra |
|------|------------|-------|-------|-------|--------------------|-------|
| Factual QA | 0–0.2 | 1.0 | — | — | — | seed for reproducibility |
| Code generation | 0.1–0.3 | 0.95 | — | — | — | stop on language fence |
| Function-call args | 0 | — | — | — | — | constrained JSON schema |
| JSON extraction | 0 | — | — | — | **none** | guided_decoding |
| Classification (gen LM) | n/a — read logprobs | — | — | — | — | argmax label tokens |
| Math (single-shot) | 0–0.3 | 0.95 | — | — | — | + CoT prompt |
| Math (self-consistency) | 0.7 | 0.95 | — | — | — | N=10–40, vote |
| Chat / instruction | 0.6–0.8 | 0.9 | — | — | 1.05–1.1 | — |
| Creative writing | 0.8–1.0 | 0.95 | — | 0.05 | 1.05–1.1 | min-p often beats top-p |
| Brainstorming | 0.9–1.2 | 0.95 | — | 0.05 | 1.1 | sample N, dedupe |
| Synthetic data gen | 0.8–1.0 | 0.95 | — | 0.05 | — | high temp = diversity |
| Eval / benchmark | 0 | 1.0 | — | — | — | reproducibility |
| Translation | 0–0.3 | — | — | — | — | beam_search=4 acceptable |
| Summarization | 0.3–0.5 | 0.9 | — | — | — | no_repeat_ngram_size=3 |

## HuggingFace Reference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Suggest 3 names for a coffee shop near MIT."}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tok(prompt, return_tensors="pt").to(model.device)

# Creative sampling
out = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    min_p=0.05,
    repetition_penalty=1.05,
    eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")],
    pad_token_id=tok.eos_token_id,
)

# Deterministic JSON-ish task
out = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,           # equivalent to T=0
    repetition_penalty=1.0,    # OFF for structured output
)

# With logprobs
out = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True, temperature=0.7,
    output_scores=True, return_dict_in_generate=True,
)
gen_ids = out.sequences[0, inputs.input_ids.shape[1]:]
lps = [torch.log_softmax(s[0], -1)[t].item() for s, t in zip(out.scores, gen_ids)]
print("avg logprob:", sum(lps) / len(lps))
```

## vLLM Reference

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16",
          gpu_memory_utilization=0.9)

# Chat
chat = SamplingParams(temperature=0.7, top_p=0.9, min_p=0.05, max_tokens=512,
                      repetition_penalty=1.05,
                      stop_token_ids=[128009])  # <|eot_id|> for Llama-3

# Self-consistency: n=10 in a single SamplingParams
sc = SamplingParams(n=10, temperature=0.7, top_p=0.95, max_tokens=1024,
                    seed=42)  # vLLM batches the n samples; prefix cache reused

# JSON-schema constrained
schema = {"type": "object",
          "properties": {"city": {"type": "string"},
                         "population": {"type": "integer"}},
          "required": ["city", "population"]}
struct = SamplingParams(temperature=0, max_tokens=200,
                        guided_decoding=GuidedDecodingParams(json=schema))

# Logprobs (top-K per position)
lp = SamplingParams(temperature=0, max_tokens=10, logprobs=20, prompt_logprobs=1)

outputs = llm.generate(["Where is MIT?"], lp)
for out in outputs[0].outputs:
    for tok_logprobs in out.logprobs:   # list of dict[token_id, Logprob]
        for tok_id, info in tok_logprobs.items():
            print(info.decoded_token, info.logprob)
```

## Common Failure Modes

1. **Hallucinations on factual queries** → temperature too high. Drop to 0.0–0.3, drop top_p to 0.9.
2. **Boring, repetitive output** → temperature too low or beam search. Raise to 0.7+, drop beam.
3. **Model never terminates** → wrong / missing `eos_token_id`. Inspect tokenizer's chat template.
4. **JSON parse failures** → repetition penalty applied. Set to 1.0; consider `guided_decoding`.
5. **Self-consistency doesn't help** → `temperature=0` so all samples identical. Use `T≥0.5`.
6. **Different outputs on same `T=0` input** → hardware drift (different GPU, batch size, kernel). Pin `seed` *and* hardware *and* batch composition. For strict reproducibility, single-batch + fixed seed + same GPU class.
7. **Logprobs look uniform / random** → check you're decoding the right tokens. HF `output_scores` gives logits *before* warpers, not after; pass `output_logits=True` (newer API) or apply your own log_softmax.
8. **Constrained sampling produces wrong answers** → schema forbade the model's preferred path. Add an "unknown" / "abstain" branch to the grammar.

## See Also

- `../llm/` — model architectures that emit the logits you're sampling
- `../transformer/` — attention mechanism and KV-cache that make inference work
- `../../ml-libraries/vllm/` — production engine for sampling at scale, `SamplingParams` reference, prefix caching
- `../../ml-libraries/sglang/` — alternative engine; constrained generation via xgrammar
- `../../ml-training/inference-optimization/` — speculative decoding, batching, quantization for serving
- `../../ml-training/prompt-engineering/` — sampling settings interact with CoT / few-shot prompting
- `../../ml-training/llm-evaluation/` — eval pipelines need `T=0` + fixed seed for reproducibility

## References

- HuggingFace `GenerationConfig` API: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
- HuggingFace blog, "How to generate text" (Patrick von Platen): https://huggingface.co/blog/how-to-generate
- vLLM API reference (SamplingParams, GuidedDecodingParams): https://docs.vllm.ai/en/latest/api/
- Holtzman et al. 2019, "The Curious Case of Neural Text Degeneration" (top-p / nucleus): https://arxiv.org/abs/1904.09751
- Meister et al. 2022, "Locally Typical Sampling": https://arxiv.org/abs/2202.00666
- Nguyen et al. 2024, "Min-p Sampling": https://arxiv.org/abs/2407.01082
- Wang et al. 2022, "Self-Consistency Improves Chain of Thought Reasoning": https://arxiv.org/abs/2203.11171
- Brown et al. 2024, "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling": https://arxiv.org/abs/2407.21787
- Chip Huyen, *AI Engineering* (O'Reilly, 2024), Chapter 2 — Sampling
