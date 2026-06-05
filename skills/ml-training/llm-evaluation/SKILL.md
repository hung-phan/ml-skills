---
name: llm-evaluation
description: Foundation-model evaluation — exact eval (functional correctness), reference-based (BLEU, ROUGE, BERTScore, COMET), AI-as-judge, hallucination detection (SelfCheckGPT, SAFE, Ragas faithfulness), public benchmarks (MMLU, HumanEval, Arena), evaluation-driven development pipelines. Use when picking between LLMs, building task-specific eval suites, debugging hallucinations, scoring RAG/agent systems, or replacing flaky public benchmarks with a real scorecard.
---

# LLM / Foundation Model Evaluation

## Why This Exists

Public benchmarks lie. A model topping MMLU may flop on your support tickets. Accuracy on multiple-choice questions tests *discrimination* (pick the right answer from 4); it does not test *generation* (write a fluent, faithful, on-format answer). Worse, every benchmark released before a model trained is probably in the training set — Schaeffer (2023) showed a 1M-param model "achieving" near-perfect benchmark scores by training exclusively on test data. Hugging Face's June 2024 leaderboard refresh dropped GSM-8K and MMLU because they were saturated and contaminated.

Foundation-model evaluation is a different toolkit from classical ML evaluation:

- **Classical ML eval** (see `../evaluation/`): F1, ROC-AUC, calibration, threshold optimization, confusion matrices on a labelled test set.
- **FM eval** (this skill): functional correctness on generated artifacts, AI-as-judge with bias mitigation, reference-free quality (perplexity, semantic similarity), hallucination detection against a context or open knowledge, multi-criterion scorecards.

Chip Huyen calls the right approach **evaluation-driven development**: define the eval before building. Without an eval pipeline you cannot tell good prompts from bad, cannot catch quality regressions when you swap models, cannot prove the chatbot is worth its cost, and cannot defend a launch decision. Huyen: "evaluation is the biggest bottleneck to AI adoption."

**Reach for this skill when:**

- Picking between models (GPT-4o vs Claude Sonnet vs Llama 3.1 vs Qwen) for a real task
- A model swap or prompt tweak silently regressed quality and you need a regression test
- Defining success criteria for a new LLM/RAG/agent application before writing code
- Debugging hallucinations in a RAG system
- Replacing "the demo looked good" with a defensible scorecard
- Picking benchmarks for a model card or external report

## The Evaluation Criteria Taxonomy

Every LLM application needs a small **scorecard** of 3–5 criteria. Bigger scorecards never get used; smaller ones miss critical failure modes. Most criteria fall into four buckets (Huyen ch. 4):

| Bucket | What it measures | Example metrics |
|---|---|---|
| **Domain capability** | Can the model do *the task* (code, medical QA, Latin translation)? | HumanEval pass@1, BIRD-SQL execution accuracy, internal task accuracy |
| **Generation capability** | Is the *output* fluent, coherent, faithful, relevant? | Factual consistency (NLI / Ragas faithfulness), AI-judge fluency, BERTScore vs reference |
| **Instruction following** | Does the output respect format, length, constraints? | IFEval, JSON-schema validity, regex match, length compliance |
| **Cost & latency** | Can you afford to ship it? | $/1M tokens, time-to-first-token (TTFT) p90, tokens/s, end-to-end p95 |

Real scorecard for a customer-support chatbot might be: factual consistency vs KB ≥ 0.9, JSON-schema validity ≥ 0.99, AI-judge helpfulness ≥ 4/5, p90 TTFT < 500 ms, $/conversation < $0.02.

---

## Method 1: Exact Evaluation

When the output has a verifiable ground truth — execute and check.

| Method | Use when | Example |
|---|---|---|
| **Functional correctness** | Code, SQL, math with checkable answer | HumanEval, MBPP, BIRD-SQL, GSM8K (extract final number) |
| **Exact match** | Short answers, span extraction | "What year did X happen?" |
| **Regex / schema match** | Structured outputs | `\d{4}-\d{2}-\d{2}` for dates, `pydantic.BaseModel.model_validate_json` for JSON |
| **Execution accuracy w/ efficiency** | SQL, code where slow ≠ correct | BIRD-SQL compares query runtime to ground truth runtime |

```python
# Functional correctness — HumanEval style.
import subprocess, tempfile, textwrap

def run_test(generated_code: str, test_code: str, timeout_s: int = 5) -> bool:
    """Return True if generated_code passes test_code under exec."""
    program = textwrap.dedent(generated_code) + "\n" + textwrap.dedent(test_code)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run(["python", path], capture_output=True, timeout=timeout_s)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False

# pass@k: probability that at least one of k samples is correct.
import numpy as np
def pass_at_k(n: int, c: int, k: int) -> float:
    """n total samples, c correct, k = budget. Closed-form unbiased estimator (Chen 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

Functional correctness is the gold standard when you can apply it. **Use it for any task where the output is executable.**

---

## Method 2: Reference-Based — Lexical

You have one or more reference outputs and want to score lexical overlap. Mostly relevant for translation and summarization with refs.

| Metric | Granularity | Notes |
|---|---|---|
| **BLEU** | n-gram precision (modified) + brevity penalty | Translation. Insensitive to meaning-preserving rewrites; corpus-level only meaningful. |
| **ROUGE** (1, 2, L) | n-gram or LCS recall | Summarization. ROUGE-L = longest common subsequence. |
| **METEOR** | unigram alignment with stem/synonym match | Better than BLEU for single-sentence MT but slower. |
| **chrF / chrF++** | char-n-gram F-score | Robust for morphologically rich languages. |
| **Exact token-level F1** | token bag-of-words F1 | SQuAD-style QA. |

```python
# sacrebleu is the reproducible corpus BLEU/chrF (don't use NLTK BLEU for papers).
import sacrebleu
refs = [["the cat sat on the mat", "a cat is on the mat"]]  # list-per-sentence
hyps = ["the cat is on the mat"]
print(sacrebleu.corpus_bleu(hyps, list(zip(*refs))).score)
print(sacrebleu.corpus_chrf(hyps, list(zip(*refs))).score)

# rouge-score for summarization.
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
print(scorer.score("reference summary text", "candidate summary text"))
```

**When NOT to use lexical metrics:** open-ended generation (chat, creative writing), questions with many valid phrasings, anything where meaning matters more than wording. They are hostile to paraphrase.

---

## Method 3: Reference-Based — Semantic

When you have references but need to credit semantically equivalent rewrites.

| Metric | Backbone | Notes |
|---|---|---|
| **BERTScore** | BERT/DeBERTa contextual embeddings, token-level cosine | General-purpose; default is `roberta-large` (English) or `xlm-roberta-large` (multilingual). |
| **BLEURT** | RemBERT fine-tuned on human ratings | Often correlates better with humans than BERTScore but English-centric. |
| **COMET** | XLM-R fine-tuned on MQM/DA judgements | State-of-the-art for translation; needs source + reference + hypothesis. |
| **MoverScore** | BERT + Earth Mover's Distance | Captures partial semantic overlap; slower than BERTScore. |

```python
# BERTScore example.
from bert_score import score
P, R, F1 = score(
    cands=["the cat is on the mat"],
    refs=["a cat sat on the mat"],
    lang="en", model_type="roberta-large", verbose=False,
)
print(F1.item())   # ~0.97

# COMET (translation only — needs source).
# pip install unbabel-comet
from comet import download_model, load_from_checkpoint
model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
print(model.predict([{"src": "le chat", "mt": "the cat", "ref": "a cat"}], gpus=0))
```

---

## Method 4: Reference-Free — Perplexity & Cross-Entropy

The model's own probability of a sequence. Useful when you don't have references.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(model_id)
mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()

@torch.no_grad()
def perplexity(text: str) -> float:
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    out = mdl(ids, labels=ids)
    return torch.exp(out.loss).item()

print(perplexity("The capital of France is Paris."))   # low
print(perplexity("Colorless green ideas sleep furiously."))   # high
```

**Critical gotcha:** Perplexity is **only comparable across models with the same tokenizer**. Llama-3 vs GPT-2 perplexity numbers are not on the same scale. Use bits-per-byte (`loss / log(2) / num_bytes`) if you must compare across tokenizers.

Perplexity is also useful for **contamination detection**: if a benchmark item has anomalously low perplexity under the candidate model, suspect leakage.

---

## Method 5: AI-as-Judge

The default for open-ended evaluation. Two flavors:

- **Pairwise** — "Is response A or B better for prompt P?" Used by Chatbot Arena, MT-Bench. More reliable than absolute scoring; avoids the "what does a 7/10 mean?" problem.
- **Pointwise** — "Score this response 1–5 on faithfulness." Used by G-Eval, Prometheus, Ragas. Scales but suffers from anchoring.

### Known biases (Zheng et al. 2023, *MT-Bench*)

| Bias | Symptom | Mitigation |
|---|---|---|
| **Position bias** | Judge prefers the first (or last) candidate | Run both orders, average; or randomize |
| **Verbosity bias** | Longer answers score higher | Cap response length; explicitly tell judge "ignore length" |
| **Self-preference** | GPT-4 prefers GPT-4 outputs | Use a different family as judge, or ensemble judges |
| **Authority/format bias** | Markdown headers, numbered lists score higher | Strip formatting before judging when not relevant |
| **Egocentric bias** | Judge agrees with whoever it generated last | Reset chat; one judgement per call |
| **Refusal contamination** | Judge gives high scores to "safe but useless" outputs | Add explicit rubric line about helpfulness |

Mitigation rule: **always run pairwise both orders and report agreement rate**; tie-rate above 30 % means weak judge or near-identical models.

### Reliable AI-judge with structured output

```python
# Anthropic SDK, structured judge with rationale + scores 1-5.
import json
import anthropic

client = anthropic.Anthropic()

JUDGE_SYSTEM = """You are an evaluation judge. Read the prompt, the assistant response,
and the rubric. Score the response 1-5 on each criterion. Be terse. Ignore length —
do not reward verbosity. Output JSON only."""

JUDGE_USER = """<prompt>{prompt}</prompt>
<response>{response}</response>
<rubric>
- faithfulness: response is supported by the provided context (no hallucination)
- helpfulness: response actually answers the user's question
- format: response follows the requested JSON schema
</rubric>

Output JSON: {{"faithfulness": int, "helpfulness": int, "format": int, "rationale": str}}"""

def judge(prompt: str, response: str) -> dict:
    msg = client.messages.create(
        model="claude-sonnet-4-5",  # use a strong, *different-family* judge
        max_tokens=400,
        temperature=0,                # MUST be 0 for reproducibility
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": JUDGE_USER.format(prompt=prompt, response=response)}],
    )
    return json.loads(msg.content[0].text)

# Pairwise — run both orders and aggregate.
def pairwise_judge(prompt: str, a: str, b: str) -> str:
    """Returns 'A', 'B', or 'tie' after running both orders."""
    forward = _ask("A", "B", prompt, a, b)
    reverse = _ask("A", "B", prompt, b, a)   # roles flipped
    if forward == reverse:           # consistent under swap → trust
        return forward
    return "tie"
```

**Always set judge `temperature=0`**, log the full judge prompt with your eval results, and report inter-judge agreement (Cohen's κ if you ensemble).

---

## Method 6: Human Evaluation

Worth the cost when:

- Stakes are high (medical, legal, safety-critical)
- AI-judge agreement with humans on your task is unverified or low
- You need a North Star to validate AI judges (LinkedIn evaluates ~500 conversations/day manually)
- Subtle quality dimensions matter (humor, empathy, voice consistency)

**Inter-annotator agreement** must be reported. Without it, your "labels" are noise.

| Coefficient | Use case |
|---|---|
| **Cohen's κ** | Two annotators, categorical labels |
| **Fleiss' κ** | 3+ annotators, categorical |
| **Krippendorff's α** | Any number of annotators, any scale, handles missing data |
| **ICC (intraclass correlation)** | Continuous ratings |

Rule of thumb: κ > 0.6 is "substantial agreement", > 0.8 "almost perfect". Below 0.4 your rubric is broken — rewrite it before continuing.

```python
# Krippendorff's alpha example
# pip install krippendorff
import krippendorff
import numpy as np

# rows = annotators, cols = items; np.nan = missing
ratings = np.array([
    [4, 5, 3, np.nan, 2],
    [4, 4, 3, 5, 2],
    [3, 5, 4, 4, 2],
])
print(krippendorff.alpha(ratings, level_of_measurement="ordinal"))
```

---

## Hallucination & Factual Consistency Detection

Two regimes (Huyen ch. 4):

- **Local** — output must be consistent with a *given context* (RAG, summarization, customer support over a KB)
- **Global** — output must be consistent with *open-world facts* (general chatbot, fact-checking)

### Local (against a context)

| Method | Idea | Library |
|---|---|---|
| **NLI-based** | Treat (context, claim) as (premise, hypothesis); use entailment classifier | DeBERTa-v3-mnli-fever-anli, AlignScore |
| **Ragas faithfulness** | Decompose response into claims; verify each via LLM judge against context | `ragas` |
| **AlignScore** | Lightweight specialized model trained on diverse alignment data | `alignscore` |
| **FactScore** | Atomic-fact verification against a reference document set | `factscore` |

```python
# Ragas — RAG faithfulness + answer relevance.
# pip install ragas
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

ds = Dataset.from_dict({
    "question":     ["What is the warranty period for product X?"],
    "answer":       ["Product X has a 2-year warranty."],
    "contexts":     [["Our flagship product X comes with a 2-year limited warranty."]],
    "ground_truth": ["The warranty for product X is 2 years."],
})
result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(result)   # {'faithfulness': 1.0, 'answer_relevancy': 0.92, ...}
```

### Global (no context — open knowledge)

| Method | Idea | Cost |
|---|---|---|
| **SelfCheckGPT** (Manakul 2023) | Generate N samples; if they disagree, the original is likely hallucinated | N+1 model calls per claim |
| **SAFE** (Wei 2024) | Decompose → make self-contained → search Google → verify | High; needs search API |
| **Chain-of-Verification** | Model drafts; model writes verification questions; model answers them; model revises | 3–4 model calls |
| **TruthfulQA** | Benchmark of 817 known-misleading questions with reference answers | Free if you accept their dataset |

```python
# SelfCheckGPT-style consistency check — pure-Python sketch.
# Real impl: pip install selfcheckgpt
import asyncio
from sentence_transformers import SentenceTransformer, util
import torch

embedder = SentenceTransformer("all-MiniLM-L6-v2")

async def generate(prompt: str, temperature: float) -> str:
    """Stub — call your LLM here."""
    ...

async def selfcheck_consistency(prompt: str, n_samples: int = 5) -> float:
    """Generate the 'main' response at temp=0 and N samples at temp=1.
    Score = mean cosine similarity between main and samples.
    Lower = more hallucinated."""
    main = await generate(prompt, temperature=0.0)
    samples = await asyncio.gather(*[generate(prompt, temperature=1.0) for _ in range(n_samples)])

    embs = embedder.encode([main] + samples, convert_to_tensor=True)
    sims = util.cos_sim(embs[0:1], embs[1:]).squeeze(0)
    return sims.mean().item()
```

The SelfCheckGPT paper proposes more sophisticated scorers (BERTScore-based, NLI-based, n-gram, prompt-based); the cosine version above is a fast first cut.

---

## Public Benchmarks: What to Use, What to Distrust

| Benchmark | Tests | Trust today | Notes |
|---|---|---|---|
| **MMLU** (Hendrycks 2020) | 57 subjects MCQ | Saturated, contaminated | Don't ship a model card with only this |
| **MMLU-Pro** (Wang 2024) | Harder MMLU, 10 options, more reasoning | Less saturated | Replaced MMLU on HF leaderboard |
| **MMLU-Redux** | Cleaned MMLU subset (errors removed) | Useful | Known label errors in original |
| **BIG-Bench / BBH** | 200+ diverse tasks; BBH = 23 hardest | BBH still useful | Most BIG-Bench tasks now solved |
| **HellaSwag** | Common-sense sentence completion | Saturated | Train sets in pretraining corpora |
| **ARC** (Easy/Challenge) | Grade-school science MCQ | ARC-C still mildly useful | Easy variant solved |
| **WinoGrande** | Pronoun resolution | Saturated | |
| **GSM8K** | Grade-school math word problems | Saturated, contaminated | Use GSM-Hard or MATH instead |
| **MATH** (Hendrycks) | Competition math | Lvl-5 still useful | Filter to hardest tier |
| **HumanEval** (Chen 2021) | 164 Python functions | Saturated | Use HumanEval+, MBPP+, LiveCodeBench |
| **MBPP** | Basic Python | Saturated | Use MBPP+ |
| **LiveCodeBench** | Codeforces/LeetCode, dated | Trustworthy | Filter to post-cutoff problems |
| **BIRD-SQL** | Real-world text-to-SQL with efficiency | Trustworthy | Execution + runtime |
| **IFEval** | 25 verifiable instruction types | Trustworthy | Format/length compliance |
| **GPQA** (Rein 2023) | Graduate-level science | Trustworthy | "Google-proof" — frontier still struggles |
| **AGIEval** | Human exams (SAT, LSAT, …) | Mixed | Some leakage |
| **Chatbot Arena** | Pairwise human votes, head-to-head | Trustworthy directionally | Voter bias toward verbose/markdown answers; battle counts skew |
| **MT-Bench** | 80 multi-turn conversations, judged by GPT-4 | Useful but small | 80 examples is too few for fine comparison |
| **TruthfulQA** | 817 misleading questions | Useful for hallucination | Specialized GPT-judge available |

### Contamination — how to spot it

- **N-gram overlap** between eval items and known training corpora (Common Crawl, The Pile, etc.). 13-token overlaps are the standard threshold.
- **Anomalously low perplexity** of eval items under the candidate model.
- **Format-shift drop**: paraphrase the eval question; if accuracy collapses, the model memorized the surface form.
- **Date-aware splits**: filter benchmarks to items released *after* the model's training cutoff (LiveCodeBench does this by design).
- **Leaderboard volatility**: a model jumping to first place on one benchmark while flat on correlated ones is a red flag.

OpenAI found 13 benchmarks with ≥ 40% contamination in GPT-3's training (Brown 2020). Treat all pre-2024 public benchmarks as partially contaminated unless proven otherwise.

---

## Building YOUR Eval Pipeline (Evaluation-Driven Development)

Public benchmarks weed out terrible models. **Your private eval picks the right model for your app.** Five steps.

### Step 1 — Define application-specific criteria

A 3–5 metric scorecard, one per critical failure mode. From Huyen's customer-support example:

| Criterion | Metric | Hard requirement | Stretch |
|---|---|---|---|
| Factual consistency vs KB | Ragas faithfulness | ≥ 0.85 | ≥ 0.95 |
| Helpfulness | AI-judge (1–5) | ≥ 4.0 mean | ≥ 4.5 |
| Format compliance | JSON-schema validity | ≥ 0.99 | 1.00 |
| Latency | TTFT p90 | < 800 ms | < 400 ms |
| Cost | $/conversation | < $0.05 | < $0.02 |

Tie metrics to business: "faithfulness ≥ 0.9 lets us auto-resolve 50% of tickets."

### Step 2 — Build the eval set

- Target **100–300 high-quality examples** per slice initially. Add more only when bootstrap confidence intervals are still wide.
- **Mine from production logs** if you have them. Synthetic eval data lies; user inputs surprise you.
- Cover slices: head/tail of distribution, known failure modes, out-of-scope inputs, adversarial inputs.
- Annotate with rubric examples (Huyen calls this "the backbone of a reliable pipeline").
- Keep an out-of-scope set: inputs the app should refuse.

OpenAI's rough sample-size rule: detect a 10% difference → ~100 samples; 3% difference → ~1000; 1% → ~10000.

### Step 3 — Pick judges per metric

| Metric type | Judge |
|---|---|
| Format / regex / schema | Functional check (free, deterministic) |
| Factual consistency vs context | Ragas / NLI model / AI judge |
| Helpfulness, tone, voice | AI judge (pairwise preferred) |
| Stickiness / engagement | Production telemetry, not offline eval |
| Subtle quality | Human spot-check on 1–5% sample |

Mix cheap and expensive: a fast NLI classifier on 100% of traffic plus an expensive AI judge on 1%.

### Step 4 — Regression-test on every change

Every prompt edit, model swap, retrieval-config tweak runs the eval suite. Ship-blocker: any hard-requirement metric regresses. Use experiment tracking (`../experiment-tracking/`) so you can answer "what changed between v17 and v18?".

### Step 5 — Monitor in production

- Sample N% of prod requests, run them through the same judges.
- Collect user feedback signals (thumbs, regenerate clicks, dwell time, escalations).
- Alert on drift: judge scores moving > 1σ from baseline over a rolling window.
- Periodically pull recent prod traffic into the eval set to keep it representative.

---

## Decision: Metric Type by Task

| Task | Primary metric | Secondary | Hallucination check |
|---|---|---|---|
| Classification / extraction | Accuracy, F1, MCC (see `../evaluation/`) | Calibration | n/a |
| Code generation | pass@k (functional) | AI-judge readability | Compilation/runtime errors |
| SQL generation | Execution accuracy + runtime ratio | AI-judge readability | n/a |
| Translation | COMET | chrF, BLEU | n/a |
| Summarization | Ragas faithfulness, ROUGE-L | AI-judge coherence | NLI vs source |
| Open-ended chat | AI-judge pairwise | User engagement | SelfCheckGPT or Ragas |
| RAG QA | Faithfulness + answer relevance | Context precision/recall | Built into faithfulness |
| Agent / tool-use | Task success rate | Step-level correctness, # turns, cost | Per-step verification |
| Structured-output extraction | JSON-schema validity + field-level accuracy | Latency | n/a |

## Decision: AI-Judge vs Human Eval

| Use AI judge when | Use human eval when |
|---|---|
| Eval set > 200 items and budget is tight | Stakes high (medical, legal, safety) |
| You need fast iteration (CI loop) | Validating a new AI judge |
| Task is verifiable from text alone | Judging humor, voice, empathy |
| You have validated agreement with humans | Edge cases the AI judge gets wrong |
| You can afford GPT-4-class judge | Final pre-launch sign-off |

**Always validate AI judges against human labels on a sample first.** If GPT-4-judge correlates < 0.6 with humans on your task, do not rely on it.

---

## Tools

| Tool | Strength | Weakness |
|---|---|---|
| **lm-evaluation-harness** (EleutherAI) | 400+ academic benchmarks, model-agnostic, standard | Academic style; not great for app-level eval |
| **HELM** (Stanford CRFM) | Holistic scenarios, well-documented, mean-win-rate aggregation | Expensive; less hackable |
| **Inspect AI** (UK AISI) | Modern Python, clean API, agent-eval first-class, sandboxed tool use | Newer; smaller benchmark catalog |
| **OpenAI Evals** | Easy registration of new evals, GPT-grader templates | OpenAI-flavored; less active than alternatives |
| **Promptfoo** | YAML-driven, great DX for prompt regression, side-by-side UI | Less suited to research-grade eval |
| **Ragas** | RAG metrics out-of-the-box (faithfulness, answer relevance, context precision/recall) | Tied to LLM-as-judge for most metrics |
| **DeepEval** | pytest-style API, large metric catalog, CI integration | Inconsistent docs across versions |
| **TruLens** | Triad of context-relevance / groundedness / answer-relevance for RAG | Smaller community |
| **Phoenix (Arize)** | Tracing + eval, OTel integration | Heavier setup |
| **LangSmith / Langfuse / Braintrust / HoneyHive** | Observability + dataset versioning + eval runs in one product | Vendor lock-in; pricing varies |

### lm-evaluation-harness CLI

```bash
# pip install lm-eval
# Evaluate a HF model on MMLU-Pro and IFEval
lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16 \
    --tasks mmlu_pro,ifeval \
    --batch_size auto \
    --device cuda \
    --output_path ./eval_results

# Against an OpenAI-compatible endpoint (vLLM, TGI, llama.cpp server)
lm_eval \
    --model local-completions \
    --model_args base_url=http://localhost:8000/v1/completions,model=my-model \
    --tasks gpqa_main_zeroshot,bbh \
    --apply_chat_template \
    --output_path ./eval_results
```

### Inspect AI sketch

```python
# pip install inspect_ai
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

@task
def my_eval():
    return Task(
        dataset=[
            Sample(input="What is the capital of France?", target="Paris"),
            Sample(input="2 + 2 = ?", target="4"),
        ],
        solver=generate(),
        scorer=includes(),
    )
# Run:  inspect eval my_eval.py --model openai/gpt-4o-mini
```

---

## Common Anti-Patterns

| Don't | Do |
|---|---|
| Quote MMLU as "model X is better" | Note the contamination risk; use MMLU-Pro or domain-specific eval |
| Use a single AI judge with no bias check | Validate against humans, run pairwise both orders, log judge prompt |
| Compare perplexity across models with different tokenizers | Compare bits-per-byte, or only same-tokenizer models |
| Average BLEU and accuracy into one number | Keep multi-criterion scorecard; weighting is application-specific |
| Eval set of 20 cherry-picked examples | 100–300 minimum; bootstrap to check stability |
| BLEU/ROUGE for open-ended chat | AI-judge or task-success metrics |
| Skip eval to ship faster | Quality regressions cost more than the eval ever does |
| Re-use the same eval set for prompt iteration *and* final reporting | Hold out a "test" eval that you only run before launch |

---

## See Also

- `../evaluation/` — **classical ML evaluation** (F1, ROC-AUC, calibration, SMOTE, threshold tuning). This skill is the foundation-model counterpart; reach for `../evaluation/` when your task is classification/regression with structured features rather than open-ended generation.
- `../prompt-engineering/` — once you have an eval, prompt iteration is bounded by it.
- `../../ml-architectures/rag/` — RAG-specific architecture; pair with Ragas faithfulness + context precision/recall here.
- `../../ml-architectures/agents/` — agent eval needs step-level + end-to-end metrics; Inspect AI is built for it.
- `../../ml-libraries/huggingface/` — `evaluate` library, `lm-evaluation-harness`, BERTScore.
- `../experiment-tracking/` — log eval runs alongside training runs so model swaps are reproducible.

---

## References

### Tools

- lm-evaluation-harness (EleutherAI): https://github.com/EleutherAI/lm-evaluation-harness
- HELM (Stanford CRFM): https://crfm.stanford.edu/helm/
- Inspect AI (UK AISI): https://github.com/UKGovernmentBEIS/inspect_ai
- OpenAI Evals: https://github.com/openai/evals
- Promptfoo: https://github.com/promptfoo/promptfoo
- Ragas: https://github.com/explodinggradients/ragas
- DeepEval: https://github.com/confident-ai/deepeval
- Chatbot Arena leaderboard: https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard

### Papers

- Zheng et al. 2023 — *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*: https://arxiv.org/abs/2306.05685
- Manakul et al. 2023 — *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection*: https://arxiv.org/abs/2303.08896
- Wei et al. 2024 — *Long-Form Factuality in Large Language Models* (SAFE): https://arxiv.org/abs/2403.18802
- Min et al. 2023 — *FactScore: Fine-grained Atomic Evaluation of Factual Precision*: https://arxiv.org/abs/2305.14251
- Hendrycks et al. 2020 — *MMLU: Measuring Massive Multitask Language Understanding*: https://arxiv.org/abs/2009.03300
- Chen et al. 2021 — *Evaluating Large Language Models Trained on Code* (HumanEval): https://arxiv.org/abs/2107.03374
- Wang et al. 2024 — *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark*: https://arxiv.org/abs/2406.01574

### Book

- Chip Huyen, *AI Engineering* (O'Reilly, 2024), ch. 4 "Evaluate AI Systems" — primary source for the eval-driven-development framing in this skill.
