---
name: model-merging
description: Combining finetuned checkpoints into one without retraining — task arithmetic, TIES-Merging, DARE, SLERP, model soup, frankenmerge, evolutionary merging, LoRA adapter merging via mergekit. Use when composing specialist models, recovering capabilities lost during alignment, building MoE-like composites cheaply, or replacing expensive multi-task training.
---

# Model Merging

Combine multiple finetuned checkpoints into a single model with **no extra training** (or minimal post-merge fine-tuning). Works on a CPU. The de-facto tool is **mergekit**.

- **mergekit**: https://github.com/arcee-ai/mergekit
- **Maxime Labonne's merging guide**: https://huggingface.co/blog/mlabonne/merge-models
- **Task arithmetic paper**: https://arxiv.org/abs/2212.04089
- **TIES-Merging**: https://arxiv.org/abs/2306.01708
- **DARE**: https://arxiv.org/abs/2311.03099
- **Model Soups**: https://arxiv.org/abs/2203.05482
- **Evolutionary Merging (Sakana)**: https://arxiv.org/abs/2403.13187

## Why This Exists

**Problem**: Training one model per task explodes serving cost. Multi-task SFT data is expensive and hard to balance — simultaneous finetuning needs more data and compute, sequential finetuning hits **catastrophic forgetting**, and even single-task finetuning often regresses on capabilities the base model had (math, coding, multilingual). RLHF and safety alignment also clip useful behaviors.

**Key insight**: For models finetuned on top of the **same base**, the difference `θ_finetuned − θ_base` is a **task vector** (delta parameters) that captures the essence of that task. Task vectors can be **added, subtracted, scaled, and combined** in weight space — and the resulting model recovers most multi-task quality for **$0 of additional training**. Most parameter changes during finetuning are redundant; pruning them before merging removes interference between specialists.

**Reach for this when**:
- You have 2+ specialist checkpoints from the same base and want one model that does all of them.
- You want to **recover** general capability lost during heavy RLHF / safety tuning by re-mixing in the base or a less-aligned finetune.
- You can't afford joint multi-task training, or task data is locked in different orgs/devices (federated).
- You want to **upscale** a model (depthwise scaling, frankenmerge into a larger N-layer net) before further training.
- You want to seed a Mixture-of-Experts model from dense checkpoints (sparse upcycling).

**Don't use this when**:
- Constituent models have **different base models / tokenizers** — weight-space merges will be garbage. Frankenmerge of the embedding/output layers may still work but typically needs further training.
- You need predictable, monotonic gains. Merging is a search problem; expect to sweep weights and eval.
- A single model already passes your eval. Merging adds risk of subtle regressions on tasks you didn't measure.

## Mental model

Three families of merge operations (Huyen Ch. 7):

| Family | What it does | Output size | Needs further training? |
|---|---|---|---|
| **Summing** (linear avg, SLERP, TIES, DARE) | Add/blend parameters of same-shape layers | Same as base | No (usually) |
| **Layer stacking** (passthrough, frankenmerge) | Take layer L from model A, layer L+1 from model B, etc. | Different (often larger) | Yes, almost always |
| **Concatenation** (LoRA rank concat) | Stack adapters side-by-side | Larger (sum of ranks) | No, but loses the memory-saving benefit of merging |

Within summing, the choice of method depends on how many candidates you have and how similar they are. The decision tree below covers the common cases.

## Method decision table

| You have... | Do this | Why |
|---|---|---|
| 2 finetunes of the same base, similar domains | **SLERP** | Smooth geodesic interpolation between two points; standard 2-model merge |
| 3+ finetunes of the same base | **TIES** or **DARE-TIES** | Resolves sign disagreements / interference between task vectors |
| Many finetunes, redundant params dominate | **DARE** (drop & rescale) | Random-mask drops 90%+ of delta weights, rescales survivors — minimal interference |
| All finetunes from same base, want simplest baseline | **Linear (Model Soup)** | Plain weighted average; works surprisingly well |
| Want to compose capabilities (math + chat + code) | **Task Arithmetic** | Add task vectors of each specialist on top of base |
| Want to **remove** a behavior (toxicity, an over-aligned refusal) | **Negate task vector** | `θ_new = θ_base − α · (θ_aligned − θ_base)` |
| Different sizes / want bigger model | **Passthrough / frankenmerge** | Stack layers — needs further finetuning |
| Two LoRA adapters, want both behaviors | PEFT `add_weighted_adapter` (linear/cat/svd/ties/dare) | Combines adapters in PEFT directly |
| Don't know what to do, have compute | **Evolutionary merging** | Let CMA-ES search merge ratios + layer assignments |

### Merge vs MoE vs Router (when to do which)

| Goal | Use |
|---|---|
| Single static model, all tasks at one inference cost | **Merge** (TIES/DARE) |
| Sparse routing, want capacity per task without latency tax | **Sparse upcycling** → MoE (start from a merge, then train router) |
| Have a strong query classifier already | **Router + N specialists** (no merge); pays N× memory but each call is cheap |
| Need ensembling-quality at any cost | Output-level **ensemble** (vote / weighted) — N× inference cost |

## The methods

### 1. Linear interpolation / Model Soup
Pure weighted average: `θ_merged = Σ w_i · θ_i` with `Σ w_i = 1` (or unconstrained for task vectors). Wortsman et al. 2022 showed averaging multiple finetuned checkpoints (different hyperparam runs) of the **same** base often beats picking the best individual run on out-of-distribution data, with no extra inference cost. The "uniform soup" averages all; the "greedy soup" adds runs one at a time only if they improve held-out eval.

### 2. Task arithmetic (Ilharco et al. 2022)
The foundational delta-parameter formulation:

```
τ_i  = θ_finetuned_i − θ_base       # task vector for task i
θ*   = θ_base + Σ_i  α_i · τ_i      # additive composition
θ-   = θ_base − α · τ_bad            # task negation (forget a task)
θ_AB = θ_base + α · (τ_A + τ_B − τ_C)  # analogy: "A is to B as C is to ?"
```

`α` is typically in `[0.3, 1.0]`. `α=1` for a single task vector reproduces the finetune exactly. Sweep `α` per task on a held-out eval — there is no closed-form optimum.

### 3. TIES-Merging (Yadav et al. 2023)
Linear summing breaks down when many task vectors disagree on the sign of a parameter. TIES fixes this in three steps:
1. **Trim**: For each task vector, keep only the top-k (typically top 20%) of parameters by magnitude; reset the rest to zero. Yadav et al. showed top 20% gives near-100% performance.
2. **Elect sign**: For each parameter index, pick the sign (+ or −) with the larger total magnitude across task vectors.
3. **Disjoint merge**: Average only the task vectors that agree with the elected sign.

This dramatically reduces interference when merging 3+ specialists.

### 4. DARE (Yu et al. 2023)
**D**rop **A**nd **RE**scale: randomly zero out a fraction `p` of task-vector entries, then rescale the survivors by `1/(1−p)` to preserve expectations. Yu et al. show you can drop **90%+** of delta parameters with negligible quality loss — making most of finetuning's parameter movement redundant. Often combined with TIES (`dare_ties` in mergekit).

### 5. SLERP — Spherical Linear Interpolation
For two vectors on a hypersphere of constant norm, SLERP traces the geodesic (great-circle arc) between them at constant angular velocity:

```
SLERP(θ_A, θ_B; t) = sin((1−t)·Ω)/sin(Ω) · θ_A  +  sin(t·Ω)/sin(Ω) · θ_B
```

where `Ω = arccos(<θ_A, θ_B> / (||θ_A||·||θ_B||))` and `t ∈ [0, 1]`. SLERP preserves the magnitude of the originals better than naive linear interpolation — useful when the two checkpoints have similar scales but moved in different directions. **Limitation**: SLERP is defined only for two vectors; chain it sequentially for 3+ models, but the result depends on order.

### 6. Pass-through / Frankenmerge
Take layers `[0..k]` from model A and layers `[k+1..N]` from model B (or interleave). Output is a different size than either constituent; almost always requires post-merge SFT to recover. Goliath-120B (alpindale, 2023) was built this way from two Llama-2-70B finetunes (Xwin + Euryale, 72 of 80 layers each).

### 7. Depthwise upscaling (Kim et al. 2023, SOLAR-10.7B)
A specific frankenmerge for **growing** a model: copy the base, sum N overlapping layers, stack the rest. SOLAR-10.7B = 32-layer 7B doubled to 64, with 16 middle layers summed → 48 layers, then continued pretrained.

### 8. Evolutionary merging (Akiba et al. 2024 / Sakana AI)
Treat per-layer merge weights and layer-assignment indices as a vector, optimize with CMA-ES against a held-out eval. Found non-obvious recipes that beat hand-tuned merges, especially across-domain (e.g. Japanese-LLM × math-LLM → Japanese-math LLM). Expensive — needs hundreds of eval rollouts — but completely automatic.

### 9. LoRA adapter merging
Adapters are tiny task vectors. PEFT exposes `add_weighted_adapter` with combination types `linear`, `cat`, `svd`, `ties`, `dare_linear`, `dare_ties`. Then `model.merge_and_unload()` folds the chosen adapter into the base for inference-latency-free serving. See PEFT docs: https://huggingface.co/docs/peft/main/en/developer_guides/lora#merge-lora-weights-into-the-base-model

## When merging works vs. fails

| Factor | Helps merging | Hurts merging |
|---|---|---|
| Same base model | ✅ required for weight-space merges | ❌ different bases → frankenmerge only |
| Same architecture / shape | ✅ required for summing | Different sizes need projection or stacking |
| Domain proximity | Instruction + chat + math from the same base | Vision encoder + LLM, or wildly different domains |
| Number of specialists | 2–4 with TIES/DARE works well | 8+ → interference compounds, evolutionary search helps |
| Permutation symmetry | Same init → aligned features | Different inits → consider Git Re-Basin alignment first |
| Magnitude balance | Task vectors of similar L2 norm | One specialist with huge norm dominates — rescale |
| Eval coverage | Multi-task held-out set | Single eval → easy to overfit merge weights |

**Catastrophic interference** is the failure mode. Two task vectors that move parameter `i` in opposite directions cancel under linear merging. TIES (sign election) and DARE (random sparsification) are direct fixes. If you suspect different inits on the same architecture (rare for HF checkpoints, but possible for re-pretrained variants), align with **Git Re-Basin** (Ainsworth et al. 2022) before averaging.

## Practical workflow

1. **Pick base + 2–3 finetunes**, all with the same architecture and tokenizer.
2. **Verify shape compatibility**: `state_dict().keys()` and per-tensor `shape` must match.
3. **Choose method** from the decision table. Default for 3+ same-base specialists: **DARE-TIES**.
4. **Sweep weights** on a held-out eval. For TIES, the knobs are per-model `weight` and `density` (top-k fraction kept). For DARE, the knob is drop probability `p`. Reasonable starting grid:
   - `weight ∈ {0.3, 0.5, 0.7}` per model
   - `density ∈ {0.5, 0.7, 0.9}` (TIES top-k)
   - `p ∈ {0.7, 0.85, 0.95}` (DARE drop prob)
5. **Eval all candidates** on a multi-task suite (one task per specialist + a general eval like MMLU/HellaSwag). Pick the Pareto-best.
6. **Optional**: short post-merge SFT (1k–10k examples on a balanced multi-task set) to clean up rough edges. Usually unnecessary for summing methods, mandatory for frankenmerge.
7. **Sanity check**: regression test on the base model's strengths — alignment merges easily clip safety behavior.

## Code: mergekit YAML for TIES merge of 3 checkpoints

Install: `pip install mergekit`

```yaml
# config.yml — TIES merge of three same-base specialists
models:
  - model: meta-llama/Llama-3.1-8B-Instruct
    # base — no parameters; serves as the reference
  - model: yourorg/llama-3.1-8b-math-sft
    parameters:
      density: 0.5     # keep top 50% of task-vector params
      weight: 0.5      # contribution to the merged task vector
  - model: yourorg/llama-3.1-8b-code-sft
    parameters:
      density: 0.5
      weight: 0.3
  - model: yourorg/llama-3.1-8b-chat-sft
    parameters:
      density: 0.5
      weight: 0.4
merge_method: ties
base_model: meta-llama/Llama-3.1-8B-Instruct
parameters:
  normalize: true       # rescale weights to sum to 1
  int8_mask: true       # save memory during merge
dtype: bfloat16
tokenizer_source: base  # use base model's tokenizer
```

Run:
```bash
mergekit-yaml config.yml ./merged-model \
  --cuda --copy-tokenizer --allow-crimes --out-shard-size 5B
# output ready for HF transformers / vLLM / SGLang
```

### DARE-TIES variant

```yaml
merge_method: dare_ties
parameters:
  normalize: true
models:
  - model: meta-llama/Llama-3.1-8B-Instruct
  - model: yourorg/llama-3.1-8b-math-sft
    parameters:
      density: 0.3     # for DARE this is "keep fraction" = 1 − drop prob
      weight: 0.5
  - model: yourorg/llama-3.1-8b-code-sft
    parameters:
      density: 0.3
      weight: 0.5
base_model: meta-llama/Llama-3.1-8B-Instruct
dtype: bfloat16
```

### SLERP between two models

```yaml
merge_method: slerp
base_model: meta-llama/Llama-3.1-8B-Instruct  # also slot 0
slices:
  - sources:
      - model: yourorg/llama-3.1-8b-math-sft
        layer_range: [0, 32]
      - model: yourorg/llama-3.1-8b-chat-sft
        layer_range: [0, 32]
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]   # per-layer interpolation factor
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5                      # default
dtype: bfloat16
```

### Frankenmerge (passthrough) — depthwise upscale

```yaml
merge_method: passthrough
slices:
  - sources:
      - model: meta-llama/Llama-3.1-8B-Instruct
        layer_range: [0, 24]
  - sources:
      - model: meta-llama/Llama-3.1-8B-Instruct
        layer_range: [8, 32]
dtype: bfloat16
# result: 24 + 24 = 48 layers (vs. 32 in original) — needs further training
```

## Code: pure-PyTorch task arithmetic

Sometimes you want full control or merging logic outside mergekit. The minimal recipe:

```python
import torch
from transformers import AutoModelForCausalLM

BASE = "meta-llama/Llama-3.1-8B-Instruct"
FT_A = "yourorg/llama-3.1-8b-math-sft"
FT_B = "yourorg/llama-3.1-8b-code-sft"

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
ft_a = AutoModelForCausalLM.from_pretrained(FT_A, torch_dtype=torch.bfloat16)
ft_b = AutoModelForCausalLM.from_pretrained(FT_B, torch_dtype=torch.bfloat16)

base_sd = base.state_dict()
sd_a    = ft_a.state_dict()
sd_b    = ft_b.state_dict()

# Compute task vectors and merge: theta_new = theta_base + a*tau_A + b*tau_B
alpha_a, alpha_b = 0.5, 0.5
merged = {}
for k, v_base in base_sd.items():
    if k in sd_a and k in sd_b and sd_a[k].shape == v_base.shape:
        tau_a = sd_a[k].to(torch.float32) - v_base.to(torch.float32)
        tau_b = sd_b[k].to(torch.float32) - v_base.to(torch.float32)
        merged[k] = (v_base.to(torch.float32) + alpha_a * tau_a + alpha_b * tau_b).to(v_base.dtype)
    else:
        merged[k] = v_base

# Optional: poor-man's TIES — keep only top-k% by magnitude per task vector
def trim(tau, density=0.2):
    flat = tau.abs().flatten()
    if flat.numel() == 0:
        return tau
    k = max(1, int(density * flat.numel()))
    thresh = torch.topk(flat, k).values.min()
    return torch.where(tau.abs() >= thresh, tau, torch.zeros_like(tau))

# Save merged weights into a fresh model and persist
base.load_state_dict(merged)
base.save_pretrained("./merged-task-arith")
# tokenizer copies from base — same tokenizer guaranteed
```

This is the underlying primitive. Real TIES adds the sign-election step over per-parameter aggregated magnitudes; real DARE adds Bernoulli masking and rescaling. Use mergekit unless you need a custom merge rule.

## Code: LoRA adapter merging with PEFT

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Load first adapter as the active one
model = PeftModel.from_pretrained(base, "yourorg/llama3-math-lora", adapter_name="math")
model.load_adapter("yourorg/llama3-code-lora",  adapter_name="code")
model.load_adapter("yourorg/llama3-chat-lora",  adapter_name="chat")

# Combine three adapters into one new adapter
model.add_weighted_adapter(
    adapters=["math", "code", "chat"],
    weights=[0.4, 0.3, 0.3],
    adapter_name="combined",
    combination_type="ties",      # also: "linear", "cat", "svd", "dare_linear", "dare_ties"
    density=0.5,                  # TIES/DARE only: keep fraction
)
model.set_adapter("combined")

# Bake the adapter into base weights for zero-overhead inference
merged = model.merge_and_unload()
merged.save_pretrained("./llama3-merged-adapter")
```

`combination_type` notes:
- `linear` — plain weighted sum of LoRA matrices
- `cat` — concatenate ranks (rank grows; output is bigger)
- `svd` — concatenate then truncate via SVD to a target rank
- `ties` / `dare_ties` — apply trimming + sign election before combining (best for 3+ adapters)

## Common pitfalls

- **Wrong base**: A finetune from `Llama-3.1-8B` and a finetune from `Llama-3-8B` will silently produce nonsense when merged in weight space. Always confirm the parent commit / base.
- **Tokenizer drift**: If one finetune extended the vocab (added special tokens), the embedding matrices have different shapes. Either re-trim or use frankenmerge for those layers.
- **Magnitude blowup**: With unnormalized task arithmetic and `α=1` for many specialists, the merged norm explodes. Set `normalize: true` in mergekit, or sweep `α<1`.
- **Overfit to merge eval**: Sweeping weights on a small eval can overfit. Hold out a final test set you don't touch during the sweep.
- **Merging chat templates**: Two finetunes with different chat templates will tokenize the same string differently. Decide on a single template post-merge; eval with that template.
- **Heavy RLHF source**: Merging in an RLHF model with strong refusal patterns can transfer those refusals. If you want to *remove* over-refusal, try **task negation**: subtract a fraction of the RLHF task vector from the base and add the SFT-only specialists.
- **Believing the leaderboard**: Open LLM leaderboard models that are merges sometimes overfit to the leaderboard's eval distribution. Always run your own task-specific eval.

## Sweep harness sketch

```python
import itertools, subprocess, json
from pathlib import Path

CONFIG_TEMPLATE = """
merge_method: ties
base_model: meta-llama/Llama-3.1-8B-Instruct
dtype: bfloat16
parameters:
  normalize: true
models:
  - model: meta-llama/Llama-3.1-8B-Instruct
  - model: yourorg/llama-3.1-8b-math-sft
    parameters: {{density: {d_math}, weight: {w_math}}}
  - model: yourorg/llama-3.1-8b-code-sft
    parameters: {{density: {d_code}, weight: {w_code}}}
"""

results = []
grid = itertools.product([0.3, 0.5, 0.7], [0.3, 0.5, 0.7], [0.5, 0.7], [0.5, 0.7])
for w_math, w_code, d_math, d_code in grid:
    out = Path(f"runs/m{w_math}_c{w_code}_dm{d_math}_dc{d_code}")
    Path("config.yml").write_text(CONFIG_TEMPLATE.format(**locals()))
    subprocess.check_call(["mergekit-yaml", "config.yml", str(out), "--cuda"])
    score = subprocess.check_output(["python", "eval.py", "--model", str(out)])
    results.append({"cfg": str(out), "score": json.loads(score)})

results.sort(key=lambda r: -r["score"]["avg"])
print(results[:5])
```

For larger sweeps, plug in CMA-ES (`cma` package) over the continuous knobs to reproduce evolutionary merging on a budget.

## See Also

- `../unsloth-sft/` — produce the specialist checkpoints you'll later merge.
- `../unsloth-advanced/` — multi-stage finetuning and DPO pipelines whose outputs are common merge candidates.
- `../distributed-grpo/` — RL alignment; the over-aligned outputs of GRPO/RLHF are a frequent input to *task negation* merges.
- `../../ml-architectures/mixture-of-experts/` — sparse upcycling and MoE construction from merged dense checkpoints.
- `../../ml-architectures/llm/` — base-model architectures that constrain which merges are possible.
- `../../ml-libraries/huggingface/` — `transformers`, `peft`, and the model formats mergekit reads/writes.

## References

- mergekit (de-facto merging tool): https://github.com/arcee-ai/mergekit
- Ilharco et al. 2022, "Editing Models with Task Arithmetic" (task vectors): https://arxiv.org/abs/2212.04089
- Yadav et al. 2023, "TIES-Merging: Resolving Interference When Merging Models": https://arxiv.org/abs/2306.01708
- Yu et al. 2023, "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch" (DARE): https://arxiv.org/abs/2311.03099
- Wortsman et al. 2022, "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time": https://arxiv.org/abs/2203.05482
- Akiba et al. 2024 (Sakana AI), "Evolutionary Optimization of Model Merging Recipes": https://arxiv.org/abs/2403.13187
- Maxime Labonne, "Merge Large Language Models with mergekit" (HF blog): https://huggingface.co/blog/mlabonne/merge-models
- PEFT — merge LoRA weights into base: https://huggingface.co/docs/peft/main/en/developer_guides/lora#merge-lora-weights-into-the-base-model
