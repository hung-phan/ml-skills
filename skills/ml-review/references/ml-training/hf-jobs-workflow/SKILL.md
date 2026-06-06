---
name: hf-jobs-workflow
description: Operational discipline for running training/fine-tuning on Hugging Face Jobs — pre-flight checklist, hardware sizing, OOM recovery without scope change, sandbox-first GPU smoke tests, push_to_hub, prebuilt attention kernels, batch-job validation. Use when launching ML jobs on managed cloud GPUs (HF Jobs or similar) and you want to avoid the failure modes that silently waste hours of compute.
---

# HF Jobs Workflow

## Why This Exists

**Problem**: Managed-GPU job services (HF Jobs, Modal, RunPod, SkyPilot, …) make it cheap to *launch* training runs and expensive to *land* them. The default failure modes are quiet and costly:

- Job runs for 6 hours, hits the default 30-minute timeout, gets killed — no checkpoint pushed.
- OOM at step 200 → engineer "creatively" switches SFT → LoRA, silently changing what the model learns.
- All 8 ablation jobs submitted at once, all fail for the same bug.
- `flash-attn` compiled from source for 40 minutes, fails on the worker's CUDA combo.
- Loss values hidden inside `tqdm` progress bars — unreadable in shipped logs.
- Filesystem deleted at job end → trained weights gone because `push_to_hub` was forgotten.

**Key insight**: Treat every cloud training run as a deployment, not a script. Pre-flight checklist + cheap GPU smoke test + one canary before fan-out catches 90% of the burns. The runtime is ephemeral; only what you push is real.

**Reach for this when**:
- Launching SFT / DPO / GRPO on HF Jobs, or any equivalent managed-GPU runner.
- You've been burned by jobs that "completed" but produced no usable artifact.
- Choosing GPU hardware sizing for a model you haven't trained before.
- An OOM is tempting you to change training method or sequence length — read this first.

For the *training algorithm itself* see [`unsloth-sft/`](../unsloth-sft/), [`unsloth-advanced/`](../unsloth-advanced/), [`distributed-grpo/`](../distributed-grpo/), or [`ray-distributed-sft/`](../ray-distributed-sft/). This skill is about the **operational wrapper** around those.

## The Pre-Flight Checklist

Before submitting any GPU training job, write down each item with a concrete value. If you can't fill one in, stop and resolve it first — don't submit half-checked.

```text
Reference implementation : <which example/repo this is based on>
Dataset format verified  : <columns confirmed match training method>
GPU sandbox smoke test   : <hardware tier + result, OR "n/a because …">
push_to_hub              : True
hub_model_id             : <user-or-org/model-name>
timeout                  : <hours, justified by model size × hardware>
Logging                  : disable_tqdm=True, logging_first_step=True
Monitoring               : <Trackio / W&B / MLflow run URL>
```

The discipline isn't bureaucracy — it's the difference between "training failed at step 0 because column was named `text` not `messages`" and "training succeeded because we caught it in a 2-minute smoke test."

## GPU Hardware Sizing

| Model size (params) | Minimum hardware | Notes |
|---------------------|------------------|-------|
| 1–3 B               | `a10g-largex2` (2× 24GB) | Single A10G OOMs on full SFT with reasonable seq len |
| 7–13 B              | `a100-large` (80GB) or `a10g-largex4` | A100-80GB fits 7B SFT comfortably; QLoRA fits on a single A10G |
| 30 B+               | `l40sx4` or `a100x4` | FSDP/DeepSpeed required; check effective batch math |
| 70 B+               | `a100x8`, `h200x4`+ | Plan for model-parallel; vanilla DDP is not enough |

**Watch for the same-GPU-different-RAM trap**: on HF Jobs, `a10g-small` and `a10g-large` have the *same* 24 GB GPU memory — the difference is host CPU/RAM, not VRAM. Picking "large" doesn't fix an OOM; you need `a10g-largex2` or higher.

Verify the live hardware catalog and pricing before pinning a flavor — the official Jobs reference (see [References](#references)) has the current list.

## OOM Recovery Without Changing the Task

When training OOMs, the *correct* hierarchy preserves what the user asked for:

1. **Reduce `per_device_train_batch_size`**, increase `gradient_accumulation_steps` proportionally.
   Effective batch = `per_device * grad_accum * num_devices` — keep this constant.
2. **Enable `gradient_checkpointing=True`** (trades ~30% compute for ~40% activation memory).
3. **Upgrade GPU tier**: a10g → a100-80GB → a100x4 → a100x8.
4. **Lower precision** if not already: bf16 over fp32 (no quality loss for most modern models).

Things that look like fixes but **change the experiment** — do NOT do these silently:

| Tempting "fix"                | Why it's wrong                                                                 |
|-------------------------------|--------------------------------------------------------------------------------|
| Switch SFT → LoRA / QLoRA     | LoRA trains a different parameter set; eval results are not comparable to SFT. |
| Reduce `max_length`           | Silently truncates training samples; the model learns from a different distribution. |
| Drop monitoring / eval        | You lose the very signal that would tell you the model converged.              |
| Switch to a smaller base model | Different model, different research question.                                  |

If the user's spec genuinely cannot fit on any reachable hardware, **stop and tell them**. Don't ship a quietly-different experiment.

## Sandbox-First Development

A single 5-minute CPU/GPU sandbox catches more bugs than any amount of code review. The pattern:

```text
write inline script
  → install deps in cheapest sandbox tier that fits
  → run with TINY dataset slice (e.g. take(32))
  → fix the inevitable column-name / import / dtype error
  → submit the real job at scale
```

Always run a GPU smoke test before submitting if any of these are true:
- Script loads a model onto CUDA
- Uses `bf16` / `fp16` / quantization
- Uses Flash-Attention or paged attention
- Calls `torch.compile`

CPU sandboxes can't validate any GPU-specific code path. Use the smallest GPU that fits (`t4-small` is often enough for a 1-step smoke). Sandbox filesystems do not survive session resumption — recreate everything you need before relying on prior state.

## The push_to_hub Discipline

Job storage is **ephemeral**. The instant the job container exits, every file you wrote is gone — unless it was pushed to the Hub or another durable store first. Forgetting `push_to_hub=True` is the single most expensive mistake possible.

```python
from trl import SFTConfig

config = SFTConfig(
    output_dir="./out",
    push_to_hub=True,                     # MANDATORY for cloud jobs
    hub_model_id="myuser/my-sft-run-001", # MANDATORY — must include username/org
    hub_strategy="every_save",            # push checkpoints, not just the final
    save_strategy="steps",
    save_steps=200,
)
```

Set `hub_strategy="every_save"` so a job that crashes at 90% complete still leaves a usable checkpoint behind. The token (`HF_TOKEN`) is auto-injected into HF Jobs secrets — don't pass it explicitly in the script.

## Training Logging That Survives the Logs Tab

`tqdm` progress bars look great in a notebook and are *useless* in piped job logs — the carriage-return updates render as a single ungrep-able line.

```python
from trl import SFTConfig

config = SFTConfig(
    disable_tqdm=True,           # plain-text per-step lines
    logging_strategy="steps",
    logging_steps=10,
    logging_first_step=True,     # see step 0 — confirms training actually started
    report_to="trackio",         # or "wandb" / "mlflow" / "tensorboard"
    run_name="sft_qwen3-4b_lr2e-5_bs128",  # descriptive — survives in the dashboard
)
```

The same applies to dataset progress bars in transformers / hub:

```python
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"]                  = "1"
os.environ["TRANSFORMERS_VERBOSITY"]        = "warning"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]     = "1"  # 5–10× faster Hub uploads
```

## Submit-One-Then-Batch

For ablations / hyperparameter sweeps:

1. Submit **one** job with the leftmost grid cell.
2. Wait for it to start training successfully (not just "queued") — confirm by reading the log line for step 1.
3. Only *then* submit the remaining N−1 jobs.

If the first job fails at step 0, all N would have failed for the same reason. The only thing fanning out earlier buys you is N× the bill.

## Dataset Format by Training Method

The single most common failure at step 0: dataset columns don't match what the trainer expects. Verify before submitting.

| Method | Required columns | Notes |
|--------|------------------|-------|
| SFT    | one of `messages`, `text`, or `prompt`+`completion` | `messages` must be a list of role/content dicts |
| DPO    | `prompt`, `chosen`, `rejected`                      | `chosen`/`rejected` are full assistant responses |
| GRPO   | `prompt` only                                       | rewards come from your reward function, not the dataset |
| ORPO   | same as DPO                                         | combined SFT+preference loss |

```python
from datasets import load_dataset
ds = load_dataset("org/dataset-name", split="train[:1%]")
print(ds.column_names)  # confirm before submitting the real job
print(ds[0])            # eyeball one row
```

## Prefer HF Hub Kernels Over Compiling Flash-Attn

Building `flash-attn` from source on a job worker is slow (often 30+ min), fragile (CUDA / PyTorch / glibc combos), and offers no upside over the prebuilt kernels Hugging Face already publishes.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    attn_implementation="kernels-community/flash-attn2",  # prebuilt, no compile
    dtype="bfloat16",
)
```

Other useful prebuilt kernels: `kernels-community/vllm-flash-attn3`, `kernels-community/paged-attention`. Browse the catalog with the URL in [References](#references). With TRL's CLI scripts you can pass `--attn_implementation kernels-community/flash-attn2` directly.

Only `pip install` an attention package when no Hub kernel covers the case, and document why in the job comment.

## Decision Table — When to Reach for HF Jobs vs. Alternatives

| Scenario                                                | Use                                | Not                                       |
|---------------------------------------------------------|------------------------------------|-------------------------------------------|
| Quick experiment, you have a local GPU                  | local + `unsloth-sft`              | HF Jobs (overhead doesn't pay back)       |
| Single training run > 4 h, no local GPU                 | HF Jobs                            | spot-on-laptop                            |
| Multi-GPU SFT (DeepSpeed/FSDP)                          | HF Jobs `a100x4`+ or `ray-distributed-sft` | single-GPU `unsloth-sft`         |
| GRPO at scale (>1 reward model)                         | `distributed-grpo` on HF Jobs      | single GPU                                |
| You need a custom container / system deps               | HF Jobs (custom Docker image) or Modal | bare HF Jobs default image            |
| Sandbox / debugging / "is this script even valid"       | HF Spaces sandbox or local         | HF Jobs (slow feedback loop)              |
| Hyperparameter sweep                                    | HF Jobs with submit-one-then-batch | local sequential                          |

Adjacent skills:
- [`unsloth-sft/`](../unsloth-sft/) — the trainer config you'd hand to a job.
- [`experiment-tracking/`](../experiment-tracking/) — Trackio / W&B / MLflow setup; this skill assumes one is wired in.
- [`data-parallel/`](../data-parallel/) — DDP / FSDP / DeepSpeed sizing, which dictates the GPU flavor.

## Common Failure Modes and Fixes

| Symptom                                              | Likely cause                                            | Fix                                                                |
|------------------------------------------------------|---------------------------------------------------------|--------------------------------------------------------------------|
| Job "completed" but no model on Hub                  | Missing `push_to_hub` or `hub_model_id`                 | Both must be set; container FS is ephemeral.                       |
| Job killed at exactly 30 minutes                     | Default timeout                                         | Set `timeout="6h"` (or whatever the run actually needs) per job.   |
| `KeyError: 'messages'` at step 0                     | Dataset column name mismatch                            | Print `ds.column_names` and adapt or remap.                        |
| `flash-attn` install hangs the job                   | Compiling from source on the wrong CUDA combo           | Use `attn_implementation="kernels-community/flash-attn2"`.         |
| Logs are unreadable, can't see loss                  | `tqdm` progress bars                                    | `disable_tqdm=True, logging_first_step=True`.                      |
| OOM only at evaluation, not training                 | Larger eval batch + grad-checkpointing off              | Lower `per_device_eval_batch_size`; or run eval less often.        |
| All 8 ablation jobs failed identically               | Submitted batch before validating                       | Submit-one-then-batch.                                             |
| Trained model loads but produces gibberish           | Tokenizer mismatch or chat template not applied         | Save the *tokenizer* alongside the model; verify chat template.    |

## References

- [Hugging Face Jobs — User Guide](https://huggingface.co/docs/huggingface_hub/guides/jobs) — official docs for the Jobs API
- [Hugging Face Jobs — Package Reference](https://huggingface.co/docs/huggingface_hub/package_reference/jobs) — `HfApi.run_job` / scheduled jobs
- [TRL `SFTTrainer` docs](https://huggingface.co/docs/trl/sft_trainer) — `SFTConfig` field reference incl. `push_to_hub`
- [TRL `DPOTrainer` docs](https://huggingface.co/docs/trl/dpo_trainer) — DPO dataset format
- [TRL `GRPOTrainer` docs](https://huggingface.co/docs/trl/grpo_trainer) — GRPO dataset format and reward function contract
- [Transformers `Trainer` docs](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) — `TrainingArguments` (logging, push_to_hub, report_to)
- [Transformers GPU inference guide](https://huggingface.co/docs/transformers/en/perf_infer_gpu_one) — `attn_implementation` options
- [`huggingface/kernels`](https://github.com/huggingface/kernels) — load prebuilt CUDA kernels from the Hub
- [Hub kernel catalog (filter by `kernel`)](https://huggingface.co/models?other=kernel) — browse `kernels-community/*`
- [`kernels-community` org on the Hub](https://huggingface.co/kernels-community) — flash-attn, paged-attention, vllm-flash-attn3
- [`huggingface/ml-intern`](https://github.com/huggingface/ml-intern) — reference implementation of an HF-Jobs-aware ML coding agent (the operational lessons here are distilled from its system prompt and tool layer)
