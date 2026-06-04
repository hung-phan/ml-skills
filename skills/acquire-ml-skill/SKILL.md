---
name: acquire-ml-skill
description: Create a new skill or make a light update to one existing skill in the ml-skills library. Guides research, writing, formatting, and quality checks for single-file work. Use when user says "add a skill", "create a skill for X", "update the skill for X", "fix the X skill", or "improve coverage of Y" AND the change is scoped to one file. For topics that span multiple skills (cross-cutting refresh, post-release audit, deep-research with propagation), use refine-ml-skill instead.
---

# Acquire ML Skill

Create a new skill or update one existing skill in the `ml-skills` plugin library.

## When to Use This vs. refine-ml-skill

| Situation | Use |
|-----------|-----|
| New topic, no existing coverage | `acquire-ml-skill` |
| One file is wrong / outdated, fix is local | `acquire-ml-skill` |
| Topic spans 2+ folders (e.g. attention variants touch attention/, llm/, vllm/) | `refine-ml-skill` |
| Need primary-source research before writing | `refine-ml-skill` (Phase 3 calls deep-research) |
| Major library release, want to audit coverage | `refine-ml-skill` |

If you start in `acquire-ml-skill` and discover the change actually touches multiple files, **stop and switch to `refine-ml-skill`** — don't try to coordinate cross-file updates from here.

## Phase 0 — Intake (do this BEFORE research)

Don't start researching or writing on a one-line request. Most quality problems trace back to skipping this step. Ask the user the smallest set of questions that resolve real ambiguity — usually 2-4, never more. Skip any question whose answer is already obvious from the request.

Required intake before proceeding:

| Question | Why it matters |
|----------|---------------|
| **What's the topic, in one sentence?** | Forces the user to commit to a scope. "Add transformer" is not a scope; "Add a skill on rotary positional embeddings (RoPE) for decoder-only LLMs" is. |
| **Add new, or update existing?** | If updating, which file path? If adding, what's the proposed folder placement? |
| **What's the audience problem?** | What will the reader be trying to do when they reach this skill? "Pick between RoPE and ALiBi" is a different skill than "Implement RoPE in PyTorch". |
| **Frameworks in scope?** | PyTorch only? sklearn too? Framework-agnostic pseudocode? Default is PyTorch + one classical alternative where applicable — confirm or override. |
| **Anchor sources?** | Any specific paper, repo, or doc the user wants treated as canonical? Avoids ranking-roulette during research. |
| **Out-of-scope items?** | What should this skill explicitly NOT cover? (e.g. "skip training recipes — that's a separate skill"). Prevents scope creep mid-write. |
| **Existing skills to cross-reference?** | If the user already knows the related neighbors, capture them now — they become "See also" links instead of duplicated content. |

Stop asking once the answers are enough to start drafting. If the user says "you decide", record the assumption explicitly in the skill's first draft and flag it in your reply so they can correct it before merge.

If you discover during research that the answers were wrong, **come back and re-ask** rather than silently changing scope.

## Critical Thinking, Not Template-Filling

This is a judgment skill, not a form. Every step below has a decision behind it that the writer has to make:

- **Placement**: Which folder? If two folders both fit, which one is the canonical home and which gets a one-line cross-reference?
- **Trigger phrases in `description:`**: What would a user actually type to need this? Generic descriptions never get matched.
- **`Why This Exists`**: What problem does this solve that the next-best alternative doesn't? If you can't articulate the alternative, you don't yet understand the topic well enough to write the skill.
- **Decision table**: Where does this fail? When is it the wrong choice? A skill that only says when to use itself is half-written.
- **References**: Are these primary sources, or is this a tutorial blog quoting a tutorial blog?

If any of these answers come out as "I don't know", stop and research — don't write through the gap.

## Repository Structure

Skills live under the `skills/` directory. Each skill is a folder containing a single `SKILL.md` file.

```
skills/
├── ml-router/                   (top-level router — entry point for any ML/DL task)
├── acquire-ml-skill/            (this skill — single-file contributor meta-skill)
├── refine-ml-skill/             (deep-research + cross-repo propagation meta-skill)
├── ml-architectures/            (ANN, CNN, RNN, Transformer, Attention, MoE, Mamba,
│                                 GAN, Diffusion, GNN, LLM, Vision, RL, Autoencoder,
│                                 Boltzmann, Quantization, Embeddings, Regression/Classification)
├── ml-libraries/                (PyTorch, HuggingFace, scikit-learn, XGBoost, pandas,
│                                 polars, numpy, Ray, DSPy, LiteLLM, vLLM, SGLang,
│                                 Triton Inference Server, keras, seaborn, plotly)
├── ml-training/                 (feature-selection, training-workflow, evaluation,
│                                 data-parallel, unsloth-sft, unsloth-advanced,
│                                 ray-distributed-sft, distributed-grpo, experiment-tracking)
├── data-prep/                   (eda, feature-engineering, data-validation)
└── gpu-lang/                    (triton, tilelang)
```

To find the plugin root from any machine: start from the directory containing this file and go up two levels (`skills/acquire-ml-skill/SKILL.md` → `skills/` → repo root).

## Quality Standards

Every skill MUST satisfy all of the following. Use this as a checklist before submitting.

### 1. Problem-First (most important)

Lead with **why this exists**, not code. The reader should understand the problem before seeing any implementation.

```markdown
## Why This Exists

**Problem**: [What breaks without this? What pain does this solve?]

**Key insight**: [The core idea in plain English — one sentence]

**Reach for this when**: [Decision criteria vs alternatives]
```

### 2. SKILL.md Format

```markdown
---
name: skill-name
description: What it does. Use when [specific triggers].
---

# Title

## Why This Exists
[problem/insight/when-to-use]

## [Content sections with code examples]

## References
[verified links to docs, papers, repos]
```

### 3. Real, Verified Links

Every skill MUST have a `## References` section. Every URL must return HTTP 200:

```bash
curl -sI "URL" | head -1
```

Include at minimum:
- Official documentation URL
- GitHub repository
- Seminal paper (arXiv or conference link) if one exists

Common link patterns:
| Source | URL Pattern |
|--------|-------------|
| PyTorch docs | https://pytorch.org/docs/stable/nn.html |
| sklearn docs | https://scikit-learn.org/stable/modules/... |
| HuggingFace | https://huggingface.co/docs/... |
| Ray docs | https://docs.ray.io/en/latest/ |
| arXiv papers | https://arxiv.org/abs/XXXX.XXXXX |
| GitHub repos | https://github.com/org/repo |

### 4. Diverse Examples

Don't be sklearn-only or PyTorch-only. Include:
- **PyTorch** examples for deep learning workflows
- **sklearn** for classical ML
- **Framework-agnostic** pseudocode for architectural concepts

### 5. Decision Table

Every skill must help the reader choose. At minimum one table of the form:

```markdown
| Scenario | Use This | Not That |
|----------|----------|----------|
| Short sequences (<500 tokens) | RNN | Transformer (overkill) |
| Long sequences | Transformer | RNN (vanishing gradients) |
```

### 6. Index Files

Each folder has its own `SKILL.md` that lists children in a table. Keep index files as concise routers — no duplicating content from child skills.

---

## Workflow: Adding a New Skill

1. **Decide placement**: Which folder does it belong in? If none fit, propose a new folder.
2. **Research**: Gather from official docs + GitHub repo + seminal paper. Use web search or spawn agents for this.
3. **Write `skills/<folder>/<topic>/SKILL.md`** satisfying all quality standards above:
   - [ ] YAML frontmatter with `name` and `description` (include trigger phrases in description)
   - [ ] `## Why This Exists` (Problem / Key insight / Reach for this when)
   - [ ] Code examples (PyTorch + sklearn/framework-agnostic where applicable)
   - [ ] Decision table vs alternatives
   - [ ] `## References` with verified links
4. **Update parent index**: Add a row to the folder's `SKILL.md` table
5. **Verify links**: `curl -sI` each URL in the References section

## Workflow: Updating an Existing Skill

1. Read the current file in full first.
2. Identify which quality standards are missing (use checklist above).
3. Common gaps:
   - Missing `## Why This Exists` → add it
   - Only one framework → add the other
   - No `## References` or broken links → research and fix
   - Outdated API → check latest docs and update
   - No decision table → add one
4. Add missing sections without removing existing content unless explicitly replacing stale info.

## Anti-Patterns (Don't Do)

- ❌ Code dumps without explaining what problem they solve
- ❌ Only one framework (sklearn OR pytorch — include both where relevant)
- ❌ Links without verifying they return 200
- ❌ Duplicate content between parent index and child skill
- ❌ Frontmatter `description` without trigger phrases ("Use when...")
- ❌ Missing `## Why This Exists` section
- ❌ Hardcoded absolute paths (use paths relative to the repo root)
