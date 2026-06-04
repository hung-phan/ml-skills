---
name: acquire-ml-skill
description: Create or update skills in the ml-skills plugin library. Guides research, writing, formatting, and quality checks. Use when user says "add a skill", "create a skill for X", "update the skill for X", "research Y and add it", or "improve the ML skills".
---

# Acquire ML Skill

Create new skills or update existing ones in the `ml-skills` plugin library.

## Repository Structure

Skills live under the `skills/` directory. Each skill is a folder containing a single `SKILL.md` file.

```
skills/
├── SKILL.md                     (root index — router, do not add content here)
├── acquire-ml-skill/            (this skill — contributor meta-skill)
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
- ❌ `README.md` inside the `skills/` tree — all skill files are named `SKILL.md`
- ❌ Frontmatter `description` without trigger phrases ("Use when...")
- ❌ Missing `## Why This Exists` section
- ❌ Hardcoded absolute paths (use paths relative to the repo root)
