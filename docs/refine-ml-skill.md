# Refine ML Skill — Maintainer Guideline

> **Status**: Repo guideline for maintainers of the `ml-skills` plugin. Not a runtime skill — read this when a topic refresh spans multiple files.

Deep-research a topic, then propagate the findings across every skill it touches — keeping cross-file claims consistent and links fresh.

## Why This Exists

**Problem**: ML evolves fast. A change in one area (a new attention variant, a deprecated API, a faster training recipe) almost always touches several skills. [`acquire-ml-skill.md`](acquire-ml-skill.md) handles one file at a time; nothing in the library coordinates a multi-file refresh, so skills drift out of sync (LLM talks about MHA while attention/ has switched to MLA; vllm/ recommends an old quant flag that quantization/ no longer mentions).

**Key insight**: Topic-level research first, file-level edits second. Gather one authoritative picture from primary sources, then sweep every skill that should reflect it — so all references to the same concept tell the same story.

**Reach for this when**:
- A topic genuinely spans 2+ skill folders (attention variants, quantization formats, distributed training APIs, serving stacks).
- An existing skill is suspected stale and a fresh source-of-truth is needed before editing.
- The user asks to "audit" or "refresh" coverage of an area.
- After a major library release (PyTorch X, vLLM Y, HF transformers Z) where APIs may have changed.

For "add one new skill" or "fix one typo", use [`acquire-ml-skill.md`](acquire-ml-skill.md) instead.

## Critical Thinking, Not Template-Filling

The phases below are a scaffold, not a recipe. The hard parts are the judgment calls inside each phase:

- **Intake (Phase 1)**: Which questions actually unblock progress, and which are nice-to-have? Asking too many turns this into a survey; asking too few means re-asking later. Calibrate.
- **Blast radius (Phase 3)**: `grep` finds keyword matches, but the writer has to decide which matches are *the same concept* (a renamed API still appearing under its old name) versus *false positives* (the word "attention" used in a non-technical sentence).
- **Canonical home (Phase 5)**: When three folders all reasonably cover a topic, only one should hold the deep treatment. Which one? Pick by where future readers will most naturally look first, not by alphabetical order.
- **What "consistency" means (Phase 6)**: It's not search-and-replace. Two skills can use different framings of the same fact for different audiences — the rule is they must not *contradict*, not that they must be identical.
- **When to stop (Phase 8)**: If research keeps surfacing more affected skills, that's a signal the topic deserves its own folder — hand off to [`acquire-ml-skill.md`](acquire-ml-skill.md) instead of widening this update indefinitely.

If any phase feels like rote work, you're missing the decision that should be happening there.

## Workflow

### Phase 1 — Intake (do this BEFORE research)

A one-line request like "refine the attention skill" is never enough. The blast radius, the canonical home, and the depth of the rewrite all depend on answers the user has but hasn't volunteered. Ask the smallest set of questions that resolves real ambiguity — usually 3-5. Skip any whose answer is already in the request.

Required intake before proceeding to Phase 2:

| Question | Why it matters |
|----------|---------------|
| **Topic, pinned to a version or release if applicable?** | "FlashAttention" is too broad; "FlashAttention-3 (Aug 2024 paper)" is what you actually research. Versions matter — APIs deprecated in PyTorch 2.5 are still recommended in older docs. |
| **What triggered this refresh?** | New release, suspected staleness, paper drop, audit, broken link report. The trigger shapes what counts as "done" — a release-driven refresh ends when API claims match the release; a staleness audit ends when every reference is verified. |
| **Scope hint — folders the user already knows are affected?** | Saves a grep round and catches false negatives the writer might miss. Treat as a starting set, not a complete one. |
| **Depth: rewrite, extend, or just fix specific claims?** | A rewrite reorganizes sections and decision tables; an extend adds material without touching what's there; a fix changes only the wrong sentences. Wrong depth choice = either bloated diffs or unfinished work. |
| **Canonical home preference, if multiple folders fit?** | If the user has an opinion (e.g. "the deep treatment lives in attention/, llm/ links to it"), capture it now — Phase 4 just enforces it. Without a preference, you'll have to argue for one in Phase 4. |
| **Anchor sources?** | Specific paper, repo, release notes, or doc the user wants treated as canonical. Bypasses the ranking problem inside `deep-research`. |
| **Out-of-scope items?** | What changes in adjacent skills should NOT happen as part of this refresh? Prevents scope creep when the grep returns surprises. |

Stop asking once you have enough to map the blast radius confidently. If the user says "you decide", record each assumption in the Phase 7 report so they can correct any wrong calls.

If Phase 3 or 4 surfaces a question the intake missed (e.g. "this topic actually deserves its own folder — ok to add?"), **come back here** rather than silently expanding scope.

### Phase 2 — Scope the topic

Synthesize the intake into a one-paragraph scope statement before any research happens:

> *"We're refreshing X (version Y) because Z. Canonical home will be `<folder>/`. Adjacent skills A, B, C may need updates. Out of scope: D, E."*

This becomes the spec for the rest of the workflow — and the lead paragraph of the Phase 7 report.

### Phase 3 — Map the blast radius

Before researching, find every skill that *currently* references the topic. Don't trust memory.

```bash
# From repo root
grep -ril "<topic-keyword>" skills/
grep -rl "<api-or-class-name>" skills/
```

List the matches in a short table:

| Skill path | Current claim about the topic |
|------------|-------------------------------|
| `skills/ml-architectures/attention/SKILL.md` | … |
| `skills/ml-libraries/vllm/SKILL.md` | … |

Also list **likely-but-missing** skills — folders that *should* mention the topic but don't. Those become add-or-extend targets.

### Phase 4 — Deep research

Invoke the `deep-research` skill with the refined topic as the question. Pass the scope hint so it knows the audience is an ML practitioner who already understands fundamentals.

What the research output must contain:
- The current authoritative description of the topic (one paragraph).
- Concrete API surface or formula — names, shapes, defaults.
- What changed recently (versions, deprecations, replacements).
- Tradeoffs vs. alternatives (decision-table material).
- 3+ verified primary sources: official docs, GitHub repo, seminal paper.

Verify every URL with `curl -sI <url> | head -1` before they go into any skill.

### Phase 5 — Decide the per-file plan

For each affected skill, decide one of:

| Action | When |
|--------|------|
| **Update in place** | The skill already covers the topic but says something now-wrong or outdated. |
| **Extend** | The skill mentions the topic in passing; promote it to a proper subsection with code + decision table. |
| **Add cross-reference** | The skill is adjacent and should link to the canonical home, not duplicate it. |
| **Leave alone** | The match was a false positive (different concept, same word). |
| **Create new sub-skill** | The topic deserves its own folder; hand off to [`acquire-ml-skill.md`](acquire-ml-skill.md) for the actual creation. |

Pick **one canonical home** for the topic — the folder where the deepest treatment lives. Other skills get short mentions + a link to the canonical home, never duplicated paragraphs.

### Phase 6 — Apply edits

For every file marked update/extend, satisfy the same quality standards as [`acquire-ml-skill.md`](acquire-ml-skill.md):
- `## Why This Exists` if missing.
- Decision table vs. alternatives.
- Code example (PyTorch + framework-agnostic where applicable).
- `## References` with verified links.

Cross-file consistency rules (the whole point of this guideline):
- The same concept must use the same name in every file (don't call it "MQA" in one and "Multi-Query Attention" in another without the abbreviation).
- The canonical home is linked from every other mention, with relative paths from the editing file (`../../ml-architectures/attention/`).
- Decision tables agree — if `attention/` says GQA is the default for Llama-3, `llm/` cannot still claim MHA.
- Version claims agree — if PyTorch 2.5 deprecates an API in one file, no other file should still recommend it.

### Phase 7 — Verify

Run before reporting done:

```bash
# All edited links return 200
for url in <list>; do curl -sI "$url" | head -1; done

# No stale references remain
grep -rn "<deprecated-name>" skills/

# Cross-references resolve
grep -rn "](\.\./" skills/ | <spot-check the relative paths>
```

If the router (`ml-router/SKILL.md`) needs a new row because a sub-skill was added or a problem-type mapping changed, edit it too.

### Phase 8 — Report

Summarize for the user, in this order:
1. What the topic now says, in two sentences.
2. Files touched, grouped by action (updated / extended / linked / unchanged).
3. Anything intentionally left for a follow-up (e.g. new sub-skill candidate worth filing via [`acquire-ml-skill.md`](acquire-ml-skill.md)).

## Anti-Patterns

- Editing files one at a time without first mapping the blast radius — leads to drift.
- Duplicating the topic's full treatment in every file — pick a canonical home, link the rest.
- Trusting memory for "what changed recently" — always pull primary sources via `deep-research`.
- Skipping link verification because "it's the official docs URL" — official URLs change too.
- Reaching for this guideline to add one new isolated skill — that's [`acquire-ml-skill.md`](acquire-ml-skill.md).
- Renaming or removing existing sections that aren't actually wrong, just to "tidy up" — preserve unrelated content.

## When to Hand Off

| Situation | Hand off to |
|-----------|-------------|
| The research surfaced a topic deserving a new sub-skill | [`acquire-ml-skill.md`](acquire-ml-skill.md) (create), then come back here to wire references |
| Just one file needs a small fix | [`acquire-ml-skill.md`](acquire-ml-skill.md) (update workflow) |
| The router itself is structurally wrong | edit `skills/ml-router/SKILL.md` directly |

## References

- [`acquire-ml-skill.md`](acquire-ml-skill.md) — quality standards every edited skill must satisfy.
- `skills/ml-router/SKILL.md` — the routing index; update if a sub-skill is added or its scope shifts.
- `deep-research` skill — invoke for primary-source gathering in Phase 4.
