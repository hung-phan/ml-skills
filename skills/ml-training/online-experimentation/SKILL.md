---
name: online-experimentation
description: Online evaluation, A/B testing, and multi-armed bandits for ML model rollout — Welch's t-test, Benjamini-Hochberg, sample-ratio mismatch, peeking, novelty/carryover, CUPED variance reduction, sequential testing, Thompson sampling, contextual bandits. Use when shipping a new model to production, choosing between A/B and bandits, or diagnosing why an A/B test result doesn't replicate.
---

# Online Experimentation

## Why This Exists

**Problem**: Offline evaluation (`evaluation/`, `training-workflow/`) tells you a model is better on held-out data. It cannot tell you whether shipping it will improve the business metric you actually care about — distribution shift, novelty effects, downstream behavior, and metric tier mismatches all break the offline-to-online pipeline. Most "model improvements" that pass offline eval do nothing — or harm — in production.

**Key insight**: There are four metric tiers — **business KPI → measurable live metric → offline eval metric → training loss** — and adjacent tiers almost never align. Online experimentation is the discipline of measuring the top two tiers under uncertainty, with statistics that survive contact with the real distribution.

**Reach for this when**: about to ship a model to a fraction of traffic; deciding between A/B test, multi-armed bandit, shadow deploy, or interleaving; suspect a past A/B result was a false positive; designing rollout for a recommender, ranker, or pricing model; need to size a test before launching it.

---

## 1. The Four-Tier Metric Hierarchy

| Tier | Example (recommender) | Where it lives | Lag |
|------|-----------------------|----------------|-----|
| Business KPI | 30-day retention, GMV | OLAP / data warehouse | Days–months |
| Live measurable | Session length, CTR, conversion | Event pipeline | Minutes–hours |
| Offline eval | NDCG@10, AUC, hit-rate@k | `evaluation/` | Per training run |
| Training loss | Cross-entropy, BPR, NCE | Trainer logs | Per step |

The rule: **misalignment between adjacent tiers is the default, not the exception.** Loss can drop while AUC stays flat; AUC can climb while CTR doesn't move; CTR can go up while retention falls (clickbait pattern). Pick the *highest tier you can measure within the experiment window* as your decision metric, and track the others as guardrails.

```python
# Decision metric + guardrails pattern
decision_metric = "session_ctr"
guardrails = {
    "session_length_min": ("not_worse_by", 0.02),     # absolute drop limit
    "complaint_rate":      ("not_worse_by", 0.001),
    "p99_latency_ms":      ("not_worse_by", 5),
}
# Ship only if decision_metric improves AND no guardrail breach.
```

---

## 2. A/B Testing — The Pitfalls Checklist

Run through this list *before* the test, not after a surprising result. Each item is a real failure mode that has shipped wrong decisions at large companies.

| # | Pitfall | Symptom | Fix |
|---|---------|---------|-----|
| 1 | **Sample-ratio mismatch (SRM)** | Bucket sizes deviate from designed split (e.g. 49.2% / 50.8% expected 50/50) | χ² test on bucket counts daily; if p<0.001, kill the test — assignment is broken |
| 2 | **Peeking / early stopping** | Repeated significance checks inflate type-I error from 5% to 30%+ | Fix sample size *before* launch, OR use sequential testing (mSPRT, always-valid p-values) |
| 3 | **Novelty effect** | Treatment wins in week 1, fades by week 4 | Pre-register a 2-3 week window; compare last-week vs first-week effect |
| 4 | **Carryover** | Same user in both arms across sessions | User-level (not request-level) randomization; cluster on user_id |
| 5 | **Network / interference** | Treatment changes how control users behave (marketplace, social) | Cluster-randomized design (geography, time, social cluster) |
| 6 | **Welch vs Student t-test** | Equal-variance assumption fails when variances differ across arms | Always use Welch's t-test by default — same answer when variances match, correct when they don't |
| 7 | **Heavy-tailed metric** | One whale accounts for 20% of revenue; CLT slow | Bootstrap CI, or winsorize/cap, or analyze on log-revenue |
| 8 | **Multiple hypotheses** | Test 20 metrics → expect 1 false positive at α=0.05 | Benjamini-Hochberg FDR control on the secondary metric set |
| 9 | **Simpson's paradox** | Aggregate effect reverses inside segments | Pre-register segments (platform, country, new vs returning); look at segments before aggregate |
| 10 | **Triggered analysis bug** | Including users who never saw the change dilutes the effect | Analyze only triggered users; report both ITT and CACE |
| 11 | **p-value ≠ effect size** | p=0.03 with effect = +0.1% on a metric where you need +2% | Always report effect size + CI; pre-register MDE |
| 12 | **Underpowered test** | "No significant effect" reported on a test that couldn't detect a real one | Compute power before launch; for null result, report the upper CI as "ruled out effects > X" |

```python
# Welch's t-test + Benjamini-Hochberg + MDE / power
import numpy as np
from scipy import stats

def welch_ttest(control, treatment):
    """Two-sided Welch's t-test. Returns (effect, p_value, ci_low, ci_high)."""
    nc, nt = len(control), len(treatment)
    mc, mt = control.mean(), treatment.mean()
    vc, vt = control.var(ddof=1), treatment.var(ddof=1)
    se = np.sqrt(vc / nc + vt / nt)
    t = (mt - mc) / se
    df = (vc / nc + vt / nt) ** 2 / ((vc / nc) ** 2 / (nc - 1) + (vt / nt) ** 2 / (nt - 1))
    p = 2 * stats.t.sf(abs(t), df)
    ci = stats.t.ppf(0.975, df) * se
    return mt - mc, p, (mt - mc) - ci, (mt - mc) + ci

def benjamini_hochberg(p_values, fdr=0.05):
    """Return mask of which hypotheses to reject under BH FDR control."""
    p = np.asarray(p_values)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = fdr * np.arange(1, n + 1) / n
    below = ranked <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool)
    cutoff = np.max(np.where(below))
    reject = np.zeros(n, dtype=bool)
    reject[order[: cutoff + 1]] = True
    return reject

def required_n_per_arm(baseline_mean, baseline_std, mde_relative, alpha=0.05, power=0.8):
    """Sample size per arm to detect mde_relative * baseline_mean with given power."""
    delta = mde_relative * baseline_mean
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return int(np.ceil(2 * (baseline_std / delta) ** 2 * (z_a + z_b) ** 2))

# Example: detect +1% lift on a metric with mean=0.10, std=0.30
print(required_n_per_arm(0.10, 0.30, mde_relative=0.01))  # ~141k per arm
```

---

## 3. CUPED — Variance Reduction with Pre-Period Data

CUPED (Controlled-experiment Using Pre-Existing Data, Microsoft 2013) is the single highest-ROI online-experimentation trick. It uses a user's pre-experiment metric as a covariate; if the same user's behavior is autocorrelated (almost always true), variance drops 30-50%, equivalent to running the test ~2× as long for free.

```python
def cuped(y, x):
    """y = post-period metric, x = pre-period metric for same users.
    Returns CUPED-adjusted y; analyze it with normal Welch t-test."""
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean())

# Usage:
y_c_adj = cuped(y_control, x_control)
y_t_adj = cuped(y_treatment, x_treatment)
welch_ttest(y_c_adj, y_t_adj)  # tighter CI than raw Welch
```

CUPED is only valid when x is measured *before* the experiment, on the same users. It does not bias the estimate; it only reduces variance.

### Beyond CUPED — CUPAC and MLRATE

Linear CUPED uses one pre-period covariate. When a stronger predictor of the post-period metric exists (a model trained on pre-period features), replace `theta * x` with an ML prediction:

| Method | Idea | When |
|--------|------|------|
| **CUPED** (Microsoft 2013) | Linear adjustment using one pre-period metric | Default — always run if pre-period data exists |
| **CUPAC** (DoorDash 2020) | Replace linear covariate with a gradient-boosted prediction of the post-period metric | A pre-trained model can predict the metric better than its own pre-period value |
| **MLRATE / DR-CUPED** (Guo 2021, Jin & Ba 2022) | Cross-fit doubly-robust ML adjustment — unbiased even when the ML model is misspecified | Productionizing variance reduction at scale; want statistical guarantees |

Typical gains: CUPAC over CUPED is another 10-30% variance cut; MLRATE/DR-CUPED tighten further with cross-fitting. DoorDash, Netflix, and Meta run these in production. Sources: https://careers.doordash.com/blog/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/ · https://arxiv.org/abs/2106.07263 · https://arxiv.org/abs/2210.04660

### Production platforms

You usually shouldn't roll your own. Statsig, Eppo, GrowthBook, and Optimizely all ship CUPED + sequential testing + SRM checks out of the box, with warehouse-native (BigQuery/Snowflake) computation. GrowthBook is the open-source default. Reach for them before hand-coding the statistics: https://docs.statsig.com/stats-engine · https://docs.geteppo.com · https://docs.growthbook.io

---

## 4. Sequential Testing — Look Without Inflating α

Standard A/B testing forbids peeking. Sequential tests (mSPRT, always-valid confidence sequences) let you monitor continuously and stop early when the evidence is clear. Spotify, Netflix, and Optimizely all use these in production.

| Method | When to use | Library |
|--------|-------------|---------|
| **Group-sequential (Pocock, O'Brien-Fleming)** | Pre-set N looks at fixed times | `rpy2` + `gsDesign` |
| **mSPRT (mixture SPRT)** | Continuous monitoring, parametric | `confseq` |
| **Always-valid CIs (Howard 2021 + Waudby-Smith & Ramdas 2024)** | Any-time-valid, non-parametric; betting-based confidence sequences (Waudby-Smith & Ramdas, JRSS-B 2024) give tighter intervals than Hoeffding-style | `confseq` |
| **Anytime-valid regression-adjusted (Lindon et al. 2022, Netflix)** | Combines anytime-valid inference with regression adjustment (CUPED-style) — Netflix's production stack | Custom — see paper |

```python
# Always-valid confidence sequence (skeleton)
# Real implementation: github.com/gostevehoward/confseq
# Key property: P(∃ t : true_effect ∉ CI_t) ≤ α regardless of stopping rule.
```

If you cannot do sequential, the safe defaults are: **fix N before launch**, **never peek**, and reserve mid-flight stops only for guardrail breaches at α=0.001.

---

## 5. Multi-Armed Bandits — Different Question, Different Math

A/B and bandit answer different questions:

| Framework | Asks | Optimizes | Use when |
|-----------|------|-----------|----------|
| **A/B test** | Which arm is best? | Statistical confidence in winner | Decision is one-shot, lift matters more than regret, need post-hoc analysis |
| **Bandit** | How do I maximize cumulative reward while learning? | Cumulative regret | Lots of arms, short reward horizon, ongoing exploration acceptable |

If you need to *make a decision and ship one variant*, A/B. If you're routing live traffic and reward each session (recommender slate, headline test, ad creative), bandit.

### Algorithms

| Algorithm | Idea | When |
|-----------|------|------|
| **ε-greedy** | Pick best arm with prob 1−ε, random arm with prob ε | Baseline; ε=0.1 default |
| **UCB1** | Pick arm with highest mean + exploration bonus √(2 ln t / n_arm) | Stationary, finite arms |
| **Thompson sampling** | Sample arm by posterior probability of being best | Bayesian, robust default for Bernoulli rewards |
| **Contextual (LinUCB, Neural)** | Use per-request features to choose arm | Personalization, recsys reranking |
| **Off-policy DR estimators (IPS / SNIPS / DR)** | Evaluate a *new* policy from logs collected under a *different* policy; doubly-robust corrects for both reward model and propensity errors | Comparing a candidate bandit / ranker against a logged production baseline without running it live |

```python
# Thompson sampling for K-armed Bernoulli bandit
class ThompsonBernoulli:
    def __init__(self, k, prior_a=1.0, prior_b=1.0):
        self.alpha = np.full(k, prior_a)
        self.beta  = np.full(k, prior_b)

    def select(self):
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm, reward):  # reward in {0,1}
        self.alpha[arm] += reward
        self.beta[arm]  += 1 - reward

bandit = ThompsonBernoulli(k=4)
for _ in range(10_000):
    a = bandit.select()
    r = np.random.binomial(1, true_p[a])  # observe reward
    bandit.update(a, r)
```

Bandit gotchas: **non-stationarity** (rewards drift — use sliding window or discount), **delayed reward** (reward arrives hours later, must batch updates), **off-policy evaluation** when comparing to a logged baseline policy — use the **Open Bandit Pipeline** (Saito et al. 2021, https://github.com/st-tech/zr-obp) for IPS/SNIPS/DR estimators with proper variance bounds, or `vowpalwabbit --cb_explore_adf` with cover/bag/squarecb explorers for production training. `mab2rec` (https://github.com/fidelity/mab2rec) wraps these for recsys.

---

## 6. Pre-Launch Rollout Patterns

| Pattern | What it tests | When |
|---------|---------------|------|
| **Shadow deploy** | Compute treatment predictions but don't serve them; compare distributions | New model, want to verify infra/latency/calibration before any user impact |
| **Canary (1%, 5%, 25%, 50%, 100%)** | Operational health (errors, latency); not statistical lift | Every rollout — catch the model that returns 500s in prod |
| **A/B test** | Causal effect on metric | Decision metric requires statistical certainty |
| **Interleaving** | Per-request mixed ranking from both models, count clicks | Search/ranking — 10× more sensitive than user-level A/B |
| **Holdout** | Long-running 1-5% never gets new features | Measure aggregate value of all shipped changes over a quarter |
| **Switchback** | Alternate treatment/control over time windows | Marketplaces with strong network effects (rideshare, ads) |
| **GeoLift / synthetic control** | Synthetic-control inference at the geo / channel level; no user-level randomization required | Marketing channels, brand campaigns, TV / OOH, B2B accounts — anywhere user-level A/B is impossible. Tools: Meta GeoLift (https://github.com/facebookincubator/GeoLift), Google CausalImpact (https://google.github.io/CausalImpact/) |

Run them in series: shadow → canary → A/B → holdout. Each catches a different failure mode.

---

## When to Use What

| Scenario | Tool |
|----------|------|
| Single launch, definitive yes/no decision | A/B with Welch + fixed-N power calc |
| Many secondary metrics flagged | Welch + Benjamini-Hochberg |
| Same-user pre-period data exists | A/B + CUPED (always — free variance reduction) |
| Want to monitor continuously without inflating α | Sequential testing (mSPRT or always-valid CIs) |
| Many arms, ongoing reward, exploration acceptable | Thompson sampling / UCB |
| Per-request features matter | Contextual bandit (LinUCB / neural) |
| New model, infra unverified | Shadow deploy first |
| Search / ranking model | Interleaving |
| Strong network effects | Switchback or cluster-randomized |
| Long-term aggregate value | Quarterly holdout |
| Heavy-tailed revenue metric | Bootstrap CI or analyze log/winsorized |

---

## Common Gotchas

1. **No SRM check** → silent assignment bug invalidates everything. χ² on bucket sizes daily.
2. **Peeking with fixed-N test** → α inflates from 5% to 20%+. Use sequential or commit to N.
3. **Reporting p-value without effect size** → "significant +0.05% lift" is meaningless when MDE was 1%.
4. **Triggered users diluted with non-triggered** → effect estimate biased toward zero.
5. **CUPED on post-experiment covariate** → biases the estimate. Covariate must be strictly pre-period.
6. **Bandit on delayed reward as if instantaneous** → systematically over-explores recently-shown arms.
7. **A/B with strong network effects** → SUTVA violated; use cluster or switchback.
8. **Comparing bandit to A/B winner** → different objectives; bandit minimizes regret, not type-I/II error.
9. **One-week window for retention metric** → metric simply doesn't exist yet; use leading indicator + holdout.
10. **Re-running a "failed" test next week** → multiple-comparison problem; pre-register.

---

## References

- Kohavi, Tang, Xu — *Trustworthy Online Controlled Experiments* (2020): https://experimentguide.com/ — the canonical practitioner reference.
- Deng et al. (2013) — CUPED variance reduction: https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf
- Howard, Ramdas, McAuliffe, Sekhon (2021) — Time-uniform, nonparametric, nonasymptotic confidence sequences: https://arxiv.org/abs/1810.08240
- Waudby-Smith & Ramdas (JRSS-B 2024) — *Estimating means of bounded random variables by betting* (tighter confidence sequences): https://arxiv.org/abs/2010.09686
- Lindon, Malek et al. (2022) — *Anytime-Valid Linear Models and Regression Adjusted Causal Inference* (Netflix production): https://arxiv.org/abs/2210.08589
- Bojinov, Simchi-Levi, Zhao (2023) — Anytime-valid switchback / panel designs: https://arxiv.org/abs/2208.14197
- Johari, Pekelis, Walsh (2017) — Always-Valid Inference (mSPRT): https://arxiv.org/abs/1512.04922
- Lattimore & Szepesvári — *Bandit Algorithms* (2020), free PDF: https://tor-lattimore.com/downloads/book/book.pdf
- Russo et al. (2018) — *A Tutorial on Thompson Sampling*: https://arxiv.org/abs/1707.02038
- Bottou et al. (2013) — Counterfactual Reasoning and Learning Systems: https://arxiv.org/abs/1209.2355
- `confseq` (always-valid CIs): https://github.com/gostevehoward/confseq
- Fabijan, Gupchup et al. (KDD 2019) — *Diagnosing Sample Ratio Mismatch in Online Controlled Experiments*: https://www.kdd.org/kdd2019/accepted-papers/view/sample-ratio-mismatch-a-noise-trap-and-srm-checks-the-essential-ab-testin
- Zheng (2015) — *Evaluating Machine Learning Models* (the four-tier metric framing originated here): https://www.oreilly.com/library/view/evaluating-machine-learning/9781492048756/

## See Also

- [`../evaluation/`](../evaluation/) — offline metrics, calibration, statistical model comparison.
- [`../training-workflow/`](../training-workflow/) — nested CV for cross-family model comparison (offline analog of A/B power analysis).
- [`../experiment-tracking/`](../experiment-tracking/) — MLflow / W&B for the offline side; this skill covers the live side.
- [`../../data-prep/data-validation/`](../../data-prep/data-validation/) — drift detection that sits alongside online metric monitoring.
