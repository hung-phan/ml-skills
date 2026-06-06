---
name: online-learning
description: Online / incremental learning and concept-drift handling — partial_fit, river, warm-start, train/score separation, model-as-data control-stream pattern, drift detection (PSI, KS, ADWIN, DDM, Page-Hinkley), event-time vs processing-time windows, brief feature-store framing. Use when models must adapt to streaming data, monitor for drift in production, retrain incrementally without full pipeline re-runs, or hot-swap a model without redeploying the serving system.
---

# Online Learning and Concept Drift

## Why This Exists

**Problem**: Batch ML assumes the world stops while you train. In production, the data distribution drifts (covariates change), the label distribution drifts (priors shift), the input/output relationship drifts (concept drift), and the model decays. Re-training the whole pipeline nightly is wasteful at best, too slow at worst — fraud, recsys, ad CTR, and IoT monitoring all need *adaptive* models. The repo has solid batch coverage; nothing addresses the live-update side.

**Key insight**: Three orthogonal problems get conflated. **Drift detection** (when did the world change?), **incremental learning** (how do I update the model from one mini-batch?), and **deployment plumbing** (how do new parameters reach the serving box without restarting?). Solve them separately, with different tools.

**Reach for this when**: deploying a model that needs to update faster than nightly retrain; production model accuracy is mysteriously decaying; serving system needs to swap weights without restart; have a streaming event source (Kafka, Kinesis) and want features computed on event-time, not arrival-time; need to choose between full retrain, warm-start, partial_fit, and a true online algorithm.

---

## 1. Drift Taxonomy — Three Things, Three Detectors

| Drift type | Definition | Detector | Where to monitor |
|------------|------------|----------|-------------------|
| **Covariate drift** | P(X) changes; P(Y\|X) does not | PSI / KS test on each feature | Input pipeline |
| **Label / prior drift** | P(Y) changes; P(X\|Y) does not | χ² on prediction or label distribution | Output / labels |
| **Concept drift** | P(Y\|X) changes (the underlying rule shifts) | Performance-based: ADWIN, DDM, Page-Hinkley on rolling error | Live evaluation |

You cannot detect concept drift without ground-truth labels. If labels arrive late (days), you must run **proxy detectors** (covariate + prior drift) early and treat them as warning signals, not proof.

```python
import numpy as np
from scipy import stats

def psi(expected, actual, bins=10):
    """Population Stability Index. Common rule of thumb:
    < 0.1 = no drift, 0.1-0.25 = moderate, > 0.25 = significant drift."""
    edges = np.histogram_bin_edges(expected, bins=bins)
    e_pct = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a_pct = np.histogram(actual,   edges)[0] / len(actual)   + 1e-6
    return float(((a_pct - e_pct) * np.log(a_pct / e_pct)).sum())

def ks_drift(expected, actual, alpha=0.01):
    """Kolmogorov-Smirnov; robust, distribution-free. Returns (drifted, p_value)."""
    stat, p = stats.ks_2samp(expected, actual)
    return p < alpha, float(p)

# Usage in monitoring loop
for col in feature_cols:
    if psi(reference[col], live_window[col]) > 0.25:
        alert(f"PSI drift on {col}")
    drifted, p = ks_drift(reference[col], live_window[col])
    if drifted:
        alert(f"KS drift on {col}, p={p:.4g}")
```

### Performance-based drift (when labels arrive)

```python
# ADWIN — adaptive sliding window. Shrinks when a change is detected.
# DDM (Drift Detection Method) — flags warning + drift on rolling error rate.
# Page-Hinkley — sequential change-point on a running mean.

# All three live in `river` and `frouros`:
from river import drift

adwin = drift.ADWIN(delta=0.002)
ddm   = drift.DDM(min_num_instances=30)
ph    = drift.PageHinkley(min_instances=30, threshold=50)

for x, y_true in stream:
    y_pred = model.predict_one(x)
    err    = int(y_pred != y_true)
    for det in (adwin, ddm, ph):
        det.update(err)
        if det.drift_detected:
            alert(f"{type(det).__name__} signaled drift")
```

**Library landscape**: [`river`](https://riverml.xyz/) is the modern incremental-ML library; [`frouros`](https://frouros.readthedocs.io/) and [`alibi-detect`](https://docs.seldon.io/projects/alibi-detect/) focus on drift detection; [`evidently`](https://docs.evidentlyai.com/) is a higher-level monitoring + dashboard tool.

---

## 2. Incremental Learning Strategies

Four levels, increasing in sophistication. Pick the simplest that meets your latency budget.

| Strategy | Mechanism | Cost | When |
|----------|-----------|------|------|
| **Periodic full retrain** | Snapshot data, train from scratch | High compute, lowest engineering risk | Daily/weekly cadence is fast enough |
| **Warm-start** | Load previous weights, train a few epochs on new data | Medium | Compute-bound but distribution drifts slowly |
| **`partial_fit` / mini-batch SGD** | Pass new mini-batches through the same model | Low | sklearn linear, NB, SGD; simple but no concept-drift handling |
| **True online (river)** | Per-record update with built-in drift adaptation | Lowest | High-velocity stream, drift expected |

```python
# sklearn partial_fit — works on linear models, SGDClassifier, MLPClassifier, GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
clf    = SGDClassifier(loss="log_loss", learning_rate="adaptive", eta0=0.01)

# First batch — fit scaler params and seed the classifier
X0, y0 = next_batch()
scaler.partial_fit(X0)
clf.partial_fit(scaler.transform(X0), y0, classes=np.array([0, 1]))

# Subsequent batches — keep updating
for X, y in stream_batches():
    scaler.partial_fit(X)              # Welford-style running stats
    clf.partial_fit(scaler.transform(X), y)
```

```python
# river — true online, one-record-at-a-time, drift-aware
from river import compose, preprocessing, linear_model, metrics

model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression(),
)
metric = metrics.LogLoss()

for x, y in stream:                    # x: dict, y: int
    y_pred = model.predict_proba_one(x)
    metric.update(y, y_pred)
    model.learn_one(x, y)              # in-place update, O(d)
```

**Catastrophic forgetting** is the failure mode: feed only fraud examples for an hour and the model forgets the negatives. Mitigations: keep a **reservoir sample** of older data and replay it; **regularize toward old weights** (L2 to prior); **train on a weighted mix** of streaming + historical batches.

For *deep* networks the modern continual-learning toolkit is broader: replay (iCaRL, GEM / A-GEM), regularization (EWC, SI, MAS), functional distillation (LwF, LwM), and parameter isolation (PackNet, HAT). See Wang et al. 2024 (TPAMI) https://www.computer.org/csdl/journal/tp/2024/08/10444954/1Vc1zg11leQ and van de Ven 2024 https://arxiv.org/abs/2403.05175 for canonical surveys. This skill stays focused on *classical-ML* online learning; deep-CL is a sibling topic worth its own skill if your LLM/vision model needs continual adaptation.

> **Note:** `river` 0.23 (Sep 2025) requires Python ≥3.11. https://riverml.xyz/dev/releases/0.23.0/

---

## 3. Train / Score Separation

The critical architectural rule: **the trainer and the scorer are different processes, possibly on different hardware, communicating via a parameter channel.**

```
        ┌─ raw events ─┐
        │              │
  ┌─────▼───┐     ┌────▼─────┐
  │ trainer │     │  scorer  │  ◀── prediction requests
  └────┬────┘     └────▲─────┘
       │               │
       └─→ params ─────┘
        (control stream / model registry)
```

Why separate:
- **Different latency budgets** — trainer is throughput-bound (seconds/minutes per update), scorer is latency-bound (single-digit ms).
- **Different hardware** — trainer wants GPU/large memory; scorer wants many small replicas.
- **Independent scaling** — N scoring replicas, 1 trainer.
- **Hot-swap without restart** — new params arrive on the channel; scorer atomically swaps in-memory.
- **Rollback is just sending the previous params** — no redeploy.

The **model-as-data control stream** pattern (Lublinsky 2017) ships serialized model parameters down a Kafka/SNS topic; every scorer replica subscribes and swaps when a new message arrives. Pass-by-value (inline bytes) for small models; pass-by-reference (S3 URI) for large ones.

```python
# Sketch — control-stream consumer in the scorer
from kafka import KafkaConsumer
import threading, joblib, io

current = {"model": load_initial_model()}

def reload_loop():
    consumer = KafkaConsumer("model-updates", group_id="scorer-1",
                             auto_offset_reset="latest")
    for msg in consumer:
        if msg.value.startswith(b"s3://"):
            blob = s3_get(msg.value.decode())  # pass-by-reference
        else:
            blob = msg.value                    # pass-by-value
        new_model = joblib.load(io.BytesIO(blob))
        current["model"] = new_model            # atomic dict assignment
        log.info("model swapped")

threading.Thread(target=reload_loop, daemon=True).start()

def score(x):
    return current["model"].predict_proba(x)[:, 1]
```

In 2024-2025 practice, the registry (MLflow / SageMaker / W&B / Vertex) is the canonical *governance plane* — versioning, staging, approval, lineage. The Lublinsky 2017 control-stream pattern remains useful as a *low-latency push channel* layered on top: registry webhook fires → message goes on Kafka → replicas swap in <1s. Use a registry alone when your hot-swap budget is tens of seconds; layer a stream on top for sub-second swap (recsys bandit weights, fraud thresholds). The two are complementary, not alternatives.

---

## 4. Event Time vs Processing Time

A streaming feature like "events in the last 5 minutes" has two clocks:
- **Event time** — when the event actually happened (the timestamp on the record).
- **Processing time** — when your pipeline observed it.

Network and queue lag can make these differ by seconds to hours. Computing windows on processing time is wrong: a record from 30 minutes ago shouldn't land in the *current* 5-minute window.

The Beam / Flink answer: window on event time, use **watermarks** to estimate "we believe we have seen all events with event_time ≤ t" and **triggers** to decide when to emit results given still-arriving late data. Akidau's "Streaming 101 / 102" articles are the canonical reference; the same model is implemented in Apache Flink, Apache Beam, Spark Structured Streaming, SQL-native streaming databases (**RisingWave** https://risingwave.com/, **Materialize** https://materialize.com/), and Rust-native engines (**Arroyo** https://github.com/ArroyoSystems/arroyo). For Python-centric ML feature pipelines, **Bytewax** is convenient but its release cadence has slowed (last open-source release Nov 2024); RisingWave is the more actively maintained SQL-first option as of 2025. Flink remains correct when you need Java/Scala custom operators or `MATCH_RECOGNIZE`.

```python
# Conceptual sketch — event-time windowing
def windowed_count(events, window_seconds=300, watermark_lag=60):
    """events sorted by arrival; each is (event_time, key)."""
    by_window = {}
    watermark = 0
    for arrived_at, ev_time, key in events:
        w_start = (ev_time // window_seconds) * window_seconds
        by_window.setdefault((w_start, key), 0)
        by_window[(w_start, key)] += 1
        watermark = max(watermark, ev_time - watermark_lag)
        # Emit windows whose end is past the watermark
        for (ws, k), c in list(by_window.items()):
            if ws + window_seconds <= watermark:
                yield (ws, k, c)
                del by_window[(ws, k)]
```

For ML feature engineering, the practical implication is: **your live feature pipeline must use the same time semantics as your training feature pipeline** — otherwise train/serve skew creeps in via the clock alone. Feature stores (Feast, Tecton, Chronon) exist primarily to enforce this consistency.

---

## 5. Feature Stores — Brief Framing

A feature store is a registry that:

1. Computes features once, **offline** for training and **online** for serving.
2. Guarantees the same transformation is used at both times (no train/serve skew).
3. Provides **point-in-time correct** historical lookups for backfills (no future leakage).

Use one when:
- Features are reused across multiple models.
- Train/serve skew has bitten you.
- You need point-in-time joins on time-stamped features.

For a single-model project with simple features, a feature store is overkill — a Polars feature pipeline shared between training and inference (literally the same Python module) is sufficient.

| Tool | Sweet spot |
|------|------------|
| [Feast](https://docs.feast.dev/) | Open-source, Python-native, lightweight; weak streaming |
| [Tecton](https://www.tecton.ai/) | Managed, real-time / streaming features — strongest for live features |
| [Hopsworks](https://www.hopsworks.ai/) | Open-source, Hudi-backed time travel, native streaming via Flink — best for hybrid / regulated workloads |
| [Featureform](https://www.featureform.com/) | Virtual feature store layered over your existing stores (no new data plane) |
| Vertex / SageMaker Feature Store | Cloud-native, lower operational overhead, slower feature velocity |
| [Chronon](https://github.com/airbnb/chronon) | Airbnb origin, niche; batch-first |

---

## 6. Decision Table — Which Approach?

| Situation | Approach |
|-----------|----------|
| Daily batch retrain meets SLA | Don't go online. Stay batch with drift monitoring. |
| Hourly retrain too slow, drift suspected | Warm-start every hour + drift detector |
| Per-record updates needed (recsys, fraud) | `river` or custom SGD with reservoir replay |
| Linear / NB / MLP, small-batch updates | sklearn `partial_fit` |
| Tree ensemble (XGBoost / LightGBM) needs incremental update | Refit with `init_model=` (warm-start); LightGBM's `refit()` for leaf-value updates only |
| Hot-swap weights without redeploy | Model-as-data control stream + atomic in-process pointer swap |
| Multiple replicas, want consistent params | MLflow / W&B model registry + poll-and-reload |
| Live feature aggregation with late-arriving events | Event-time windowing + watermarks (Flink / Bytewax / Beam) |
| Drift detection without labels | PSI + KS on features; χ² on prediction distribution |
| Drift detection with labels | ADWIN / DDM / Page-Hinkley on rolling error |
| Train/serve skew suspected | Move to a feature store (Feast / Tecton) |
| Need offline + online consistency for many features and models | Feature store |
| Catastrophic forgetting in online model | Reservoir replay + L2-toward-prior regularization |

---

## Common Gotchas

1. **No drift baseline saved at training time** → can't compute PSI later. Snapshot the reference distribution as a model artifact.
2. **PSI on a feature with mostly-zero distribution** → spurious alerts from rare bins. Use KS for sparse/skewed.
3. **Concept drift detected but auto-retrain pipeline broken** → detector fires daily, no one looks. Wire detection to alerts AND to a clear retrain runbook.
4. **`partial_fit` without seeing all classes in batch 0** → use `classes=` argument explicitly; otherwise the model "forgets" classes it hasn't seen yet.
5. **Online learning without reservoir replay** → catastrophic forgetting on rare classes.
6. **Hot-swap without atomicity** → mid-prediction request sees partially-loaded model. Use a single-pointer swap (assignment) or a read-write lock.
7. **Train pipeline uses pandas, serve pipeline uses Polars** → silent train/serve skew on edge-case rows. Share the exact transformation code.
8. **Window features on processing time** → events arriving late are double-counted or dropped. Use event time + watermarks.
9. **Deploying a freshly-online-updated model without canary** → bad update poisons all replicas instantly. Always canary updates, even rapid ones.
10. **Treating drift detection as model evaluation** → drift is a *warning*, not a verdict. The model can perform fine through covariate drift if P(Y|X) is unchanged.

---

## References

- Bifet & Gavaldà (2007) — *Learning from Time-Changing Data with Adaptive Windowing* (ADWIN): https://www.cs.upc.edu/~gavalda/papers/adwin06.pdf
- Gama et al. (2004) — Drift Detection Method (DDM): https://link.springer.com/chapter/10.1007/978-3-540-28645-5_29
- Page (1954) — Continuous inspection schemes (Page-Hinkley original): https://doi.org/10.1093/biomet/41.1-2.100
- `river` documentation: https://riverml.xyz/
- `frouros` (drift detection): https://frouros.readthedocs.io/
- `alibi-detect`: https://docs.seldon.io/projects/alibi-detect/
- `evidently`: https://docs.evidentlyai.com/
- Akidau — *Streaming 101* / *Streaming 102*: https://www.oreilly.com/radar/the-world-beyond-batch-streaming-101/
- Apache Flink event-time: https://nightlies.apache.org/flink/flink-docs-stable/docs/concepts/time/
- Feast: https://docs.feast.dev/
- Lublinsky, Kuznetsov, Skidanov — *Serving Machine Learning Models* (2017) — origin of the model-as-data control-stream pattern: https://www.oreilly.com/library/view/serving-machine-learning/9781492024095/
- Wampler — *Fast Data Architectures for Streaming Applications* (2016) — origin of the train/score separation framing used here: https://www.oreilly.com/library/view/fast-data-architectures/9781492045861/
- Sculley et al. (2015) — *Hidden Technical Debt in Machine Learning Systems* (training/serving skew): https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html

## See Also

- [`../evaluation/`](../evaluation/) — offline metric selection and model comparison.
- [`../experiment-tracking/`](../experiment-tracking/) — MLflow / W&B model registries that implement the model-as-data pattern.
- [`../../data-prep/data-validation/`](../../data-prep/data-validation/) — schema validation that pairs naturally with drift detection.
- [`../../data-prep/time-series-features/`](../../data-prep/time-series-features/) — windowed features that this skill turns into live, drift-resilient features.
- [`../online-experimentation/`](../online-experimentation/) — A/B and bandits for evaluating each rolled-out update.
- [`../../ml-libraries/ray/`](../../ml-libraries/ray/), [`../../ml-libraries/polars/`](../../ml-libraries/polars/) — runtime substrates for streaming feature pipelines.
