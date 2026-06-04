---
name: ml-data-validation
description: Data validation for ML pipelines -- schema enforcement, distribution drift detection, and quality gates that prevent garbage-in-garbage-out failures silently breaking models.
triggers:
  - data validation
  - data quality
  - schema validation
  - data drift
  - distribution shift
  - pandera
  - great expectations
  - evidently
  - data pipeline testing
  - feature validation
---

# ML Data Validation

## 1 — Why This Exists

Models fail silently. The training loss looks fine, metrics pass CI, then production degrades over weeks because:

- **Schema drift**: upstream adds a nullable column, renames a field, changes a categorical encoding
- **Distribution shift**: feature means drift 2σ from training baseline, but no error is thrown
- **Silent corruption**: NaN propagation, string-encoded nulls ("None", "null", ""), type coercion bugs
- **Stale features**: joins return NULL when a source table partitions change

Without validation gates, you discover these problems via model performance decay — days or weeks after the data broke.

**Mental model**: Data validation is the `assert` statement for your data pipeline. Fail fast, fail loud.

---

## 2 — Pandera (DataFrame Schema Validation)

**When to use**: You have pandas/polars DataFrames flowing through a pipeline and want compile-time-like guarantees on shape, types, and statistical properties.

**Install**: `pip install pandera`

**Docs**: https://pandera.readthedocs.io

### Schema Definition

```python
import pandera as pa
from pandera import Column, Check, Index

schema = pa.DataFrameSchema(
    columns={
        "user_id": Column(int, Check.gt(0), nullable=False),
        "age": Column(int, Check.in_range(0, 120), nullable=True),
        "score": Column(float, Check.in_range(0.0, 1.0)),
        "category": Column(str, Check.isin(["A", "B", "C"])),
        "embedding": Column(object, Check(lambda s: s.apply(len) == 128)),
    },
    index=Index(int, Check.ge(0)),
    coerce=True,  # attempt type coercion before failing
)

validated_df = schema.validate(df, lazy=True)  # lazy=True collects ALL errors
```

### Decorator Style (Class-based)

```python
from pandera import DataFrameModel, Field
from pandera.typing import Series

class FeatureSchema(DataFrameModel):
    user_id: Series[int] = Field(gt=0)
    age: Series[int] = Field(ge=0, le=120, nullable=True)
    score: Series[float] = Field(ge=0.0, le=1.0)

    class Config:
        coerce = True
        strict = True  # reject extra columns

    @pa.check("score")
    def score_not_all_zero(cls, series):
        return series.mean() > 0.01

@pa.check_io(df=FeatureSchema, out=FeatureSchema)
def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    ...
```

### Hypothesis Testing

```python
from pandera import Hypothesis

schema = pa.DataFrameSchema({
    "revenue": Column(float, [
        Hypothesis.two_sample_ttest(
            "group", "control", "treatment",
            relationship="greater_than", alpha=0.05
        ),
    ]),
})
```

### Gotchas

| Gotcha | Fix |
|--------|-----|
| `lazy=False` (default) raises on FIRST error only | Always use `lazy=True` in pipelines |
| Polars support is experimental | Use `pandera.polars` explicitly, check version |
| `coerce=True` silently converts — may mask bugs | Set `strict=True` + explicit coercion in ETL |
| Large DataFrames: validation is O(n) per check | Sample for statistical checks, full scan for schema |

---

## 3 — Great Expectations (Expectation Suites & Data Docs)

**When to use**: Enterprise-scale pipelines needing versioned expectation suites, auto-generated data docs, and checkpoint-based validation integrated with orchestrators (Airflow, Dagster).

**Install**: `pip install great_expectations`

**Docs**: https://docs.greatexpectations.io

### Core Concepts

- **Expectation**: A single assertion (`expect_column_values_to_not_be_null`)
- **Suite**: A collection of expectations for one dataset
- **Checkpoint**: Validates a batch against a suite, produces results
- **Data Docs**: Auto-generated HTML reports of validation results

### Quickstart

```python
import great_expectations as gx

context = gx.get_context()  # or gx.get_context(project_root_dir="./gx")

# Connect to data (GX 1.x API uses data_sources)
datasource = context.data_sources.add_pandas("my_source")
asset = datasource.add_dataframe_asset("training_features")
batch_request = asset.build_batch_request(dataframe=df)

# Create expectations
validator = context.get_validator(batch_request=batch_request)
validator.expect_column_values_to_not_be_null("user_id")
validator.expect_column_values_to_be_between("age", min_value=0, max_value=120)
validator.expect_column_mean_to_be_between("score", min_value=0.3, max_value=0.7)
validator.expect_column_proportion_of_unique_values_to_be_between("user_id", min_value=0.95)
validator.save_expectation_suite()

# Run checkpoint
checkpoint = context.add_or_update_checkpoint(
    name="training_data_check",
    validations=[{"batch_request": batch_request, "expectation_suite_name": "my_suite"}],
)
result = checkpoint.run()
assert result.success
```

### Profiler (Auto-generate expectations from reference data)

> **Note**: `UserConfigurableProfiler` was removed in GX 1.0. Use the
> `DataAssistant` API or manually define expectations based on the reference data.

```python
# GX 1.x: Use onboarding data assistant for auto-profiling
data_assistant_result = context.assistants.onboarding.run(
    batch_request=batch_request,
)
suite = data_assistant_result.get_expectation_suite(expectation_suite_name="auto_generated")
```

### Gotchas

| Gotcha | Fix |
|--------|-----|
| Heavy setup for simple use cases | Use Pandera for lightweight; GX for org-scale |
| GX 1.x API breaking change from 0.18 | `context.sources` → `context.data_sources` in GX 1.x |
| `context.data_sources` API differs filesystem vs cloud | Use `get_context()` and let it detect mode |
| Data Docs generation slow on large suites | Run docs build async or in CI only |

---

## 4 — Pydantic for Row-Level Validation

**When to use**: Streaming data, API inputs, or when you need per-record validation with rich error messages (e.g., validating individual training examples before batching).

```python
from pydantic import BaseModel, field_validator, model_validator
from typing import List, Optional
import numpy as np

class TrainingExample(BaseModel):
    user_id: int
    features: List[float]
    label: float
    timestamp: int

    @field_validator("features")
    @classmethod
    def check_embedding_dim(cls, v):
        if len(v) != 128:
            raise ValueError(f"Expected 128-d embedding, got {len(v)}")
        if any(np.isnan(x) for x in v):
            raise ValueError("NaN in features")
        return v

    @field_validator("label")
    @classmethod
    def label_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"Label {v} not in [0,1]")
        return v

    @model_validator(mode="after")
    def timestamp_not_future(self):
        import time
        if self.timestamp > time.time() + 3600:
            raise ValueError("Timestamp is in the future")
        return self

# Batch validation with error collection
errors = []
valid = []
for i, row in enumerate(raw_data):
    try:
        valid.append(TrainingExample(**row))
    except ValidationError as e:
        errors.append((i, e.errors()))

if len(errors) / len(raw_data) > 0.01:
    raise DataQualityError(f"{len(errors)} invalid rows ({len(errors)/len(raw_data):.1%})")
```

### When Pydantic vs Pandera

| Scenario | Use |
|----------|-----|
| Validate entire DataFrame at once | Pandera |
| Validate individual records/streaming | Pydantic |
| API request/response validation | Pydantic |
| Statistical assertions (mean, std, distribution) | Pandera or GX |
| Need JSON Schema export | Pydantic |

---

## 5 — Distribution Drift Detection

**When to use**: Monitoring feature distributions over time to detect when production data diverges from training data.

### Kolmogorov-Smirnov Test

```python
from scipy.stats import ks_2samp

def detect_drift_ks(reference: np.ndarray, current: np.ndarray, threshold: float = 0.05):
    """Returns True if distributions differ significantly."""
    stat, p_value = ks_2samp(reference, current)
    return p_value < threshold, stat, p_value

# Per-feature drift check
for col in feature_columns:
    drifted, stat, p = detect_drift_ks(train_df[col].values, prod_df[col].values)
    if drifted:
        logger.warning(f"DRIFT: {col} KS={stat:.3f} p={p:.4f}")
```

### Population Stability Index (PSI)

```python
def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI < 0.1: no shift. 0.1-0.2: moderate. > 0.2: significant."""
    ref_percents = np.histogram(reference, bins=bins)[0] / len(reference)
    cur_percents = np.histogram(current, bins=bins)[0] / len(current)

    # Avoid log(0)
    ref_percents = np.clip(ref_percents, 1e-4, None)
    cur_percents = np.clip(cur_percents, 1e-4, None)

    psi = np.sum((cur_percents - ref_percents) * np.log(cur_percents / ref_percents))
    return psi
```

### Evidently (Full Drift Reports)

**Install**: `pip install evidently`

**Docs**: https://docs.evidentlyai.com

> **API Note**: Evidently v0.7+ (April 2025) introduced a breaking API change.
> The code below uses the **new API** (v0.7+). If using v0.6.x, import from
> `evidently.future` instead. For v0.5 and earlier, see [old docs](https://docs-old.evidentlyai.com).

```python
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataQualityPreset

# Wrap data in Dataset objects (required in v0.7+)
reference = Dataset.from_pandas(
    train_df,
    data_definition=DataDefinition()  # auto-detects column types
)
current = Dataset.from_pandas(
    prod_df,
    data_definition=DataDefinition()
)

# Compare reference (training) vs current (production)
report = Report([
    DataDriftPreset(),           # all columns
    DataQualityPreset(),         # nulls, duplicates, types
])
result = report.run(current, reference)
result.save_html("drift_report.html")

# Programmatic access
result_dict = result.as_dict()
```

### Evidently Tests (CI-friendly assertions)

In v0.7+, Tests are unified into Reports via the `tests` parameter on metrics:

```python
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently.metrics import MaxValue, ShareOfMissingValues
from evidently.tests import lt, gt

reference = Dataset.from_pandas(train_df, data_definition=DataDefinition())
current = Dataset.from_pandas(prod_df, data_definition=DataDefinition())

# Tests are now part of the Report object
report = Report([
    DataDriftPreset(),
    ShareOfMissingValues(tests=[lt(0.05)]),  # < 5% missing
])
result = report.run(current, reference)
# Check if all tests passed via the result object
```

### Decision Table — Which Drift Method

| Method | Best For | Limitations |
|--------|----------|-------------|
| KS test | Continuous features, small-medium data | Sensitive to sample size; p-values unreliable at N>10K |
| PSI | Binned/categorical, production monitoring | Bin choice affects result; not great for multimodal |
| Evidently | Full pipeline monitoring, reports, CI | Heavier dependency; overkill for single-feature check |
| Jensen-Shannon | Probability distributions, symmetric | Requires density estimation step |
| Wasserstein | Continuous, captures magnitude of shift | Computationally expensive for high-dim |

---

## 6 — Integration with Training Pipelines

### Pattern: Validation Gates

```python
# training_pipeline.py
from dataclasses import dataclass

@dataclass
class ValidationResult:
    passed: bool
    errors: list
    drift_report: dict

def validate_training_data(df, reference_stats) -> ValidationResult:
    errors = []

    # 1. Schema validation (Pandera)
    try:
        FeatureSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        errors.extend(e.failure_cases.to_dict("records"))

    # 2. Drift detection
    drift = {}
    for col in NUMERIC_FEATURES:
        psi = calculate_psi(reference_stats[col], df[col].values)
        if psi > 0.2:
            drift[col] = psi
            errors.append(f"Drift: {col} PSI={psi:.3f}")

    # 3. Business rules
    if df["label"].mean() < 0.01 or df["label"].mean() > 0.99:
        errors.append(f"Label imbalance: mean={df['label'].mean():.4f}")

    return ValidationResult(
        passed=len(errors) == 0,
        errors=errors,
        drift_report=drift,
    )

# In pipeline
result = validate_training_data(train_df, baseline_stats)
if not result.passed:
    if any("Drift" in e for e in result.errors):
        alert_oncall(f"Data drift detected: {result.drift_report}")
        # Option: retrain with recent data, or halt
    raise DataQualityError(result.errors)
```

### Pattern: Reference Stats Snapshot

```python
import json

def save_reference_stats(df, path="reference_stats.json"):
    """Save distribution stats from validated training data."""
    stats = {}
    for col in df.select_dtypes(include="number").columns:
        stats[col] = {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max(),
            "quantiles": df[col].quantile([0.25, 0.5, 0.75]).tolist(),
            "histogram": np.histogram(df[col].dropna(), bins=20)[0].tolist(),
        }
    with open(path, "w") as f:
        json.dump(stats, f)
```

### Pattern: Airflow/Dagster Integration

```python
# Airflow operator
from airflow.decorators import task
from airflow.exceptions import AirflowFailException

@task
def validate_features(ds=None):
    df = load_features(ds)
    ref = load_reference_stats()
    result = validate_training_data(df, ref)
    if not result.passed:
        raise AirflowFailException(f"Validation failed: {result.errors[:5]}")
    return {"rows": len(df), "drift": result.drift_report}

# DAG: extract >> validate_features >> train_model
```

---

## 7 — Gotchas & Anti-Patterns

| Anti-Pattern | Better |
|-------------|--------|
| Validate only in development, skip in prod | Validate at every pipeline boundary |
| Hard-fail on any drift | Set thresholds (PSI > 0.2 = block, 0.1-0.2 = warn) |
| Validate after model training | Validate BEFORE training starts |
| One giant schema for all stages | Separate schemas: raw → cleaned → features → model input |
| Ignoring validation errors in logs | Fail the pipeline or page oncall |
| Statistical tests on tiny batches | KS test needs N≥50; PSI needs N≥200 per bin |
| Checking only means/nulls | Also check correlations, cardinality, and tail behavior |

---

## 8 — References

- Pandera docs: https://pandera.readthedocs.io
- Pandera readthedocs (full API): https://pandera.readthedocs.io/
- Pandera GitHub (unionai-oss): https://github.com/unionai-oss/pandera
- Great Expectations docs: https://docs.greatexpectations.io
- Great Expectations expectations gallery: https://greatexpectations.io/expectations/
- Evidently AI docs: https://docs.evidentlyai.com
- Evidently GitHub: https://github.com/evidentlyai/evidently
- Pydantic docs: https://docs.pydantic.dev
- scipy.stats.ks_2samp: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
- "ML Writing" (Eugene Yan): https://eugeneyan.com/writing/
- "Monitoring ML Models in Production" (Google): https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
