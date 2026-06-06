---
name: time-series-features
description: Classical (non-deep-learning) time-series feature engineering — lags, rolling/expanding windows, autocorrelation, Fourier/periodogram features, calendar/seasonality, point-process aggregations (time-since-last-event, inter-arrival), purged/embargoed CV. Use when forecasting tabular time-series with XGBoost/sklearn, building features for CTR/recsys from event logs, or extracting features from user activity streams without using an RNN/Transformer.
---

# Time-Series Feature Engineering

## Why This Exists

**Problem**: Most production "time-series" workloads aren't ARIMA or RNNs — they're a tabular forecasting or classification problem (sales, demand, CTR, churn) where rows have a timestamp and the model is XGBoost. The hard part is converting an irregular, leak-prone temporal signal into a set of fixed-width features that a tabular learner can consume *without leaking the future*. The library's existing TS coverage (`rnn/`, `mamba/`) is all deep-learning; the day-to-day classical feature-engineering work has had no home.

**Key insight**: Classical TS features fall into three families — **windowed aggregates** (mean/std/min/max over a lookback), **temporal-domain transforms** (lags, diffs, autocorrelation, Fourier), and **point-process aggregates** (time-since-last-event, inter-arrival statistics). The dual rule: any series can be framed as a regularly-sampled signal *or* as a point process — pick the cheaper one for your features.

**Reach for this when**: forecasting demand/sales/traffic with XGBoost/LightGBM; building CTR/recsys features from event logs; need recency features for churn/conversion models; have irregular event timestamps and need fixed-width features; want to avoid leakage in time-aware cross-validation.

---

## 0. When to Skip This Skill: TS Foundation Models and AutoML

Before hand-engineering features, ask whether you should skip this skill entirely. The 2023-2025 wave of time-series foundation models has changed the starting point for a real slice of forecasting work.

| Situation | Try first | Why |
|-----------|-----------|-----|
| Univariate forecasting, limited covariates, short history | **Chronos-Bolt** / **Chronos-2** (Oct 2025) or **TimesFM-2.0** zero-shot | SOTA on GIFT-Eval / fev-bench; no training, no feature engineering |
| Tabular forecasting, want a 3-line baseline with no manual work | **`autogluon.timeseries`** (1.2, Nov 2024) | Ensembles statistical + GBM + GluonTS deep + Chronos-Bolt with auto lag/calendar features |
| Many exogenous covariates, irregular event streams, latency/cost-bound, or non-forecasting task (TS-features-for-classification) | **Stay in this skill** | FMs underperform on multivariate/CTR/point-process; opaque feature importance |

```python
# Chronos-Bolt zero-shot (no training)
from chronos import ChronosPipeline
pipe = ChronosPipeline.from_pretrained("amazon/chronos-bolt-base")
forecasts = pipe.predict(context=history_tensor, prediction_length=28)

# AutoGluon-TS one-liner
from autogluon.timeseries import TimeSeriesPredictor
predictor = TimeSeriesPredictor(prediction_length=28).fit(train_data, presets="best_quality")
predictions = predictor.predict(train_data)
```

**Decision rule**: try a foundation model in 30 minutes; if the zero-shot accuracy beats your business floor, you're done. If covariates dominate or the use case isn't pure forecasting, the rest of this skill applies. Foundation-model coverage is also why per-event recsys / CTR features remain in scope here — no FM currently handles point processes.

References: https://github.com/amazon-science/chronos-forecasting · https://github.com/google-research/timesfm · https://auto.gluon.ai/stable/tutorials/timeseries/index.html · https://arxiv.org/abs/2410.10393 (GIFT-Eval benchmark).

---

## 1. Lag and Window Features

The bread-and-butter. Always compute on **strictly past** data (no current row, no future).

Synthetic dataset shape: `date: date, store_id: int, units: int` (one row per store per day).

```python
import polars as pl

df = pl.read_parquet("sales.parquet")  # cols: date, store_id, units

# Lag features per group
df = df.with_columns([
    pl.col("units").shift(1).over("store_id").alias("lag_1"),
    pl.col("units").shift(7).over("store_id").alias("lag_7"),
    pl.col("units").shift(28).over("store_id").alias("lag_28"),
])

# Rolling window features (closed='left' = strictly past, NO leakage)
df = df.with_columns([
    pl.col("units").rolling_mean(7,  min_samples=7, closed="left").over("store_id").alias("rmean_7"),
    pl.col("units").rolling_mean(28, min_samples=28, closed="left").over("store_id").alias("rmean_28"),
    pl.col("units").rolling_std(28,  min_samples=28, closed="left").over("store_id").alias("rstd_28"),
    pl.col("units").rolling_max(28,  min_samples=28, closed="left").over("store_id").alias("rmax_28"),
])

# Expanding window (cumulative, also closed='left')
df = df.with_columns(
    pl.col("units").cum_sum().shift(1).over("store_id").alias("cumsum_so_far")
)

# Differences and ratios (signal-derived features)
df = df.with_columns([
    (pl.col("units") - pl.col("lag_7")).alias("week_delta"),
    (pl.col("rmean_7") / pl.col("rmean_28")).alias("short_over_long"),
])
```

**The leakage rule**: `closed="left"` (or `shift(1)` then `rolling_*`) excludes the current row. `closed="right"` includes it — *that's leakage on a forecasting target.* Almost every leakage bug in tabular TS comes from this single mistake.

### pandas equivalent

```python
import pandas as pd

df["rmean_7"] = (
    df.groupby("store_id")["units"]
      .shift(1)                          # exclude current row first
      .rolling(7, min_periods=7).mean()
      .reset_index(level=0, drop=True)
)
```

---

## 2. Calendar and Seasonality Features

Cheap, high-signal features for any human-driven series (sales, ads, traffic).

```python
df = df.with_columns([
    pl.col("date").dt.weekday().alias("dow"),                  # 1..7
    pl.col("date").dt.day().alias("dom"),
    pl.col("date").dt.month().alias("month"),
    pl.col("date").dt.week().alias("iso_week"),
    pl.col("date").dt.ordinal_day().alias("doy"),              # 1..366
    pl.col("date").dt.year().alias("year"),
    # Cyclical encoding — keeps Dec close to Jan, Sun close to Mon
    (2 * 3.14159 * pl.col("date").dt.month() / 12).sin().alias("month_sin"),
    (2 * 3.14159 * pl.col("date").dt.month() / 12).cos().alias("month_cos"),
    (2 * 3.14159 * pl.col("date").dt.weekday() / 7).sin().alias("dow_sin"),
    (2 * 3.14159 * pl.col("date").dt.weekday() / 7).cos().alias("dow_cos"),
])
```

For business cycles add: holiday indicators (`holidays` package), payday flags, school-term flags, store-specific local events. Use `lag_7` plus `dow` together — they encode different things (last week's *level* vs *which day of week we are on*).

---

## 3. Autocorrelation and Spectral Features

When the series has hidden periodicity or persistence, point-statistics on the time domain do not capture it. Two compact families fix this.

```python
import numpy as np
from numpy.fft import rfft

def acf_features(x, lags=(1, 7, 28)):
    """Autocorrelation at given lags, computed from past `x` only."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = (x ** 2).mean()
    return {f"acf_{k}": float((x[k:] * x[:-k]).mean() / (var + 1e-12)) for k in lags}

def spectral_features(x, top_k=3):
    """Top-k Fourier components by amplitude — captures dominant periods."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    spec = np.abs(rfft(x))
    idx = np.argsort(spec)[-top_k:][::-1]
    feats = {}
    for i, k in enumerate(idx):
        period = len(x) / (k + 1e-9)
        feats[f"fft_amp_{i}"]    = float(spec[k])
        feats[f"fft_period_{i}"] = float(period)
    return feats
```

For batched feature extraction at scale, use `tsfresh`'s catalog (700+ features) or `sktime` / `darts`:

```python
# tsfresh — automatic feature extraction, then prune by p-value relevance
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series

# Roll the long series into windows tsfresh can ingest
rolled = roll_time_series(df.to_pandas(), column_id="store_id",
                          column_sort="date", max_timeshift=28, min_timeshift=7)
features = extract_features(rolled, column_id="id", column_sort="date")
# Then select_features(features, y) for supervised pruning.
```

`tsfresh` is heavyweight — it computes hundreds of features per window. Use it as an *exploration tool* to discover which families matter, then re-implement the winners in Polars for production.

---

## 4. Point-Process Features (Irregular Event Streams)

Logs of "user clicked at t1, t2, t3..." are *point processes*, not regularly-sampled series. Bin-then-rolling-window throws away resolution. Compute event-level features directly.

```python
def point_process_features(timestamps_so_far, now):
    """timestamps_so_far: sorted array of past event times (strictly < now)."""
    if len(timestamps_so_far) == 0:
        return dict(time_since_last=np.nan, n_events_1h=0, n_events_24h=0,
                    iat_mean=np.nan, iat_std=np.nan)
    t = np.asarray(timestamps_so_far)
    iat = np.diff(t) if len(t) >= 2 else np.array([np.nan])
    return {
        "time_since_last": float(now - t[-1]),
        "n_events_1h":     int(((now - t) <= 3600).sum()),
        "n_events_24h":    int(((now - t) <= 86400).sum()),
        "iat_mean":        float(np.nanmean(iat)),
        "iat_std":         float(np.nanstd(iat)),
        # Hawkes-style decayed activity (recent events weigh more)
        "decay_score_1h":  float(np.exp(-(now - t) / 3600).sum()),
    }
```

Common point-process features: **time-since-last-event**, **count in window** (1h/24h/7d), **inter-arrival mean/std/CV**, **exponentially decayed event sum** (poor-man's Hawkes intensity). These dominate recsys/CTR feature importance — recency is almost always the strongest signal.

---

## 5. Time-Aware Cross-Validation (Avoid Future Leakage in CV)

Standard k-fold leaks the future into the past. Three correct splits:

| Split | Idea | When |
|-------|------|------|
| **Expanding window** (`TimeSeriesSplit`) | Fold k uses [0, t_k] for train, [t_k, t_{k+1}] for val | Default for forecasting |
| **Sliding window** | Fixed-width train window slides forward | Concept drift, want recent-only train |
| **Purged + embargoed** (López de Prado) | Drop train rows whose label window overlaps val (purge); add a buffer after val (embargo) | Multi-horizon labels, financial / leakage-prone |
| **Combinatorial Purged CV (CPCV)** | Generate C(N, k) train/test combinations → many backtest paths + Probability of Backtest Overfitting (PBO) and Deflated Sharpe Ratio diagnostics | Strategy selection, multiple-testing risk, financial backtests — preferred over single-path purged k-fold |

For finance, prefer **CPCV over single-path purged k-fold** — single-path purging gives one backtest, CPCV gives many and lets you compute PBO/DSR to detect strategies that look good purely by chance. See `purgedcv` (https://pypi.org/project/purgedcv/) and the ML4T writeup (https://ml4trading.io/docs/diagnostic/methods/cpcv/).

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Expanding window
tscv = TimeSeriesSplit(n_splits=5, gap=7)  # `gap` = embargo days
for train_idx, val_idx in tscv.split(X):
    fit_and_score(X[train_idx], y[train_idx], X[val_idx], y[val_idx])

def purged_kfold(timestamps, label_horizon, n_splits=5, embargo=7):
    """Each train fold drops rows whose [t, t+horizon] overlaps the val fold,
    plus an `embargo` buffer after the val fold."""
    n = len(timestamps)
    ts = np.asarray(timestamps).astype("datetime64[D]")
    fold_size = n // n_splits
    for k in range(n_splits):
        val_start, val_end = k * fold_size, (k + 1) * fold_size
        val_idx = np.arange(val_start, val_end)
        val_t_min, val_t_max = ts[val_idx].min(), ts[val_idx].max()
        # Purge: training rows whose label window touches val window
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False
        train_t_end = ts + np.timedelta64(label_horizon, "D")
        overlap = (train_t_end >= val_t_min) & (ts <= val_t_max + np.timedelta64(embargo, "D"))
        train_mask[overlap] = False
        yield np.where(train_mask)[0], val_idx
```

If your label is "did the user convert in the next 7 days" and you don't purge, training windows for rows near the validation cutoff *contain the validation labels*. This is the most common silent leak in tabular TS at scale.

---

## 6. Anti-Leakage Checklist

Run through this before training any tabular TS model:

1. **Every rolling/expanding feature uses `closed="left"`** (or `shift(1)` first).
2. **Group-aware** — `.over(group_col)` / `groupby(group_col)` for every per-entity rolling.
3. **Target encoding uses past only** — encode with cumulative mean over time, never global mean.
4. **CV is time-aware** — no random k-fold on time-series; use `TimeSeriesSplit` or purged.
5. **Test set is strictly later** than train+val. No shuffled holdout.
6. **Calendar features that depend on the future** (e.g. "days until next holiday") are OK. Features that depend on future *labels* are not.
7. **Static features computed at row-time, not load-time** — store-size as of today is leakage if today's row is from 3 years ago.
8. **Look at feature importance** — if a "future-blind" feature dominates wildly above what's plausible, you have leakage.

---

## 7. End-to-End Pattern (Polars + LightGBM)

```python
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# 1. Load + sort
df = pl.read_parquet("sales.parquet").sort(["store_id", "date"])

# 2. Build lag/window features (closed='left')
df = df.with_columns([
    pl.col("units").shift(1).over("store_id").alias("lag_1"),
    pl.col("units").shift(7).over("store_id").alias("lag_7"),
    pl.col("units").rolling_mean(28, closed="left").over("store_id").alias("rmean_28"),
    pl.col("units").rolling_std(28,  closed="left").over("store_id").alias("rstd_28"),
    pl.col("date").dt.weekday().alias("dow"),
    pl.col("date").dt.month().alias("month"),
])

# 3. Drop the warmup rows that have NaN in the longest window
df = df.drop_nulls()

# 4. Time-aware CV
features = [c for c in df.columns if c not in ("date", "store_id", "units")]
X = df.select(features).to_numpy()
y = df["units"].to_numpy()

tscv = TimeSeriesSplit(n_splits=5, gap=7)
for fold, (tr, va) in enumerate(tscv.split(X)):
    model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05, max_depth=-1,
                              num_leaves=127, min_data_in_leaf=200)
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(50)])
    print(fold, model.best_score_)
```

---

## When to Use What

| Need | Use |
|------|-----|
| Univariate forecasting zero-shot (limited covariates) | Chronos-Bolt / Chronos-2 / TimesFM-2.0 — see §0 |
| Tabular AutoML baseline with no manual work | `autogluon.timeseries` — see §0 |
| Forecasting tabular sales/demand with XGBoost/LightGBM | This skill (lags + windows + calendar + time-aware CV) |
| Self-supervised embeddings as features (lots of unlabeled series) | TS2Vec / T-Rep — https://github.com/zhihanyue/ts2vec, https://github.com/let-it-care/t-rep |
| Long sequence DL (>10k tokens) | `../../ml-architectures/mamba/` |
| Standard sequence DL | `../../ml-architectures/rnn/` or `transformer/` |
| Auto-extract 700+ features for exploration | `tsfresh` |
| Forecasting framework with backtester | `darts`, `sktime`, `mlforecast` |
| Per-event recsys / CTR features | This skill (point-process + decayed counts) |
| Concept drift / online learning | [`../../ml-training/online-learning/`](../../ml-training/online-learning/) |
| Detecting feature drift between train and serve | [`../data-validation/`](../data-validation/) |
| Choosing the metric and reporting | [`../../ml-training/evaluation/`](../../ml-training/evaluation/) |

---

## Common Gotchas

1. **`closed="right"` rolling**: includes the current row → leaks the target. Default to `"left"`.
2. **Random shuffled CV on time series**: trivially leaks. Use `TimeSeriesSplit`.
3. **Multi-horizon labels without purging**: train labels overlap val window. Use purged + embargoed CV.
4. **Group-naive rolling on multi-entity data**: rolls across stores. Always `.over(group)`.
5. **Target-encoding categorical with global mean**: leaks future labels. Use expanding-mean target encoding.
6. **Filling NaNs with global mean before split**: leaks. Fill inside the fold.
7. **Tsfresh on raw long series**: blows up memory. Roll into windows first.
8. **Feature drift but no monitoring**: model decays silently in production. Pair with `data-validation/` + `online-learning/`.
9. **`fft` on series with trend**: low-frequency component dominates spectrum. Detrend (subtract mean / linear fit) before FFT.
10. **Confusing inter-arrival CV with calendar period**: a period-7 process has IAT mean ≈ 1 day, not 7. They measure different things.

---

## References

- López de Prado — *Advances in Financial Machine Learning* (2018), Ch. 7 (purged + embargoed CV): https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
- `tsfresh` documentation: https://tsfresh.readthedocs.io/en/latest/
- `sktime`: https://www.sktime.net/
- `darts` (Unit8): https://unit8co.github.io/darts/
- `mlforecast` (Nixtla): https://nixtlaverse.nixtla.io/mlforecast/
- Polars rolling expressions: https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.rolling_mean.html
- Hyndman & Athanasopoulos — *Forecasting: Principles and Practice* (3e), free online: https://otexts.com/fpp3/
- Hawkes processes (intensity-based event modelling): https://en.wikipedia.org/wiki/Hawkes_process
- Brink, Richards, Fetherolf — *Real-World Machine Learning* (2017), Ch. 7 — origin of the windowed-features taxonomy used here: https://www.manning.com/books/real-world-machine-learning

## See Also

- [`../feature-engineering/`](../feature-engineering/) — non-temporal feature engineering (encoding, interactions, scaling).
- [`../data-validation/`](../data-validation/) — schema and drift checks for time-aware data.
- [`../../ml-training/evaluation/`](../../ml-training/evaluation/) — pick metrics and statistically compare TS models.
- [`../../ml-training/online-learning/`](../../ml-training/online-learning/) — incremental updates and drift handling for live TS models.
- [`../../ml-architectures/rnn/`](../../ml-architectures/rnn/), [`../../ml-architectures/mamba/`](../../ml-architectures/mamba/) — when to graduate to deep sequence models.
