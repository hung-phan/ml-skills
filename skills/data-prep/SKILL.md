---
name: data-prep
description: Data preparation and validation before training — EDA, feature engineering, and data quality gates. Use when exploring raw data, creating features, or preventing garbage-in-garbage-out.
---

# Data Prep & Validation

| Skill | Covers |
|-------|--------|
| [eda](eda/) | Statistical summaries, distributions, correlations, outliers, missing data, hypothesis testing, profiling |
| [feature-engineering](feature-engineering/) | Numeric transforms, categorical encoding, text/datetime features, interactions, featuretools, tsfresh |
| [time-series-features](time-series-features/) | Lag/rolling windows, calendar/seasonality, autocorrelation/Fourier, point-process aggregates, purged + embargoed CV — classical (non-DL) TS feature engineering for tabular forecasting and recsys/CTR |
| [data-validation](data-validation/) | Schema enforcement (Pandera), expectation suites (Great Expectations), drift detection (Evidently, KS, PSI) |

For training, evaluation, and feature selection see `ml-training/`. For live drift handling and incremental updates see [`ml-training/online-learning/`](../ml-training/online-learning/).
