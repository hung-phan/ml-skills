---
name: pandas
description: Use when user wants to work with DataFrames, mentions pandas, groupby, merge, join, pivot, read_csv, parquet, or asks about data manipulation and SettingWithCopyWarning.
---

# Pandas Skill

## Why This Exists

1. **Problem solved**: Tabular data manipulation without writing raw loops over arrays. Pandas gives you labeled rows/columns, SQL-like operations (filter, group, join, pivot), and automatic alignment — things that would take hundreds of lines of manual index-tracking code with plain Python lists or numpy arrays.

2. **When to pick this over alternatives**: Choose pandas over polars when you need ecosystem compatibility (sklearn, seaborn, matplotlib all expect pandas DataFrames), rapid EDA in notebooks, or your team already knows it. Choose polars when performance matters (>1M rows, repeated pipelines). Choose numpy when you only need numerical arrays without labels.

3. **Mental model**: pandas = labeled arrays with SQL-like operations. A DataFrame is a dict of named Series (columns) sharing a common index (row labels). Every operation either returns a new DataFrame (method chaining) or modifies in-place. The index enables automatic alignment in arithmetic and joins.

> Comprehensive pandas patterns for data manipulation, performance, and correctness.

## Triggers

Use when user mentions: pandas, dataframe, series, groupby, merge, join, pivot, melt, read_csv, parquet, vectorized, SettingWithCopyWarning, chained indexing, memory optimization, window functions, rolling, apply, dtypes, categories.

---

## 1. DataFrame Creation

```python
import pandas as pd
import numpy as np

# From dict (column-oriented) → shape (3, 2)
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

# From records (row-oriented) → shape (2, 3)
df = pd.DataFrame.from_records([{"x": 1, "y": 2, "z": 3}, {"x": 4, "y": 5, "z": 6}])

# From numpy → shape (100, 4)
df = pd.DataFrame(np.random.randn(100, 4), columns=list("ABCD"))

# Typed creation (memory-efficient from the start)
df = pd.DataFrame({
    "id": pd.array([1, 2, 3], dtype="int32"),
    "name": pd.Categorical(["a", "b", "a"]),
    "score": pd.array([1.1, 2.2, 3.3], dtype="float32"),
})
```

---

## 2. Indexing: loc / iloc / at / iat

```python
# loc: label-based (inclusive on both ends)
df.loc[0:5, "col_a":"col_c"]        # rows 0-5, cols col_a through col_c
df.loc[df["age"] > 30, ["name"]]    # boolean mask + column selection

# iloc: integer position-based (exclusive end)
df.iloc[0:5, 0:3]                   # first 5 rows, first 3 cols
df.iloc[-1]                         # last row as Series

# at/iat: scalar access (faster than loc/iloc for single values)
df.at[0, "name"]                    # label-based scalar
df.iat[0, 2]                        # position-based scalar
```

### ⚠️ Anti-pattern: Chained Indexing

```python
# ❌ NEVER DO THIS — triggers SettingWithCopyWarning, may silently fail
df[df["a"] > 1]["b"] = 99

# ✅ ALWAYS use loc for assignment
df.loc[df["a"] > 1, "b"] = 99
```

---

## 3. GroupBy Patterns

```python
# Basic aggregation → shape (n_groups, n_agg_cols)
df.groupby("category")["revenue"].agg(["sum", "mean", "count"])

# Multiple columns + named aggregation (pandas >= 0.25)
df.groupby("category").agg(
    total_rev=("revenue", "sum"),
    avg_price=("price", "mean"),
    n_orders=("order_id", "nunique"),
)

# Transform: broadcast back to original shape
df["pct_of_group"] = df.groupby("category")["revenue"].transform(
    lambda x: x / x.sum()
)

# Filter: keep groups meeting condition
df.groupby("category").filter(lambda g: g["revenue"].sum() > 1000)

# Iterate (use sparingly — prefer agg/transform)
for name, group in df.groupby("category"):
    process(group)
```

### ⚠️ Anti-pattern: apply with aggregation logic

```python
# ❌ Slow — Python-level loop per group
df.groupby("cat")["val"].apply(lambda x: x.sum())

# ✅ Use built-in aggregation
df.groupby("cat")["val"].sum()
```

---

## 4. Merge / Join / Concat

```python
# merge (SQL-style joins) — default inner join
result = pd.merge(left, right, on="key")                         # inner
result = pd.merge(left, right, on="key", how="left")             # left outer
result = pd.merge(left, right, left_on="id", right_on="key_id")  # different col names

# Validate merge cardinality (catch silent fanout)
result = pd.merge(left, right, on="key", validate="one_to_one")
# Options: "one_to_one", "one_to_many", "many_to_one", "many_to_many"

# indicator column for debugging
result = pd.merge(left, right, on="key", how="outer", indicator=True)
# _merge column: "left_only", "right_only", "both"

# concat (stack vertically or horizontally)
combined = pd.concat([df1, df2], ignore_index=True)              # vertical, reset index
combined = pd.concat([df1, df2], axis=1)                         # horizontal

# join (index-based merge shortcut)
result = left.join(right, on="key", how="left")
```

### ⚠️ Anti-pattern: Repeated concat in loop

```python
# ❌ O(n²) memory — copies entire DataFrame each iteration
result = pd.DataFrame()
for chunk in chunks:
    result = pd.concat([result, chunk])

# ✅ Collect then concat once — O(n)
result = pd.concat(list(chunks), ignore_index=True)
```

---

## 5. Window Functions

```python
# Rolling (fixed window)
df["ma_7"] = df["price"].rolling(7).mean()
df["std_30"] = df["price"].rolling(30).std()

# Rolling with min_periods (handle NaN at start)
df["ma_7"] = df["price"].rolling(7, min_periods=1).mean()

# Expanding (cumulative from start)
df["cummax"] = df["price"].expanding().max()
df["cumsum"] = df["revenue"].expanding().sum()

# EWM (exponentially weighted)
df["ema_12"] = df["price"].ewm(span=12).mean()

# Rank within group (common for feature engineering)
df["rank"] = df.groupby("category")["score"].rank(method="dense", ascending=False)

# Shift / diff / pct_change
df["prev_day"] = df["price"].shift(1)
df["daily_change"] = df["price"].diff()
df["daily_return"] = df["price"].pct_change()
```

---

## 6. Apply vs Vectorized Operations

**Rule: Vectorized > built-in methods > apply with numpy > apply with Python.**

```python
# ❌ apply with Python function — slowest
df["result"] = df["a"].apply(lambda x: x ** 2 + 1)

# ✅ Vectorized — 10-100x faster
df["result"] = df["a"] ** 2 + 1

# ❌ Row-wise apply — extremely slow
df["full_name"] = df.apply(lambda row: f"{row['first']} {row['last']}", axis=1)

# ✅ Vectorized string ops
df["full_name"] = df["first"] + " " + df["last"]

# When you MUST use apply (complex logic, no vectorized equivalent):
# Use raw=True for numpy array access (avoids Series overhead)
df["result"] = df[["a", "b", "c"]].apply(np.sum, axis=1, raw=True)

# Better yet: use .values for numpy
df["result"] = df[["a", "b", "c"]].values.sum(axis=1)
```

---

## 7. Memory Optimization

```python
# Check memory usage
df.info(memory_usage="deep")
df.memory_usage(deep=True).sum() / 1e6  # MB

# Downcast numeric types
df["int_col"] = pd.to_numeric(df["int_col"], downcast="integer")
df["float_col"] = pd.to_numeric(df["float_col"], downcast="float")

# Use categories for low-cardinality strings (massive savings)
# Before: 100K rows × "status" col with 5 unique values = ~6.4 MB
# After:  same data as Categorical = ~0.1 MB
df["status"] = df["status"].astype("category")

# Optimal dtype selection function
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["int"]):
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float"]):
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["object"]):
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype("category")
    return df

# Use nullable dtypes (avoid float promotion from NaN)
df["id"] = df["id"].astype("Int64")      # nullable integer
df["flag"] = df["flag"].astype("boolean") # nullable boolean
```

---

## 8. Common Pitfalls

### SettingWithCopyWarning

```python
# Root cause: operating on a VIEW vs a COPY
subset = df[df["a"] > 1]  # might be a view
subset["b"] = 99          # ⚠️ may not modify original df

# Fix 1: explicit copy
subset = df[df["a"] > 1].copy()
subset["b"] = 99  # safe — modifying independent copy

# Fix 2: operate on original with loc
df.loc[df["a"] > 1, "b"] = 99
```

### pandas 2.0+ Copy-on-Write (CoW)

```python
# Enable globally (default in pandas 3.0)
pd.options.mode.copy_on_write = True
# With CoW, every indexing op returns a copy — no more ambiguity
```

### Silent dtype coercion

```python
# ❌ NaN in integer column silently promotes to float64
df = pd.DataFrame({"id": [1, 2, None]})  # id becomes float64

# ✅ Use nullable integer
df = pd.DataFrame({"id": pd.array([1, 2, None], dtype="Int64")})
```

### Index alignment surprises

```python
# ❌ Assigning Series with different index → NaN injection
s = pd.Series([10, 20, 30], index=[0, 1, 2])
df["new"] = s.values  # ✅ use .values to bypass index alignment
```

---

## 9. Performance: eval / query / pipe

```python
# query: readable filtering (uses numexpr under the hood)
df.query("age > 30 and city == 'NYC'")
# Supports @variable references
min_age = 30
df.query("age > @min_age")

# eval: fast column operations without temp arrays
df.eval("profit = revenue - cost", inplace=True)
df.eval("margin = profit / revenue")

# pipe: composable transformations (method chaining)
result = (
    df.pipe(clean_names)
      .pipe(filter_outliers, column="price", n_std=3)
      .pipe(add_features)
      .pipe(optimize_dtypes)
)

# Define pipe functions with df as first arg
def filter_outliers(df: pd.DataFrame, column: str, n_std: int = 3) -> pd.DataFrame:
    mean, std = df[column].mean(), df[column].std()
    return df[df[column].between(mean - n_std * std, mean + n_std * std)]
```

---

## 10. I/O Patterns

```python
# CSV — always specify dtypes for large files
df = pd.read_csv("data.csv", dtype={"id": "int32", "name": "category"})

# Chunked reading for large CSVs
chunks = pd.read_csv("huge.csv", chunksize=100_000)
result = pd.concat(
    [chunk.query("status == 'active'") for chunk in chunks],
    ignore_index=True,
)

# Parquet — preferred for analytics (columnar, compressed, typed)
df.to_parquet("data.parquet", engine="pyarrow", compression="snappy")
df = pd.read_parquet("data.parquet", columns=["col_a", "col_b"])  # read only needed cols

# JSON — use lines=True for JSONL (one record per line)
df = pd.read_json("data.jsonl", lines=True)
df.to_json("out.jsonl", orient="records", lines=True)

# Excel
df = pd.read_excel("data.xlsx", sheet_name="Sheet1", engine="openpyxl")

# SQL (use SQLAlchemy connection)
from sqlalchemy import create_engine
engine = create_engine("postgresql://user:pass@host/db")
df = pd.read_sql("SELECT * FROM table WHERE date > '2024-01-01'", engine)

# Feather (fast serialization for intermediate results)
df.to_feather("cache.feather")
df = pd.read_feather("cache.feather")
```

### Performance hierarchy for file formats:
1. **Parquet** — best for analytics, compression, column pruning
2. **Feather** — fastest read/write, no compression, good for caching
3. **CSV** — universal but slow, no type info, large on disk

---

## 11. Quick Reference: Method Chaining Style

```python
# Readable, composable, no intermediate variables
result = (
    pd.read_parquet("sales.parquet")
    .query("region == 'US' and revenue > 0")
    .assign(
        margin=lambda d: d["revenue"] - d["cost"],
        quarter=lambda d: d["date"].dt.to_period("Q"),
    )
    .groupby(["quarter", "product"])
    .agg(total_margin=("margin", "sum"), n_sales=("id", "count"))
    .reset_index()
    .sort_values("total_margin", ascending=False)
)
```

---

## 12. Anti-Patterns Summary

| ❌ Don't | ✅ Do |
|----------|-------|
| `df[mask]["col"] = val` | `df.loc[mask, "col"] = val` |
| `df.apply(lambda x: x**2)` | `df["col"] ** 2` |
| Loop `pd.concat` in for-loop | Collect list, concat once |
| `df.iterrows()` for computation | Vectorized ops or `.values` |
| Default dtypes for large data | Downcast + categories |
| `df.append()` (deprecated) | `pd.concat([df, new])` |
| Read entire parquet file | `columns=` param for needed cols |
| `axis=1` apply for string ops | Vectorized `.str` methods |
| `inplace=True` everywhere | Method chaining (returns new df) |

## When to Use

| ✅ Use pandas | ❌ Don't Use |
|---|---|
| Exploratory analysis, quick prototyping | Data exceeds RAM (>10GB) |
| Complex groupby/merge/pivot operations | Need speed on >1M rows (use Polars) |
| Ecosystem compatibility (sklearn, seaborn) | Pure numerical compute (use NumPy) |
| Time series with DatetimeIndex | Streaming/real-time data processing |
| Legacy codebases and team familiarity | When memory efficiency is critical |

**Decision rule**: Default for EDA and small-medium data. Switch to Polars when speed/memory matters.

---

## References

- [pandas Documentation](https://pandas.pydata.org/docs/)
- [pandas GitHub](https://github.com/pandas-dev/pandas)