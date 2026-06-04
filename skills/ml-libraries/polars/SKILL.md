---
name: polars
description: Use when user wants fast DataFrame operations, mentions polars, lazy evaluation, scan_parquet, streaming, or asks about high-performance data processing as a pandas alternative.
---

# Polars Skill

## Why This Exists

1. **Problem solved**: High-performance DataFrame operations that scale to datasets larger than RAM. Polars eliminates pandas' single-threaded GIL-bound execution and eager memory allocation by using a Rust engine with automatic parallelism, lazy query optimization, and streaming execution — turning multi-minute pandas jobs into seconds.

2. **When to pick this over alternatives**: Choose polars over pandas when data exceeds ~1M rows, when pipelines run repeatedly (ETL, batch jobs), or when you need out-of-memory processing. Choose pandas when you need ecosystem compatibility (sklearn, seaborn), quick one-off EDA, or your team doesn't know the expression API. Choose Spark/Dask for truly distributed (multi-node) workloads.

3. **Mental model**: polars = declarative expression engine with lazy optimization. You describe WHAT you want (filter, group, join) using composable expressions, and the query planner decides HOW to execute — pushing predicates to file readers, pruning unused columns, and parallelizing across cores automatically. Think SQL query optimizer, not step-by-step scripting.

Fast DataFrame library in Rust with Python bindings. Use instead of pandas for performance-critical data work.

## Core Concepts

### Lazy vs Eager Evaluation

```python
import polars as pl

# Eager -- executes immediately (like pandas)
df = pl.read_csv("data.csv")
result = df.filter(pl.col("age") > 30).select("name", "age")

# Lazy -- builds query plan, optimizes, then executes
result = (
    pl.scan_csv("data.csv")
    .filter(pl.col("age") > 30)
    .select("name", "age")
    .collect()  # triggers execution
)

# Convert eager to lazy
lazy = df.lazy()
# Convert lazy to eager
eager = lazy.collect()

# Stream large results (constant memory)
lazy.collect(streaming=True)
```

**Rule:** Default to lazy (`scan_*`) for files. Use eager only for small in-memory transforms.

## Expression API

Expressions are the core building block. Never use Python loops on DataFrames.

```python
# col() -- reference columns
pl.col("name")
pl.col("price", "quantity")  # multiple cols
pl.col("^revenue_.*$")      # regex select

# lit() -- literal values
pl.lit(42)
pl.lit(None)

# Arithmetic
pl.col("price") * pl.col("quantity")
(pl.col("a") + pl.col("b")) / pl.col("c")

# when-then-otherwise (vectorized if/else)
pl.when(pl.col("age") >= 18)
  .then(pl.lit("adult"))
  .when(pl.col("age") >= 13)
  .then(pl.lit("teen"))
  .otherwise(pl.lit("child"))
  .alias("category")

# Casting
pl.col("id").cast(pl.Int64)
pl.col("date_str").str.to_date("%Y-%m-%d")
```

## GroupBy & Aggregations

```python
df.group_by("department").agg(
    pl.col("salary").mean().alias("avg_salary"),
    pl.col("salary").max().alias("max_salary"),
    pl.col("name").count().alias("headcount"),
    pl.col("bonus").sum(),
    pl.col("name").first(),
    (pl.col("salary") > 100_000).sum().alias("high_earners"),
)

# Multiple group keys
df.group_by("dept", "level").agg(...)

# Dynamic groupby (time-based)
df.group_by_dynamic("timestamp", every="1h").agg(
    pl.col("value").mean()
)

# Rolling groupby
df.group_by_dynamic("date", every="1d", period="7d").agg(
    pl.col("sales").sum().alias("rolling_7d_sales")
)
```

## Join Types

```python
# Inner join (default)
df1.join(df2, on="id")
df1.join(df2, left_on="user_id", right_on="id")

# Left/right/outer/cross/semi/anti
df1.join(df2, on="id", how="left")
df1.join(df2, on="id", how="anti")   # rows in df1 NOT in df2
df1.join(df2, on="id", how="semi")   # rows in df1 that ARE in df2

# Multiple join keys
df1.join(df2, on=["id", "date"])

# Asof join (nearest match, for time series)
df1.join_asof(df2, on="timestamp", strategy="backward", tolerance="5m")

# Suffix for duplicate columns
df1.join(df2, on="id", suffix="_right")
```

## Window Functions (over)

Compute aggregates without collapsing rows.

```python
# Rank within group
df.with_columns(
    pl.col("salary").rank().over("department").alias("dept_rank")
)

# Running sum per group
df.with_columns(
    pl.col("amount").cum_sum().over("account_id").alias("running_total")
)

# Group statistics as new columns
df.with_columns(
    pl.col("salary").mean().over("department").alias("dept_avg"),
    pl.col("salary").max().over("department").alias("dept_max"),
)

# Multiple partition keys
pl.col("value").sum().over("region", "year")

# Mapping: how to handle output length
pl.col("val").rank().over("grp", mapping_strategy="explode")  # one row per input
pl.col("val").sort().over("grp", mapping_strategy="join")     # broadcast sorted
```

## String Expressions

```python
df.with_columns(
    pl.col("name").str.to_lowercase(),
    pl.col("email").str.contains("@amazon"),
    pl.col("url").str.extract(r"https?://([^/]+)", 1).alias("domain"),
    pl.col("text").str.replace_all(r"\s+", " "),
    pl.col("full_name").str.split(" ").list.first().alias("first_name"),
    pl.col("code").str.starts_with("US-"),
    pl.col("tags").str.strip_chars(),
    pl.col("a").str.concat(pl.col("b"), separator="-"),  # use pl.concat_str for multi
)

# concat_str for multiple columns
pl.concat_str(["first", "last"], separator=" ").alias("full_name")
```

## Datetime Expressions

```python
df.with_columns(
    pl.col("ts").dt.year().alias("year"),
    pl.col("ts").dt.month(),
    pl.col("ts").dt.weekday(),
    pl.col("ts").dt.hour(),
    pl.col("ts").dt.truncate("1h").alias("hour_bucket"),
    pl.col("ts").dt.offset_by("3d"),
    (pl.col("end") - pl.col("start")).dt.total_hours().alias("duration_h"),
)

# Parse strings to datetime
pl.col("date_str").str.to_datetime("%Y-%m-%d %H:%M:%S")
pl.col("epoch_ms").cast(pl.Datetime("ms"))

# Timezone
pl.col("ts").dt.convert_time_zone("America/New_York")
pl.col("ts").dt.replace_time_zone("UTC")
```

## Schema Handling

```python
# Inspect schema
df.schema       # dict of {col_name: dtype}
df.dtypes       # list of dtypes
df.columns      # list of column names

# Explicit schema on read
pl.read_csv("f.csv", schema={"id": pl.Int64, "name": pl.Utf8, "val": pl.Float64})
pl.scan_csv("f.csv", dtypes={"id": pl.Int64})

# Rename
df.rename({"old_name": "new_name"})

# Select with type
df.select(pl.col(pl.Float64))           # all float columns
df.select(pl.col("^metric_.*$"))        # regex

# Add/remove columns
df.with_columns(pl.lit(0).alias("new_col"))
df.drop("unwanted_col", "another")

# Schema enforcement
df.cast({"price": pl.Float64, "qty": pl.Int32})
```

## Reading Large Files

```python
# Lazy scan -- only reads what's needed after optimization
lf = pl.scan_csv("huge.csv")
lf = pl.scan_parquet("huge.parquet")
lf = pl.scan_parquet("s3://bucket/data/**/*.parquet")  # glob

# Streaming collect for out-of-memory
result = lf.filter(...).select(...).collect(streaming=True)

# Parquet: predicate pushdown reads only matching row groups
lf = pl.scan_parquet("data.parquet").filter(pl.col("year") == 2024)

# Read specific columns only (projection pushdown)
lf = pl.scan_parquet("data.parquet").select("id", "value")

# Sink directly to file (never fully materializes)
lf.filter(...).sink_parquet("output.parquet")
lf.filter(...).sink_csv("output.csv")

# IPC/Arrow for fastest I/O
pl.scan_ipc("data.arrow")

# Chunked reads (manual)
reader = pl.read_csv_batched("huge.csv", batch_size=100_000)
while (batch := reader.next_batches(1)):
    process(batch[0])
```

## Polars vs Pandas Translation

| Pandas | Polars |
|--------|--------|
| `df["col"]` | `df.select("col")` or `df["col"]` (Series) |
| `df[df["a"] > 5]` | `df.filter(pl.col("a") > 5)` |
| `df["new"] = df["a"] + 1` | `df.with_columns((pl.col("a") + 1).alias("new"))` |
| `df.groupby("g").agg({"v": "sum"})` | `df.group_by("g").agg(pl.col("v").sum())` |
| `df.merge(df2, on="k")` | `df.join(df2, on="k")` |
| `df.apply(fn)` | `df.with_columns(pl.col("x").map_elements(fn))` ⚠️ slow |
| `df.sort_values("col")` | `df.sort("col")` |
| `df.drop_duplicates()` | `df.unique()` |
| `df.fillna(0)` | `df.fill_null(0)` |
| `df.isna()` | `pl.col("x").is_null()` |
| `df.rename(columns={...})` | `df.rename({...})` |
| `pd.concat([df1, df2])` | `pl.concat([df1, df2])` |
| `df.pivot_table(...)` | `df.pivot(on=..., index=..., values=...)` |
| `df.melt(...)` | `df.unpivot(on=..., index=...)` |
| `df.iterrows()` | Don't. Use expressions. |

**Key mindset shift:** In polars, you describe WHAT you want (expressions), not HOW to compute it step-by-step. Avoid `map_elements`/`apply` -- there's almost always a native expression.

## Performance Tips

### Query Optimization (Lazy mode)

```python
# Predicate pushdown: filter early, polars pushes to scan
pl.scan_parquet("data.parquet")
  .filter(pl.col("year") == 2024)  # pushed to parquet reader
  .select("name", "value")          # only these columns read

# Projection pushdown: only selected columns are read from disk
# Happens automatically in lazy mode

# Check the optimized plan
lf.explain()           # show optimized plan
lf.explain(optimized=False)  # show unoptimized for comparison
```

### General Performance

1. **Use lazy mode** -- enables all optimizations
2. **Avoid `map_elements`** -- drops to Python, loses parallelism
3. **Prefer expressions over Python UDFs** -- 10-100x faster
4. **Use categoricals** for low-cardinality strings: `df.cast({"status": pl.Categorical})`
5. **Use `streaming=True`** for datasets larger than RAM
6. **Parquet > CSV** -- columnar, compressed, supports predicate pushdown
7. **Pre-sort join keys** if joining repeatedly on same key
8. **Use `sink_parquet`** instead of `collect()` + `write_parquet()` for large outputs
9. **Avoid `clone()`** -- polars uses copy-on-write internally
10. **Set `n_rows`** on `scan_csv` during development to iterate fast

### Streaming for Out-of-Memory

```python
# Process arbitrarily large files
(
    pl.scan_csv("100gb_file.csv")
    .filter(pl.col("status") == "active")
    .group_by("region")
    .agg(pl.col("revenue").sum())
    .collect(streaming=True)
)

# Sink to avoid materializing
(
    pl.scan_parquet("huge/")
    .filter(...)
    .with_columns(...)
    .sink_parquet("output/result.parquet")
)
```

## Common Patterns

```python
# Conditional column creation
df.with_columns(
    pl.when(pl.col("score") >= 90).then(pl.lit("A"))
      .when(pl.col("score") >= 80).then(pl.lit("B"))
      .otherwise(pl.lit("C"))
      .alias("grade")
)

# Explode list column
df.explode("tags")

# Struct unnesting
df.unnest("metadata")  # expands struct fields to columns

# Null handling
df.with_columns(
    pl.col("value").fill_null(strategy="forward"),
    pl.col("name").fill_null("unknown"),
    pl.coalesce("preferred_name", "full_name", "username").alias("display_name"),
)

# Row numbers
df.with_row_index("idx")

# Sample
df.sample(n=1000)
df.sample(fraction=0.1)

# Horizontal operations
df.with_columns(
    pl.sum_horizontal("a", "b", "c").alias("total"),
    pl.max_horizontal("x", "y").alias("max_xy"),
)
```

## When to Use

| ✅ Use Polars | ❌ Don't Use |
|---|---|
| Large datasets where speed matters | Tiny scripts where pandas is simpler |
| Lazy evaluation for query optimization | Need sklearn/seaborn direct compat (expects pandas) |
| Multi-core utilization out of the box | Team doesn't know Polars expression API |
| Memory-efficient columnar operations | Interop with libraries that only accept pandas |
| ETL pipelines, data processing at scale | Quick one-off EDA (pandas is faster to write) |

**Decision rule**: If data >1M rows or pipeline runs repeatedly → Polars. If quick EDA or ecosystem compat needed → pandas.

---

## References

- [Polars Documentation](https://docs.pola.rs/)
- [Polars GitHub](https://github.com/pola-rs/polars)