---
name: seaborn
description: Statistical visualization library on top of matplotlib — heatmaps, pairplots, FacetGrid, distribution plots, and regression overlays. Use when creating publication-quality static plots from pandas DataFrames or doing exploratory statistical visualization.
---

# Seaborn Skill

Statistical data visualization built on matplotlib. Use when generating publication-quality plots from DataFrames.

## Why This Exists

**Problem**: Matplotlib's raw API requires substantial boilerplate for statistical visualizations common in ML — drawing distributions, confidence intervals, regression lines, and categorical comparisons each demand manual aggregation and layout code.

**Key insight**: Seaborn adds a grammar-of-graphics-like layer over matplotlib that produces publication-quality statistical plots (with automatic confidence intervals, faceting, and sensible defaults) in a single function call.

**Reach for this when**: You need static statistical charts for papers, reports, or presentations — distributions, heatmaps, pairplots, regression overlays, or any plot where the data lives in a DataFrame and interactivity is not required. Use Plotly instead when charts need to be interactive or embedded in a web app.

## Figure-Level vs Axes-Level Functions

**Figure-level** (return `FacetGrid`/`PairGrid`/`JointGrid`): create their own figure, support `col`/`row` faceting.
**Axes-level** (return `matplotlib.axes.Axes`): draw into existing axes, composable in subplots.

```python
# Figure-level -- creates its own figure, facets natively
g = sns.relplot(data=df, x="x", y="y", col="category", kind="scatter")
g.set_titles("{col_name}")
g.fig.suptitle("Title", y=1.02)

# Axes-level -- draws into existing axes
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.scatterplot(data=df, x="x", y="y", ax=axes[0])
sns.histplot(data=df, x="x", ax=axes[1])
```

| Figure-level | Axes-level counterparts | Domain |
|---|---|---|
| `relplot` | `scatterplot`, `lineplot` | Relationships |
| `displot` | `histplot`, `kdeplot`, `ecdfplot`, `rugplot` | Distributions |
| `catplot` | `stripplot`, `swarmplot`, `boxplot`, `violinplot`, `barplot`, `pointplot`, `countplot` | Categorical |
| `lmplot` | `regplot`, `residplot` | Regression |
| `jointplot` | (composite) | Bivariate + marginals |
| `pairplot` | (composite) | All-pairs grid |

## relplot (Relational)

```python
sns.relplot(
    data=df, x="total_bill", y="tip",
    hue="smoker", style="time", size="size",
    col="day", col_wrap=2,
    kind="scatter",  # or "line"
    height=4, aspect=1.2
)
```

For time series with confidence intervals:
```python
sns.relplot(data=fmri, x="timepoint", y="signal",
            hue="region", style="event", kind="line",
            errorbar="sd")  # "ci", "pi", "sd", or (func, level)
```

## displot (Distributions)

```python
sns.displot(data=df, x="total_bill", hue="time",
            kind="kde",  # "hist", "kde", "ecdf"
            fill=True, common_norm=False,
            facet_kws=dict(sharey=False))

# Bivariate
sns.displot(data=df, x="total_bill", y="tip", kind="kde", fill=True)
```

## catplot (Categorical)

```python
sns.catplot(data=df, x="day", y="total_bill",
            hue="smoker", col="time",
            kind="violin",  # "strip","swarm","box","violin","bar","point","count"
            split=True, inner="quart",
            height=5, aspect=0.7)
```

## Statistical Estimation

Control error bars / confidence intervals globally:

```python
# Bootstrap CI (default: 95%)
sns.barplot(data=df, x="day", y="total_bill", errorbar=("ci", 95))

# Standard deviation
sns.pointplot(data=df, x="day", y="total_bill", errorbar="sd")

# Standard error
sns.lineplot(data=df, x="x", y="y", errorbar=("se", 1))

# Percentile interval
sns.lineplot(data=df, x="x", y="y", errorbar=("pi", 50))

# Disable
sns.barplot(data=df, x="day", y="total_bill", errorbar=None)

# Custom estimator
sns.barplot(data=df, x="day", y="total_bill", estimator="median")
```

## FacetGrid and PairGrid

```python
# FacetGrid -- map any plotting function across facets
g = sns.FacetGrid(df, col="time", row="smoker", hue="sex",
                  height=3, aspect=1.5, margin_titles=True)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip")
g.add_legend()
g.set_axis_labels("Total Bill ($)", "Tip ($)")

# PairGrid -- full control over diagonal/upper/lower
g = sns.PairGrid(df, vars=["x", "y", "z"], hue="category",
                 corner=True)  # lower triangle only
g.map_lower(sns.scatterplot, alpha=0.6)
g.map_diag(sns.histplot, kde=True)
g.add_legend()
```

## Themes and Contexts

```python
# Themes control style (background, grid, spines)
sns.set_theme(style="whitegrid")  # "darkgrid","white","dark","ticks"

# Context scales font, line width, tick size for target medium
sns.set_theme(context="talk")  # "paper","notebook","talk","poster"

# Combined
sns.set_theme(style="ticks", context="paper", font_scale=1.1,
              rc={"axes.spines.right": False, "axes.spines.top": False})

# Temporary
with sns.axes_style("white"):
    sns.boxplot(data=df, x="day", y="total_bill")
    sns.despine(trim=True)
```

## Color Palettes

```python
# Qualitative (categorical)
sns.set_palette("Set2")            # or "tab10", "husl", "Paired"
sns.color_palette("husl", n_colors=8)

# Sequential (ordered)
sns.color_palette("viridis", as_cmap=True)  # "rocket","mako","flare","crest"

# Diverging (centered)
sns.color_palette("coolwarm", as_cmap=True)  # "vlag","icefire","RdBu_r"

# Use in plot
sns.scatterplot(data=df, x="x", y="y", hue="value",
                palette="viridis")  # string or list of colors

# Custom
custom = sns.color_palette(["#e74c3c", "#2ecc71", "#3498db"])
sns.barplot(data=df, x="cat", y="val", palette=custom)

# Continuous hue
sns.scatterplot(data=df, x="x", y="y", hue="continuous_var",
                palette="flare", hue_norm=(0, 100))
```

## Hue / Style / Size Semantics

Map data dimensions to visual properties:

```python
sns.relplot(
    data=df, x="x", y="y",
    hue="category",      # color mapping (categorical or continuous)
    style="group",       # marker shape / line dash
    size="magnitude",    # marker size / line width
    sizes=(20, 200),     # min/max for size
    style_order=["A", "B"],
    hue_order=["cat1", "cat2"],
    legend="full"        # "full", "brief", "auto", False
)
```

## jointplot / pairplot for EDA

```python
# Quick bivariate EDA
sns.jointplot(data=df, x="x", y="y", hue="group",
              kind="kde",  # "scatter","kde","hist","hex","reg","resid"
              marginal_kws=dict(fill=True))

# All-pairs overview
sns.pairplot(df, hue="species",
             diag_kind="kde",  # "hist","kde",None
             plot_kws=dict(alpha=0.5),
             corner=True)
```

## Heatmaps and Clustermaps

```python
# Basic heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f",
            cmap="coolwarm", center=0,
            vmin=-1, vmax=1,
            square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8})

# Clustermap -- hierarchical clustering on rows/cols
g = sns.clustermap(data_matrix, method="ward", metric="euclidean",
                   z_score=0,  # standardize rows
                   cmap="vlag", center=0,
                   figsize=(10, 10),
                   row_colors=label_colors,  # sidebar annotation
                   dendrogram_ratio=0.15)
g.ax_heatmap.set_xlabel("Features")
```

## Regression Plots

```python
# lmplot -- figure-level, supports faceting
sns.lmplot(data=df, x="x", y="y", hue="group",
           col="category", robust=True,  # robust regression
           order=2,  # polynomial degree
           scatter_kws={"alpha": 0.5},
           line_kws={"lw": 2})

# regplot -- axes-level
ax = sns.regplot(data=df, x="x", y="y",
                 lowess=True,  # locally weighted regression
                 scatter_kws={"s": 10})

# residplot -- check fit quality
sns.residplot(data=df, x="x", y="y", lowess=True)
```

## Customization (Matplotlib Integration)

```python
import matplotlib.pyplot as plt

# Modify figure-level plot after creation
g = sns.catplot(data=df, x="day", y="total_bill", kind="box")
g.fig.set_size_inches(8, 5)
g.set_titles("{col_name} meals")
g.set_xlabels("Day of week")
g.set_ylabels("Total bill ($)")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Spending by Day")

# Access underlying axes
for ax in g.axes.flat:
    ax.axhline(20, ls="--", color="red", alpha=0.5)
    ax.set_ylim(0, 60)

# Combine seaborn + matplotlib
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="day", y="total_bill", ax=ax)
ax.set_title("Custom Title")
ax.annotate("Outlier", xy=(2, 50), fontsize=9)
plt.tight_layout()
plt.savefig("plot.png", dpi=150, bbox_inches="tight")
```

## Common Pitfalls

### Figure size with figure-level functions
```python
# WRONG -- fig-level ignores figsize
plt.figure(figsize=(10, 6))
sns.relplot(data=df, x="x", y="y")  # creates NEW figure

# RIGHT -- use height/aspect params
sns.relplot(data=df, x="x", y="y", height=5, aspect=1.5)

# Or resize after
g = sns.relplot(data=df, x="x", y="y")
g.fig.set_size_inches(10, 6)
```

### Legend placement
```python
# Move legend outside
g = sns.relplot(data=df, x="x", y="y", hue="cat")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

# For axes-level
ax = sns.scatterplot(data=df, x="x", y="y", hue="cat")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()  # prevent clipping
```

### Overplotting
```python
# Use alpha, jitter, or switch plot type
sns.stripplot(data=df, x="day", y="tip", jitter=0.2, alpha=0.4)
# Or use swarmplot for small N (slow for large N)
sns.swarmplot(data=df, x="day", y="tip", size=3)
```

### Categorical axis ordering
```python
# Seaborn uses data order by default
sns.boxplot(data=df, x="day", y="total_bill",
            order=["Thur", "Fri", "Sat", "Sun"])
```

### Saving figures from figure-level objects
```python
# WRONG
plt.savefig("out.png")  # may save empty figure

# RIGHT
g = sns.relplot(data=df, x="x", y="y")
g.savefig("out.png", dpi=150, bbox_inches="tight")
```

### Long labels / tick overlap
```python
g = sns.catplot(data=df, x="long_category", y="value", kind="bar")
g.set_xticklabels(rotation=45, ha="right")
# Or flip axes
sns.catplot(data=df, y="long_category", x="value", kind="bar")
```

### Multiple plots sharing axes incorrectly
```python
# Each figure-level call creates a new figure
# To overlay, use axes-level functions on the same ax:
fig, ax = plt.subplots()
sns.kdeplot(data=df, x="x", hue="group", ax=ax)
sns.rugplot(data=df, x="x", hue="group", ax=ax)
```

## When to Use

| ✅ Use seaborn | ❌ Don't Use |
|---|---|
| Statistical plots (distributions, regressions) | Interactive dashboards (use Plotly) |
| Publication-quality static figures | Web apps needing user interaction |
| Quick EDA with sensible defaults | Real-time updating charts |
| Showing relationships in data (hue, facets) | 3D visualizations |
| Papers, reports, presentations | When you need tooltips/zoom/pan |

**Decision rule**: Static statistical visualization for papers/reports → seaborn. Interactive web → Plotly.

---

## References

- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Seaborn GitHub](https://github.com/mwaskom/seaborn)