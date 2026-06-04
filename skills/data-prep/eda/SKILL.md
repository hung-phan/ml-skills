---
name: EDA
description: Use when user wants exploratory data analysis, statistical profiling, distribution testing, outlier detection, correlation analysis, hypothesis testing, or automated profiling with ydata-profiling/sweetviz.
---

# Exploratory Data Analysis (EDA)

Complete patterns for statistical exploration, distribution testing, outlier detection, correlation analysis, hypothesis testing, and automated profiling.

---

## Why This Exists

**Problem**: Jumping straight to modeling without exploring the data leads to wrong assumptions about distributions, undetected outliers, and missed data quality issues — the model learns from garbage. Silent failures include: a regression target that is log-normal being fed to a linear model expecting Gaussian residuals, outliers inflating variance and distorting feature scaling, or two features that are perfectly correlated making coefficient estimates unstable.

**Key insight**: EDA is the discipline of understanding your data before trusting it — every modeling decision (choice of loss function, scaling strategy, imputation method, feature encoding) should be grounded in what EDA reveals about distributions, missingness patterns, and relationships.

**Reach for this when**: Starting any ML project before writing a single model line; diagnosing unexpectedly poor model performance; auditing a new dataset for a production pipeline; or comparing train/test distributions to detect leakage or shift.

---

## 1. Statistical Summaries

```python
import pandas as pd
import numpy as np
from scipy.stats import entropy

def extended_describe(df: pd.DataFrame) -> pd.DataFrame:
    """Extended statistical summary beyond df.describe()."""
    numeric = df.select_dtypes(include="number")
    stats = numeric.describe().T
    stats["skew"] = numeric.skew()
    stats["kurtosis"] = numeric.kurtosis()
    stats["iqr"] = stats["75%"] - stats["25%"]
    stats["cv"] = stats["std"] / stats["mean"]  # coefficient of variation
    stats["missing_pct"] = df[numeric.columns].isna().mean() * 100
    stats["nunique"] = numeric.nunique()
    stats["zeros_pct"] = (numeric == 0).mean() * 100
    return stats

def categorical_entropy(df: pd.DataFrame) -> pd.Series:
    """Shannon entropy for categorical columns (higher = more uniform)."""
    cats = df.select_dtypes(include=["object", "category"])
    result = {}
    for col in cats.columns:
        counts = cats[col].value_counts(normalize=True)
        result[col] = entropy(counts, base=2)
    return pd.Series(result, name="entropy_bits")

# Usage
stats = extended_describe(df)
cat_entropy = categorical_entropy(df)
print(stats[stats["cv"].abs() > 2])  # high variability columns
print(cat_entropy.sort_values())  # low entropy = dominated by few values
```

---

## 2. Distribution Analysis

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def test_normality(series: pd.Series, alpha: float = 0.05) -> dict:
    """Test normality with Shapiro-Wilk and KS test."""
    clean = series.dropna()
    n = len(clean)

    # Shapiro-Wilk (best for n < 5000)
    if n < 5000:
        sw_stat, sw_p = stats.shapiro(clean)
    else:
        sw_stat, sw_p = np.nan, np.nan

    # Kolmogorov-Smirnov against normal
    ks_stat, ks_p = stats.kstest(clean, "norm", args=(clean.mean(), clean.std()))

    return {
        "n": n,
        "mean": clean.mean(),
        "std": clean.std(),
        "skewness": stats.skew(clean),
        "kurtosis": stats.kurtosis(clean),
        "shapiro_stat": sw_stat,
        "shapiro_p": sw_p,
        "ks_stat": ks_stat,
        "ks_p": ks_p,
        "is_normal": (sw_p > alpha if n < 5000 else ks_p > alpha),
    }

def qq_plot(series: pd.Series, title: str = "QQ Plot") -> plt.Figure:
    """Generate QQ plot against normal distribution."""
    fig, ax = plt.subplots(figsize=(6, 6))
    clean = series.dropna()
    stats.probplot(clean, dist="norm", plot=ax)
    ax.set_title(title)
    ax.get_lines()[0].set_markerfacecolor("steelblue")
    ax.get_lines()[0].set_alpha(0.6)
    plt.tight_layout()
    return fig

def distribution_report(df: pd.DataFrame) -> pd.DataFrame:
    """Test all numeric columns for normality."""
    numeric = df.select_dtypes(include="number")
    results = []
    for col in numeric.columns:
        result = test_normality(numeric[col])
        result["column"] = col
        results.append(result)
    return pd.DataFrame(results).set_index("column")

# Usage
report = distribution_report(df)
non_normal = report[~report["is_normal"]]
print(f"Non-normal columns: {len(non_normal)}/{len(report)}")
fig = qq_plot(df["target_col"], title="Target QQ Plot")
```

---

## 3. Correlation Analysis

```python
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

def multi_correlation(df: pd.DataFrame, target: str = None) -> dict:
    """Compute Pearson, Spearman, Kendall correlations."""
    numeric = df.select_dtypes(include="number")
    result = {
        "pearson": numeric.corr(method="pearson"),
        "spearman": numeric.corr(method="spearman"),
        "kendall": numeric.corr(method="kendall"),
    }
    if target and target in numeric.columns:
        result["target_correlations"] = pd.DataFrame({
            "pearson": numeric.corr(method="pearson")[target],
            "spearman": numeric.corr(method="spearman")[target],
            "kendall": numeric.corr(method="kendall")[target],
        }).drop(target).sort_values("spearman", key=abs, ascending=False)
    return result

def mutual_information_scores(
    df: pd.DataFrame, target: str, task: str = "regression"
) -> pd.Series:
    """Mutual information between features and target."""
    numeric = df.select_dtypes(include="number").dropna()
    X = numeric.drop(columns=[target])
    y = numeric[target]

    if task == "regression":
        mi = mutual_info_regression(X, y, random_state=42)
    else:
        mi = mutual_info_classif(X, y, random_state=42)

    return pd.Series(mi, index=X.columns, name="mutual_info").sort_values(ascending=False)

def phik_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Phi_K correlation matrix (works for mixed types)."""
    import phik
    return df.phik_matrix()

# Usage
corrs = multi_correlation(df, target="price")
print(corrs["target_correlations"].head(10))

mi_scores = mutual_information_scores(df, target="price", task="regression")
print(mi_scores.head(10))

# Phi_K for mixed categorical + numeric
phik_matrix = phik_correlation(df[["category", "region", "price", "quantity"]])
```

---

## 4. Outlier Detection

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def iqr_outliers(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """IQR-based outlier detection. Returns boolean mask."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return (series < lower) | (series > upper)

def modified_zscore_outliers(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Modified Z-score using MAD (robust to existing outliers)."""
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        mad = series.std() * 0.6745  # fallback
    modified_z = 0.6745 * (series - median) / mad
    return np.abs(modified_z) > threshold

def isolation_forest_outliers(
    df: pd.DataFrame, contamination: float = 0.05
) -> pd.Series:
    """Isolation Forest for multivariate outlier detection."""
    numeric = df.select_dtypes(include="number").dropna()
    iso = IsolationForest(
        contamination=contamination, random_state=42, n_jobs=-1
    )
    labels = iso.fit_predict(numeric)
    return pd.Series(labels == -1, index=numeric.index, name="is_outlier")

def lof_outliers(
    df: pd.DataFrame, n_neighbors: int = 20, contamination: float = 0.05
) -> pd.Series:
    """Local Outlier Factor for density-based outlier detection."""
    numeric = df.select_dtypes(include="number").dropna()
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination
    )
    labels = lof.fit_predict(numeric)
    return pd.Series(labels == -1, index=numeric.index, name="is_outlier")

def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """Compare outlier methods across all numeric columns."""
    numeric = df.select_dtypes(include="number")
    results = []
    for col in numeric.columns:
        s = numeric[col].dropna()
        results.append({
            "column": col,
            "iqr_outliers": iqr_outliers(s).sum(),
            "mad_outliers": modified_zscore_outliers(s).sum(),
            "iqr_pct": iqr_outliers(s).mean() * 100,
            "mad_pct": modified_zscore_outliers(s).mean() * 100,
        })
    report = pd.DataFrame(results).set_index("column")
    # Add multivariate
    iso_mask = isolation_forest_outliers(df)
    lof_mask = lof_outliers(df)
    report.loc["__multivariate_iso__", "total_outliers"] = iso_mask.sum()
    report.loc["__multivariate_lof__", "total_outliers"] = lof_mask.sum()
    return report

# Usage
report = outlier_report(df)
print(report.sort_values("iqr_pct", ascending=False).head(10))
```

---

## 5. Missing Data Analysis

```python
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Missing data pattern analysis."""
    missing = df.isna().sum()
    pct = df.isna().mean() * 100
    summary = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": pct,
        "dtype": df.dtypes,
    })
    summary = summary[summary["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )
    summary["pattern"] = summary["missing_pct"].apply(
        lambda x: "high" if x > 50 else "moderate" if x > 10 else "low"
    )
    return summary

def missingness_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation between missingness indicators (detect MNAR patterns)."""
    missing_cols = df.columns[df.isna().any()]
    indicators = df[missing_cols].isna().astype(int)
    return indicators.corr()

def littles_mcar_approximation(df: pd.DataFrame) -> dict:
    """
    Approximate Little's MCAR test using chi-square on group means.
    True Little's test requires EM; this is a practical approximation.
    """
    numeric = df.select_dtypes(include="number")
    missing_pattern = df.isna().any(axis=1)

    complete = numeric[~missing_pattern]
    incomplete = numeric[missing_pattern]

    if len(complete) == 0 or len(incomplete) == 0:
        return {"test": "insufficient_data", "conclusion": "cannot_test"}

    # Compare means of complete vs incomplete rows
    from scipy.stats import chi2
    n_vars = len(numeric.columns)
    chi2_stat = 0
    for col in numeric.columns:
        c_mean = complete[col].mean()
        i_mean = incomplete[col].dropna().mean()
        pooled_var = numeric[col].var()
        if pooled_var > 0:
            chi2_stat += (c_mean - i_mean) ** 2 / pooled_var

    dof = n_vars
    p_value = 1 - chi2.cdf(chi2_stat, dof)

    return {
        "chi2_stat": chi2_stat,
        "dof": dof,
        "p_value": p_value,
        "conclusion": "MCAR plausible" if p_value > 0.05 else "NOT MCAR (MAR or MNAR likely)",
    }

def plot_missing_patterns(df: pd.DataFrame) -> plt.Figure:
    """Visualize missing data patterns with missingno."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    msno.matrix(df, ax=axes[0], sparkline=False)
    axes[0].set_title("Missing Data Matrix")
    msno.heatmap(df, ax=axes[1])
    axes[1].set_title("Missingness Correlation")
    plt.tight_layout()
    return fig

# Usage
summary = missing_summary(df)
print(summary)
miss_corr = missingness_correlation(df)
mcar_result = littles_mcar_approximation(df)
print(f"MCAR test: {mcar_result['conclusion']} (p={mcar_result.get('p_value', 'N/A'):.4f})")
```

---

## 6. Automated Profiling

```python
# ydata-profiling (minimal config for large datasets)
from ydata_profiling import ProfileReport

def quick_profile(df: pd.DataFrame, title: str = "EDA Report") -> ProfileReport:
    """Generate profile report with minimal config (fast for large data)."""
    return ProfileReport(
        df,
        title=title,
        minimal=True,  # skip heavy computations
        explorative=True,
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": False},  # slow
            "phi_k": {"calculate": False},  # slow
        },
        missing_diagrams={"bar": True, "matrix": True, "heatmap": True},
        samples={"head": 5, "tail": 5},
    )

def full_profile(df: pd.DataFrame, title: str = "Full EDA Report") -> ProfileReport:
    """Full profile for smaller datasets (< 50K rows)."""
    return ProfileReport(
        df,
        title=title,
        minimal=False,
        explorative=True,
        interactions={"continuous": True},
    )

# Usage
profile = quick_profile(df, title="Sales Data EDA")
profile.to_file("eda_report.html")


# sweetviz comparison report
import sweetviz as sv

def compare_datasets(
    train: pd.DataFrame, test: pd.DataFrame, target: str = None
) -> sv.DataframeReport:
    """Compare train/test distributions with sweetviz."""
    report = sv.compare(
        [train, "Train"],
        [test, "Test"],
        target_feat=target,
    )
    return report

def analyze_single(df: pd.DataFrame, target: str = None) -> sv.DataframeReport:
    """Single dataset analysis with sweetviz."""
    return sv.analyze(df, target_feat=target)

# Usage
report = compare_datasets(train_df, test_df, target="label")
report.show_html("sweetviz_comparison.html")
```

---

## 7. Hypothesis Testing with Effect Sizes

```python
import numpy as np
import pandas as pd
from scipy import stats

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size for two independent groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std

def cramers_v(contingency_table: pd.DataFrame) -> float:
    """Cramér's V effect size for chi-square test."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

def eta_squared(groups: list[np.ndarray]) -> float:
    """Eta-squared effect size for ANOVA."""
    all_data = np.concatenate(groups)
    grand_mean = all_data.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_data - grand_mean) ** 2)
    return ss_between / ss_total

def two_sample_test(
    group1: pd.Series, group2: pd.Series, alpha: float = 0.05
) -> dict:
    """Two-sample comparison with appropriate test selection."""
    g1, g2 = group1.dropna().values, group2.dropna().values

    # Check normality
    normal_1 = stats.shapiro(g1[:5000])[1] > alpha if len(g1) >= 8 else False
    normal_2 = stats.shapiro(g2[:5000])[1] > alpha if len(g2) >= 8 else False
    both_normal = normal_1 and normal_2

    if both_normal:
        # Welch's t-test (does not assume equal variance)
        t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=False)
        test_name = "Welch's t-test"
    else:
        # Mann-Whitney U (non-parametric)
        t_stat, p_value = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        test_name = "Mann-Whitney U"

    d = cohens_d(g1, g2)
    effect_interp = (
        "negligible" if abs(d) < 0.2
        else "small" if abs(d) < 0.5
        else "medium" if abs(d) < 0.8
        else "large"
    )

    return {
        "test": test_name,
        "statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "cohens_d": d,
        "effect_size": effect_interp,
        "n1": len(g1),
        "n2": len(g2),
    }

def chi_square_test(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Chi-square test of independence with Cramér's V."""
    ct = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    v = cramers_v(ct)
    return {
        "test": "Chi-square independence",
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "cramers_v": v,
        "significant": p < 0.05,
    }

def multi_group_test(groups: list[pd.Series], alpha: float = 0.05) -> dict:
    """ANOVA or Kruskal-Wallis for multiple groups."""
    arrays = [g.dropna().values for g in groups]

    # Check normality of all groups
    all_normal = all(
        stats.shapiro(g[:5000])[1] > alpha for g in arrays if len(g) >= 8
    )

    if all_normal:
        stat, p_value = stats.f_oneway(*arrays)
        test_name = "One-way ANOVA"
    else:
        stat, p_value = stats.kruskal(*arrays)
        test_name = "Kruskal-Wallis"

    eta2 = eta_squared(arrays)

    return {
        "test": test_name,
        "statistic": stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "eta_squared": eta2,
        "n_groups": len(arrays),
        "group_sizes": [len(g) for g in arrays],
    }

# Usage
result = two_sample_test(df[df["group"] == "A"]["value"], df[df["group"] == "B"]["value"])
print(f"{result['test']}: p={result['p_value']:.4f}, d={result['cohens_d']:.3f} ({result['effect_size']})")

chi_result = chi_square_test(df, "category", "outcome")
print(f"Chi-square: p={chi_result['p_value']:.4f}, V={chi_result['cramers_v']:.3f}")

groups = [df[df["region"] == r]["sales"] for r in df["region"].unique()]
anova_result = multi_group_test(groups)
print(f"{anova_result['test']}: p={anova_result['p_value']:.4f}, η²={anova_result['eta_squared']:.3f}")
```

---

## 8. Time Series EDA

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt

def stl_decomposition(
    series: pd.Series, period: int = 7, robust: bool = True
) -> dict:
    """STL decomposition into trend, seasonal, residual."""
    stl = STL(series.dropna(), period=period, robust=robust)
    result = stl.fit()
    return {
        "trend": result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
        "seasonal_strength": 1 - result.resid.var() / (result.seasonal + result.resid).var(),
        "trend_strength": 1 - result.resid.var() / (result.trend + result.resid).var(),
    }

def adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test for stationarity."""
    clean = series.dropna()
    result = adfuller(clean, autolag="AIC")
    return {
        "test_statistic": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }

def plot_acf_pacf(series: pd.Series, lags: int = 40) -> plt.Figure:
    """Plot ACF and PACF for lag selection."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title("Autocorrelation Function (ACF)")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title("Partial Autocorrelation Function (PACF)")
    plt.tight_layout()
    return fig

def time_series_report(series: pd.Series, period: int = 7) -> dict:
    """Complete time series EDA report."""
    adf = adf_test(series)
    stl = stl_decomposition(series, period=period)
    return {
        "n_obs": len(series.dropna()),
        "date_range": f"{series.index.min()} to {series.index.max()}",
        "is_stationary": adf["is_stationary"],
        "adf_p_value": adf["p_value"],
        "trend_strength": stl["trend_strength"],
        "seasonal_strength": stl["seasonal_strength"],
    }

# Usage
report = time_series_report(df["daily_sales"], period=7)
print(f"Stationary: {report['is_stationary']} (p={report['adf_p_value']:.4f})")
print(f"Trend strength: {report['trend_strength']:.3f}, Seasonal: {report['seasonal_strength']:.3f}")
fig = plot_acf_pacf(df["daily_sales"], lags=30)
```

---

## 9. Multivariate Analysis

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def pca_analysis(df: pd.DataFrame, n_components: int = None) -> dict:
    """PCA with scree plot data and explained variance."""
    numeric = df.select_dtypes(include="number").dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric)

    if n_components is None:
        n_components = min(X_scaled.shape)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    n_95 = np.argmax(cumulative_var >= 0.95) + 1

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cumulative_var,
        "n_components_95pct": n_95,
        "components": pd.DataFrame(
            pca.components_[:5],
            columns=numeric.columns,
            index=[f"PC{i+1}" for i in range(min(5, n_components))],
        ),
        "transformed": X_pca,
    }

def plot_scree(pca_result: dict) -> plt.Figure:
    """Scree plot with cumulative variance."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    n = len(pca_result["explained_variance_ratio"])
    x = range(1, n + 1)

    ax1.bar(x, pca_result["explained_variance_ratio"], alpha=0.6, label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")

    ax2 = ax1.twinx()
    ax2.plot(x, pca_result["cumulative_variance"], "r-o", markersize=4, label="Cumulative")
    ax2.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Cumulative Explained Variance")

    n95 = pca_result["n_components_95pct"]
    ax2.axvline(x=n95, color="green", linestyle="--", alpha=0.5, label=f"95% at PC{n95}")

    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.title("PCA Scree Plot")
    plt.tight_layout()
    return fig

def tsne_embedding(
    df: pd.DataFrame,
    n_components: int = 2,
    perplexity: float = 30.0,
    labels: pd.Series = None,
) -> plt.Figure:
    """t-SNE visualization for cluster structure discovery."""
    numeric = df.select_dtypes(include="number").dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric)

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42,
        n_iter=1000,
    )
    X_embedded = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    if labels is not None:
        for label in labels.unique():
            mask = labels == label
            ax.scatter(
                X_embedded[mask, 0], X_embedded[mask, 1],
                label=label, alpha=0.6, s=20,
            )
        ax.legend()
    else:
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.5, s=20)

    ax.set_title(f"t-SNE (perplexity={perplexity})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    return fig

# Usage
pca_result = pca_analysis(df)
print(f"Components for 95% variance: {pca_result['n_components_95pct']}")
print(pca_result["components"])

fig = plot_scree(pca_result)
fig = tsne_embedding(df, perplexity=30, labels=df["cluster_label"])
```

---

## When to Use

| Task | Method | When |
|------|--------|------|
| Quick overview | `extended_describe` | First look at any dataset |
| Normality check | Shapiro-Wilk / KS test | Before choosing parametric vs non-parametric tests |
| Feature-target relationship | Spearman + mutual information | Feature selection, non-linear detection |
| Mixed-type correlation | Phi_K | Dataset has both categorical and numeric |
| Univariate outliers | Modified Z-score (MAD) | Robust to existing outliers, unlike standard Z |
| Multivariate outliers | Isolation Forest | High-dimensional, no distribution assumption |
| Density-based outliers | LOF | Clusters of varying density |
| Missing pattern detection | Missingness correlation | Distinguish MCAR vs MAR vs MNAR |
| A/B test significance | `two_sample_test` | Automatic parametric/non-parametric selection |
| Multi-group comparison | ANOVA / Kruskal-Wallis | 3+ groups, with eta-squared effect size |
| Categorical association | Chi-square + Cramér's V | Two categorical variables |
| Stationarity | ADF test | Before time series modeling |
| Seasonality detection | STL decomposition | Quantify trend and seasonal strength |
| Dimensionality | PCA scree plot | Decide n_components for downstream models |
| Cluster structure | t-SNE | Visual exploration before clustering |
| Full automated report | ydata-profiling minimal | Stakeholder-facing HTML report |
| Train/test comparison | sweetviz compare | Detect distribution shift |

---

## Common Gotchas

| Gotcha | Impact | Fix |
|--------|--------|-----|
| Shapiro-Wilk on n > 5000 | Always rejects (too powerful) | Use KS test or visual QQ for large samples |
| Pearson on non-linear relationships | Misses quadratic/exponential patterns | Use Spearman or mutual information |
| IQR outliers on skewed data | Flags too many on one tail | Use modified Z-score or log-transform first |
| Multiple hypothesis testing | Inflated false positive rate | Apply Bonferroni or FDR correction |
| t-SNE perplexity sensitivity | Different structures at different perplexity | Run at 5, 30, 50 and compare |
| PCA on unscaled data | Dominated by high-variance features | Always StandardScaler before PCA |
| ydata-profiling on >100K rows | Extremely slow | Use `minimal=True` or sample first |
| Cohen's d with unequal variance | Misleading pooled SD | Use Glass's delta or Hedges' g |
| Correlation ≠ causation | Spurious correlations in observational data | Use with domain knowledge only |
| Missing data deletion | Biased estimates if not MCAR | Test MCAR assumption first, consider imputation |

---

## References

- **pandas**: https://pandas.pydata.org/docs/
- **scipy.stats**: https://docs.scipy.org/doc/scipy/reference/stats.html
- **scikit-learn outlier detection**: https://scikit-learn.org/stable/modules/outlier_detection.html
- **ydata-profiling**: https://docs.profiling.ydata.ai/
- **sweetviz**: https://github.com/fbdesignpro/sweetviz
- **statsmodels**: https://www.statsmodels.org/stable/index.html
- **missingno**: https://github.com/ResidentMario/missingno
- **phik correlation**: https://github.com/KaveIO/PhiK
- **Cohen (1988)**: *Statistical Power Analysis for the Behavioral Sciences* — effect size conventions
- **Little (1988)**: "A Test of Missing Completely at Random" — *JASA* 83(404)
- **Cleveland et al. (1990)**: "STL: A Seasonal-Trend Decomposition" — *Journal of Official Statistics* 6(1)
