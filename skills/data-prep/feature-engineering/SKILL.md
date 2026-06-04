---
name: feature-engineering
description: Feature engineering — numeric transforms (log, Box-Cox, binning), categorical encoding (target, WoE, one-hot), text vectorization (TF-IDF, embeddings), datetime cyclical encoding, interactions, and automated generation with featuretools/tsfresh. Use when transforming raw data into ML-ready features.
---

# Feature Engineering

Transform raw data into informative ML features through numeric scaling, categorical encoding, text vectorization, temporal decomposition, and automated generation.

---

## Why This Exists

**Problem**: Raw features — timestamps, categorical strings, free text, raw numerics — are rarely in the form a model can learn from efficiently. A linear model given a raw timestamp sees a monotone integer; a tree model given a string category sees nothing it can split on. Without deliberate transformation, models either fail to train, converge slowly, or systematically underfit patterns that are obvious once the right representation is chosen.

**Key insight**: Transforming inputs into meaningful numeric representations is often the highest-leverage step in an ML project — more impactful than hyperparameter tuning or model selection — because it determines what signal is even available for the model to learn.

**Reach for this when**: Building any supervised ML pipeline before model selection; adding domain knowledge (cyclical time encoding, interaction terms, group aggregates) that a model can't discover on its own; or hitting a performance ceiling where the model architecture is sound but the raw features lack discriminative structure.

---

## 1. Numeric Transforms

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    PowerTransformer, QuantileTransformer, KBinsDiscretizer,
    PolynomialFeatures, FunctionTransformer
)

df = pd.DataFrame({
    'revenue': [100, 5000, 200, 80000, 350],
    'age': [25, 45, 30, 60, 35],
    'visits': [1, 50, 3, 200, 10]
})

# Log1p (handles zeros, invertible with expm1)
df['revenue_log1p'] = np.log1p(df['revenue'])

# Square root (milder than log for moderate skew)
df['visits_sqrt'] = np.sqrt(df['visits'])

# Box-Cox (requires strictly positive values)
box_cox = PowerTransformer(method='box-cox')
df[['revenue_boxcox']] = box_cox.fit_transform(df[['revenue']])

# Yeo-Johnson (handles zeros and negatives)
yeo_johnson = PowerTransformer(method='yeo-johnson')
df[['visits_yeojohnson']] = yeo_johnson.fit_transform(df[['visits']])

# Quantile transform (uniform or normal output)
quantile_tf = QuantileTransformer(output_distribution='normal', random_state=42)
df[['revenue_quantile']] = quantile_tf.fit_transform(df[['revenue']])

# Binning (equal-width, equal-frequency, k-means)
binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
df[['age_binned']] = binner.fit_transform(df[['age']])

# Polynomial features (degree=2, interaction_only for just crosses)
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
poly_features = poly.fit_transform(df[['age', 'visits']])
poly_names = poly.get_feature_names_out(['age', 'visits'])
df_poly = pd.DataFrame(poly_features, columns=poly_names)
```

---

## 2. Categorical Encoding

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import category_encoders as ce

df = pd.DataFrame({
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'NYC'],
    'education': ['high_school', 'bachelors', 'masters', 'phd', 'bachelors', 'masters'],
    'target': [1, 0, 1, 1, 0, 1]
})

# One-hot encoding (sparse output, handles unknown categories)
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_matrix = ohe.fit_transform(df[['city']])
ohe_df = pd.DataFrame(ohe_matrix, columns=ohe.get_feature_names_out())

# Ordinal encoding (when order matters)
ordinal = OrdinalEncoder(categories=[['high_school', 'bachelors', 'masters', 'phd']])
df['education_ordinal'] = ordinal.fit_transform(df[['education']])

# Target encoding (mean of target per category, with smoothing)
target_enc = ce.TargetEncoder(cols=['city'], smoothing=1.0)
df['city_target'] = target_enc.fit_transform(df['city'], df['target'])

# Frequency encoding
freq_map = df['city'].value_counts(normalize=True)
df['city_freq'] = df['city'].map(freq_map)

# Binary encoding (hash-like, fewer columns than one-hot)
binary_enc = ce.BinaryEncoder(cols=['city'])
df_binary = binary_enc.fit_transform(df[['city']])

# Weight of Evidence (WoE) -- good for logistic regression / credit scoring
woe_enc = ce.WOEEncoder(cols=['city'])
df['city_woe'] = woe_enc.fit_transform(df['city'], df['target'])

# Leave-one-out encoding (reduces target leakage vs plain target encoding)
loo_enc = ce.LeaveOneOutEncoder(cols=['city'])
df['city_loo'] = loo_enc.fit_transform(df['city'], df['target'])
```

---

## 3. Text Features

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

docs = [
    "machine learning is great for predictions",
    "deep learning uses neural networks",
    "feature engineering improves model accuracy",
    "random forests are ensemble methods"
]

# TF-IDF pipeline (sublinear tf, ngrams, max features)
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
    strip_accents='unicode'
)
tfidf_matrix = tfidf.fit_transform(docs)

# TF-IDF in a classification pipeline
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000))
])
# text_pipeline.fit(X_train_text, y_train)

# Count vectorizer (bag of words, for topic models / Naive Bayes)
count_vec = CountVectorizer(max_features=5000, stop_words='english')
bow_matrix = count_vec.fit_transform(docs)

# Dense embeddings with sentence-transformers (384-dim, fast)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs, show_progress_bar=False)
embed_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
```

---

## 4. Datetime Features

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
    'value': np.random.randn(100).cumsum()
})
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Basic decomposition
df['hour'] = df['timestamp'].dt.hour
df['dow'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)

# Cyclical encoding (preserves proximity: hour 23 is near hour 0)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Lag features
for lag in [1, 3, 7, 24]:
    df[f'value_lag_{lag}'] = df['value'].shift(lag)

# Rolling window statistics
for window in [3, 7, 24]:
    df[f'value_rolling_mean_{window}'] = df['value'].rolling(window).mean()
    df[f'value_rolling_std_{window}'] = df['value'].rolling(window).std()
    df[f'value_rolling_min_{window}'] = df['value'].rolling(window).min()
    df[f'value_rolling_max_{window}'] = df['value'].rolling(window).max()

# Expanding stats (since beginning of series)
df['value_expanding_mean'] = df['value'].expanding().mean()

# Time since event
df['hours_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600

# Holiday flags (US example)
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2024-01-01', end='2024-12-31')
df['is_holiday'] = df['timestamp'].dt.normalize().isin(holidays).astype(int)
```

---

## 5. Interaction and Cross Features

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

df = pd.DataFrame({
    'price': [10, 25, 50, 100, 200],
    'quantity': [100, 50, 20, 10, 5],
    'category': ['A', 'B', 'A', 'C', 'B'],
    'rating': [4.5, 3.2, 4.8, 2.1, 3.9]
})

# Manual interaction features
df['revenue'] = df['price'] * df['quantity']
df['price_per_rating'] = df['price'] / df['rating']
df['log_price_x_qty'] = np.log1p(df['price']) * np.log1p(df['quantity'])

# Ratio features
df['price_to_mean'] = df['price'] / df['price'].mean()

# Group-level stats as features (target-free aggregates)
group_stats = df.groupby('category')['price'].agg(['mean', 'std', 'count'])
group_stats.columns = ['cat_price_mean', 'cat_price_std', 'cat_count']
df = df.merge(group_stats, left_on='category', right_index=True, how='left')

# Deviation from group mean
df['price_dev_from_cat'] = df['price'] - df['cat_price_mean']

# Polynomial interactions (sklearn)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
numeric_cols = ['price', 'quantity', 'rating']
interactions = poly.fit_transform(df[numeric_cols])
interaction_names = poly.get_feature_names_out(numeric_cols)
df_interactions = pd.DataFrame(interactions, columns=interaction_names)
```

---

## 6. Automated Feature Engineering

```python
import numpy as np
import pandas as pd
import featuretools as ft
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

# === Featuretools: Deep Feature Synthesis ===
# Entity setup
customers = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'signup_date': pd.to_datetime(['2023-01-01', '2023-03-15', '2023-06-01']),
    'age': [25, 40, 35]
})
transactions = pd.DataFrame({
    'txn_id': range(10),
    'customer_id': [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
    'amount': [10, 50, 30, 100, 200, 50, 75, 20, 40, 60],
    'timestamp': pd.date_range('2023-06-01', periods=10, freq='D')
})

# Create EntitySet
es = ft.EntitySet(id='ecommerce')
es = es.add_dataframe(
    dataframe_name='customers',
    dataframe=customers,
    index='customer_id',
    time_index='signup_date'
)
es = es.add_dataframe(
    dataframe_name='transactions',
    dataframe=transactions,
    index='txn_id',
    time_index='timestamp'
)
es = es.add_relationship('customers', 'customer_id', 'transactions', 'customer_id')

# Deep Feature Synthesis
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='customers',
    agg_primitives=['mean', 'sum', 'count', 'std', 'max', 'min'],
    trans_primitives=['month', 'weekday', 'hour'],
    max_depth=2
)

# === tsfresh: Time-series feature extraction ===
timeseries_df = pd.DataFrame({
    'id': [1]*50 + [2]*50,
    'time': list(range(50)) * 2,
    'value': np.random.randn(100).cumsum().tolist()
})

# Extract ~800 features per time series
extracted = extract_features(
    timeseries_df,
    column_id='id',
    column_sort='time',
    column_value='value',
    impute_function=impute
)
```

---

## 7. Pipeline Integration

```python
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, FunctionTransformer, PowerTransformer
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Define column groups
numeric_features = ['age', 'income', 'visits']
categorical_features = ['city', 'education']
text_features = 'description'

# Numeric pipeline: impute -> power transform -> scale
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('power', PowerTransformer(method='yeo-johnson')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute -> one-hot
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Text pipeline: TF-IDF
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2)))
])

# Combine all with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features),
        ('text', text_pipeline, text_features),
    ],
    remainder='drop'
)

# Full ML pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Usage:
# scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='roc_auc')
# model_pipeline.fit(X_train, y_train)
# predictions = model_pipeline.predict(X_test)
```

---

## When to Use

| Technique | Best For | Avoid When |
|-----------|----------|------------|
| Log/Box-Cox | Right-skewed numeric (revenue, counts) | Data has zeros (use log1p) or negatives (use Yeo-Johnson) |
| Quantile transform | Non-parametric normalization | Small datasets (<100 samples) |
| Binning | Non-linear relationships, noisy features | Already smooth relationship with target |
| One-hot | Low-cardinality categoricals (<20 levels) | High cardinality (use target/frequency encoding) |
| Target encoding | High-cardinality categoricals | Small data without CV-based regularization |
| WoE | Credit scoring, logistic regression | Multi-class problems |
| TF-IDF | Text classification, sparse text | Need semantic similarity (use embeddings) |
| Sentence embeddings | Semantic search, short texts | Very long documents (chunk first) |
| Cyclical encoding | Hours, days, months (periodic features) | Non-cyclical ordinals |
| Lag/rolling features | Time series forecasting | Cross-sectional (non-temporal) data |
| Polynomial/interactions | Known feature interactions | High dimensionality (causes explosion) |
| featuretools DFS | Relational data with multiple tables | Single flat table |
| tsfresh | Time-series classification/regression | Real-time inference (slow extraction) |

---

## Common Gotchas

1. **Target leakage**: Always fit encoders (target, WoE) on train split only. Use `fit_transform` on train, `transform` on test.
2. **Box-Cox requires positive values**: Add a constant or use Yeo-Johnson instead.
3. **One-hot explosion**: 1000+ categories → use target encoding, hashing, or embeddings.
4. **Cyclical encoding needs both sin AND cos**: Using only sin loses information (sin(0) = sin(π)).
5. **Rolling features create NaNs**: First `window-1` rows are NaN. Decide: drop, backfill, or use expanding.
6. **tsfresh is slow**: Extract on training data, use `select_features()` to prune, apply same selected set to test.
7. **Frequency encoding is unstable**: Train/test frequency distributions differ. Use train frequencies for both.
8. **QuantileTransformer memorizes training distribution**: With few samples, test transforms are unreliable.
9. **PolynomialFeatures scales quadratically**: degree=2 with 50 features → 1,326 output columns. Use `interaction_only=True` or manual selection.
10. **Pipeline ordering matters**: Impute before encoding/scaling. Scale after encoding (not before one-hot).

---

## References

- sklearn preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
- category_encoders: https://contrib.scikit-learn.org/category_encoders/
- featuretools: https://github.com/alteryx/featuretools
- tsfresh: https://tsfresh.readthedocs.io/
- sentence-transformers: https://www.sbert.net/
- Feature Engineering and Selection (Max Kuhn): http://www.feat.engineering/
- sklearn ColumnTransformer: https://scikit-learn.org/stable/modules/compose.html
- sklearn pipelines: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
