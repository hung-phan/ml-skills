---
name: ml-libraries
description: ML library skills — data manipulation, visualization, deep learning frameworks, LLM tooling, inference serving, and distributed computing. Use when working with any ML Python library.
---

# ML Libraries

| Skill | Description |
|-------|-------------|
| [dspy](dspy/) | Declarative LLM pipelines — modules, optimizers, async training, evaluation |
| [huggingface](huggingface/) | Transformers, datasets, tokenizers, PEFT, Accelerate, Hub |
| [keras](keras/) | Neural networks, Sequential/Functional API, multi-backend (JAX, PyTorch, TF) |
| [litellm](litellm/) | Unified LLM API — 100+ providers, routing, fallbacks, caching, proxy |
| [numpy](numpy/) | Arrays, broadcasting, linear algebra, vectorization, numerical computing |
| [pandas](pandas/) | DataFrames, groupby, merge, pivot, read_csv, parquet, memory optimization |
| [plotly](plotly/) | Interactive charts, 3D plots, animations, Dash dashboards |
| [polars](polars/) | Fast DataFrames, lazy evaluation, scan_parquet, streaming, high-performance ETL |
| [pytorch](pytorch/) | nn.Module, autograd, DataLoader, DDP, custom training loops |
| [ray](ray/) | Distributed computing — tasks, actors, data streaming, model serving, tuning |
| [scikit-learn](scikit-learn/) | ML pipelines, preprocessing, model selection, ensembles, calibration |
| [seaborn](seaborn/) | Statistical visualizations, heatmaps, pairplots, publication-quality static plots |
| [sglang](sglang/) | LLM serving with RadixAttention prefix caching, compressed FSM, frontend DSL |
| [triton-inference-server](triton-inference-server/) | Multi-framework model serving — dynamic batching, ensembles, K8s deployment |
| [vllm](vllm/) | LLM inference engine — PagedAttention, continuous batching, structured output |
| [xgboost](xgboost/) | Gradient boosting (XGBoost, LightGBM, CatBoost) for tabular data |

## When to Use What

| Need | Library |
|------|---------|
| Tabular data <10GB | pandas |
| Tabular data, speed critical | polars |
| Numeric arrays, math | numpy |
| Static plots for papers | seaborn |
| Interactive/web plots | plotly |
| Quick prototyping DL | keras |
| Research / custom DL | pytorch |
| Distribute across cluster | ray |
| Classical ML pipelines | scikit-learn |
| Tabular data, structured | xgboost |
| Pretrained model fine-tuning | huggingface |
| LLM pipeline optimization | dspy |
| Call any LLM with one API | litellm |
| Serve LLM (general default) | vllm |
| Serve LLM (multi-turn, structured output) | sglang |
| Serve multiple model types (not just LLM) | triton-inference-server |
