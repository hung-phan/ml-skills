---
name: experiment-tracking
description: Experiment tracking and model registry patterns for reproducible ML — MLflow, Weights & Biases, comparison, and best practices
---

# Experiment Tracking & Model Registry

## Why This Exists

Without systematic experiment tracking:
- You can't reproduce results from 3 weeks ago ("which learning rate gave 94% accuracy?")
- You can't compare runs across hyperparameter sweeps
- You lose track of which model checkpoint is deployed in production
- Collaborators can't build on your work without re-running everything
- Model lineage is invisible — no one knows which data/code produced which artifact

Experiment tracking solves this by recording every run's parameters, metrics, artifacts, and code version automatically. Model registries extend this by managing the lifecycle of trained models (staging → production → archived).

## MLflow

### Core Concepts

| Concept | Purpose |
|---------|---------|
| **Experiment** | Named group of runs (e.g., "bert-finetuning-v2") |
| **Run** | Single training execution with params, metrics, artifacts |
| **Model Registry** | Central store for versioned models with stage transitions |
| **Artifact** | Any file logged with a run (checkpoints, plots, data samples) |

### Tracking API

```python
import mlflow

# Set experiment (creates if not exists)
mlflow.set_experiment("sentiment-classifier")

with mlflow.start_run(run_name="lr-sweep-001"):
    # Log hyperparameters
    mlflow.log_param("learning_rate", 3e-5)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 10)
    mlflow.log_params({"optimizer": "adamw", "weight_decay": 0.01})

    # Log metrics (step-aware for curves)
    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }, step=epoch)

    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("config.yaml")

    # Log model with signature
    from mlflow.models import infer_signature
    sig = infer_signature(sample_input, model(sample_input))
    mlflow.pytorch.log_model(model, "model", signature=sig)
```

### Autolog (Zero-Code Tracking)

```python
# Sklearn — logs params, metrics, model, feature importance
mlflow.sklearn.autolog()
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)  # Everything logged automatically

# PyTorch Lightning
mlflow.pytorch.autolog()

# HuggingFace Transformers
mlflow.transformers.autolog()

# XGBoost
mlflow.xgboost.autolog()
```

### Model Registry

```python
# Register model from a run
model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri, "sentiment-bert")

# Transition stages
from mlflow import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name="sentiment-bert",
    version=mv.version,
    stage="Production",  # None | Staging | Production | Archived
)

# Load production model
model = mlflow.pyfunc.load_model("models:/sentiment-bert/Production")

# Compare model versions
runs = mlflow.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.val_accuracy > 0.9",
    order_by=["metrics.val_accuracy DESC"],
)
```

### MLflow UI

```bash
# Start tracking server
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns

# Production setup (PostgreSQL + S3)
mlflow server \
  --backend-store-uri postgresql://user:pass@host/mlflow \
  --default-artifact-root s3://mlflow-artifacts/ \
  --host 0.0.0.0 --port 5000
```

## Weights & Biases (W&B)

### Core API

```python
import wandb

# Initialize run
run = wandb.init(
    project="sentiment-classifier",
    name="bert-lr3e5-bs32",
    config={
        "learning_rate": 3e-5,
        "batch_size": 32,
        "architecture": "bert-base",
        "dataset": "imdb-50k",
        "epochs": 10,
    },
    tags=["baseline", "bert", "production-candidate"],
)

# Log metrics (step auto-increments)
for epoch in range(10):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    wandb.log({
        "train/loss": train_loss,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "epoch": epoch,
        "learning_rate": scheduler.get_last_lr()[0],
    })

# Log media
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    y_true=labels, preds=preds, class_names=class_names
)})
wandb.log({"predictions": wandb.Table(
    columns=["text", "true", "pred", "confidence"],
    data=sample_predictions
)})

# Save model artifact
artifact = wandb.Artifact("sentiment-model", type="model")
artifact.add_file("model.pt")
run.log_artifact(artifact)

run.finish()
```

### Sweeps (Hyperparameter Search)

```python
# Define sweep config
sweep_config = {
    "method": "bayes",  # grid, random, bayes
    "metric": {"name": "val/accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
        "batch_size": {"values": [16, 32, 64]},
        "weight_decay": {"distribution": "uniform", "min": 0.0, "max": 0.1},
        "warmup_steps": {"distribution": "int_uniform", "min": 0, "max": 500},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 3},
}

# Launch sweep
sweep_id = wandb.sweep(sweep_config, project="sentiment-classifier")

def train():
    run = wandb.init()
    config = wandb.config
    model = build_model(lr=config.learning_rate, wd=config.weight_decay)
    for epoch in range(10):
        loss, acc = train_epoch(model)
        wandb.log({"val/accuracy": acc, "train/loss": loss})

wandb.agent(sweep_id, function=train, count=50)
```

### Dataset Versioning with Artifacts

```python
# Log a dataset version
artifact = wandb.Artifact("imdb-processed", type="dataset", metadata={"size": 50000})
artifact.add_dir("data/processed/")
wandb.log_artifact(artifact)

# Use a specific dataset version in training
run = wandb.init()
dataset = run.use_artifact("imdb-processed:v3")
data_dir = dataset.download()
```

## Full Examples

### PyTorch Training Loop + MLflow

```python
import torch
import torch.nn as nn
import mlflow
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

mlflow.set_experiment("image-classifier")

with mlflow.start_run():
    # Log all hyperparameters
    config = {"lr": 1e-4, "epochs": 20, "batch_size": 64, "weight_decay": 0.01}
    mlflow.log_params(config)
    mlflow.set_tag("model_type", "resnet50")
    mlflow.set_tag("dataset", "cifar100")

    model = build_resnet50(num_classes=100).cuda()
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(config["epochs"]):
        # Train
        model.train()
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)

        train_loss = running_loss / len(train_loader)
        val_acc = correct / total
        scheduler.step()

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "learning_rate": scheduler.get_last_lr()[0],
        }, step=epoch)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            mlflow.log_artifact("best_model.pt")
            mlflow.log_metric("best_val_accuracy", best_acc)

    # Register the best model
    mlflow.pytorch.log_model(model, "model")
```

### HuggingFace Trainer + W&B

```python
import wandb
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    TrainingArguments, Trainer,
)
from datasets import load_dataset

wandb.init(project="text-classification", tags=["bert", "imdb"])

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("imdb")

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="wandb",  # This is all you need
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()},
)

trainer.train()
wandb.finish()
```

### Optuna + MLflow (Hyperparameter Optimization)

```python
import optuna
import mlflow

mlflow.set_experiment("optuna-hpo")

def objective(trial):
    with mlflow.start_run(nested=True):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        n_layers = trial.suggest_int("n_layers", 2, 6)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)

        mlflow.log_params({"lr": lr, "batch_size": batch_size,
                           "n_layers": n_layers, "dropout": dropout})

        model = build_model(n_layers=n_layers, dropout=dropout)
        val_acc = train_and_evaluate(model, lr=lr, batch_size=batch_size)

        mlflow.log_metric("val_accuracy", val_acc)
        return val_acc

with mlflow.start_run(run_name="optuna-sweep"):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=100, timeout=3600)

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_val_accuracy", study.best_value)
```

## Comparison Table

| Feature | MLflow | Weights & Biases | Neptune | TensorBoard |
|---------|--------|-----------------|---------|-------------|
| **Hosting** | Self-hosted or Databricks | Cloud (SaaS) | Cloud (SaaS) | Local / TensorBoard.dev |
| **Pricing** | Free (OSS) | Free tier (100GB), Team $50/user/mo | Free tier, Team $79/user/mo | Free |
| **Model Registry** | ✅ Built-in (stages) | ✅ (Model Registry) | ✅ (Model Registry) | ❌ |
| **Autolog** | ✅ sklearn, pytorch, tf, xgb, transformers | ✅ via integrations | ✅ via integrations | ❌ Manual only |
| **HPO Sweeps** | ❌ (use Optuna/Ray Tune) | ✅ Built-in Bayesian | ❌ (use Optuna) | ❌ |
| **Dataset Versioning** | ✅ (Artifacts) | ✅ (Artifacts) | ✅ (Artifacts) | ❌ |
| **Collaboration** | Basic (shared server) | ✅ Teams, reports, comments | ✅ Teams, workspaces | ❌ |
| **Offline Mode** | ✅ (local files) | ✅ (wandb offline) | ✅ (offline mode) | ✅ (local logs) |
| **UI Quality** | Good (functional) | Excellent (polished) | Very good | Good (scalars/graphs) |
| **Custom Dashboards** | ❌ | ✅ Reports, panels | ✅ Custom views | Limited |
| **Data Privacy** | ✅ Full control (self-host) | ⚠️ Cloud default, private cloud available | ⚠️ Cloud default | ✅ Local |
| **Git Integration** | ✅ (code version logged) | ✅ (code saving, diff) | ✅ (source code) | ❌ |
| **Best For** | Enterprise/self-hosted, MLOps pipelines | Research teams, rapid experimentation | Large teams needing structure | Quick local visualization |

### When to Choose What

- **MLflow**: You need self-hosting, Databricks integration, or a production model registry with CI/CD
- **W&B**: You want the best experiment comparison UI, built-in sweeps, and team collaboration
- **Neptune**: Enterprise teams needing granular access control and structured metadata
- **TensorBoard**: Quick local debugging, don't need persistence or collaboration

## Best Practices

### What to Log

| Category | What to Log | Why |
|----------|-------------|-----|
| **Hyperparameters** | LR, batch size, optimizer, scheduler, architecture, seed | Reproducibility |
| **Training Curves** | train_loss, val_loss, val_metric per step/epoch | Diagnose overfitting |
| **Learning Rate** | Actual LR at each step (from scheduler) | Debug warmup/decay issues |
| **Gradients** | Gradient norm (global), per-layer norms | Detect vanishing/exploding |
| **Sample Predictions** | 10-20 examples with true/pred labels | Qualitative error analysis |
| **Confusion Matrix** | At end of training and at checkpoints | Class imbalance visibility |
| **System Metrics** | GPU utilization, memory, throughput (samples/sec) | Optimization opportunities |
| **Data Info** | Dataset version, split sizes, class distribution | Data drift detection |
| **Model Size** | Parameter count, checkpoint size | Deployment constraints |
| **Environment** | Python version, package versions, git hash | Exact reproducibility |

### Run Organization

```python
# Use structured naming
wandb.init(
    project="sentiment-v2",          # One project per task
    group="bert-experiments",         # Group related runs
    name="bert-base-lr3e5-bs32",     # Descriptive run name
    tags=["baseline", "bert", "v2"], # Searchable tags
    notes="Testing lower LR with cosine schedule",
)

# MLflow equivalent
mlflow.set_experiment("sentiment-v2")
with mlflow.start_run(run_name="bert-base-lr3e5-bs32"):
    mlflow.set_tags({
        "group": "bert-experiments",
        "stage": "baseline",
        "model_family": "bert",
    })
```

### Model Versioning Strategy

```
v1.0.0 — Initial production model (baseline)
v1.1.0 — Same architecture, new training data
v1.2.0 — Hyperparameter tuning improvement
v2.0.0 — Architecture change (BERT → DeBERTa)
```

Register models with metadata:
```python
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="sentiment-classifier",
    tags={"task": "binary-sentiment", "framework": "pytorch", "data_version": "v3"},
)
```

### Reproducibility Checklist

```python
import random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Log everything needed to reproduce
mlflow.log_params({
    "seed": 42,
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
})
mlflow.log_artifact("requirements.txt")
mlflow.log_artifact("config.yaml")
```

## Gotchas

1. **MLflow artifact storage grows fast** — Set up artifact garbage collection or use S3 lifecycle policies
2. **W&B free tier has 100GB limit** — Use `wandb.log()` selectively for large projects; avoid logging full datasets
3. **Autolog captures too much** — Disable with `mlflow.sklearn.autolog(disable=True)` when you need manual control
4. **Nested runs in MLflow** — Use `nested=True` for HPO trials, otherwise runs overwrite each other
5. **W&B offline mode** — Set `WANDB_MODE=offline` for air-gapped environments, then `wandb sync` later
6. **Model registry stage transitions** — Always add a description explaining WHY a model moved to Production
7. **TensorBoard scalars don't survive** — Logs are ephemeral local files; use MLflow/W&B for anything you need to keep
8. **Step vs epoch confusion** — Be consistent: log per-step for training loss, per-epoch for validation metrics
9. **Large artifacts block runs** — Log checkpoints asynchronously or only log the best checkpoint
10. **Concurrent runs** — MLflow uses `mlflow.start_run()` context managers; don't share runs across threads

## References

1. [MLflow Documentation](https://mlflow.org/docs/latest/index.html) — Official docs, tracking API, model registry
2. [MLflow Tutorials](https://mlflow.org/docs/latest/tutorials-and-examples/index.html) — End-to-end examples
3. [Weights & Biases Documentation](https://docs.wandb.ai/) — Experiments, sweeps, artifacts, reports
4. [W&B Integrations](https://docs.wandb.ai/guides/integrations) — PyTorch, HuggingFace, Lightning, Keras
5. [Neptune Documentation](https://docs.neptune.ai/) — Experiment tracking, model registry, integrations
6. [Optuna + MLflow Integration](https://optuna.readthedocs.io/en/stable/reference/integration.html) — HPO with tracking
7. [HuggingFace + W&B Guide](https://docs.wandb.ai/guides/integrations/huggingface) — Trainer integration
