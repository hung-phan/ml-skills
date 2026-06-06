---
name: ray-tune
description: Distributed hyperparameter tuning with search algorithms and early stopping. Covers Tuner API, search spaces, Optuna/HyperOpt, ASHA/PBT schedulers, and Ray Train integration. Use when tuning learning rate, architecture, or any hyperparameters across many trials in parallel.
---

# Ray Tune

- **Docs**: https://docs.ray.io/en/latest/tune/index.html
- **API**: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.html
- **Examples**: https://docs.ray.io/en/latest/tune/examples/index.html

## Why This Exists

**Problem**: Manual hyperparameter search is sequential and wasteful — running 50 trials one-at-a-time burns days of GPU time, and naive grid search scales exponentially while ignoring promising regions of the search space.

**Key insight**: Ray Tune parallelizes trials across a cluster and applies early-stopping schedulers (ASHA) or intelligent search algorithms (Optuna/TPE) to kill bad trials fast and concentrate compute on the most promising configurations.

**Reach for this when**: You have more than ~10 hyperparameters to explore, training runs take longer than a few minutes (making early stopping worthwhile), or you want Optuna/HyperOpt search algorithms with distributed parallelism — prefer it over Optuna standalone once you need multi-GPU or multi-node trials.

## Core API

```python
from ray import tune
from ray.tune import TuneConfig, RunConfig

tuner = tune.Tuner(
    trainable,                    # function or class
    param_space={"lr": tune.loguniform(1e-5, 1e-1)},
    tune_config=TuneConfig(
        metric="val_loss",
        mode="min",
        num_samples=50,           # total trials
        search_alg=...,           # optional
        scheduler=...,            # optional
    ),
    run_config=RunConfig(
        storage_path="s3://bucket/tune-results",
        name="my-experiment",
    ),
)
results = tuner.fit()
best = results.get_best_result(metric="val_loss", mode="min")
print(best.config)  # best hyperparameters
```

## Search Space

| Function | Use Case | Example |
|----------|----------|---------|
| `tune.uniform(lo, hi)` | Float range | `tune.uniform(0.0, 1.0)` |
| `tune.loguniform(lo, hi)` | Log-scale float (learning rate) | `tune.loguniform(1e-5, 1e-1)` |
| `tune.randint(lo, hi)` | Integer [lo, hi) | `tune.randint(16, 256)` |
| `tune.choice(list)` | Categorical | `tune.choice(["adam", "sgd"])` |
| `tune.grid_search(list)` | Exhaustive | `tune.grid_search([32, 64, 128])` |
| `tune.quniform(lo, hi, q)` | Quantized float | `tune.quniform(0.1, 1.0, 0.1)` |
| `tune.sample_from(fn)` | Custom | `tune.sample_from(lambda _: 2**np.random.randint(4,8))` |

## Example: PyTorch + ASHA Early Stopping

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
import tempfile, torch

def train_mnist(config):
    model = build_model(config["hidden"], config["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(100):
        loss = train_epoch(model, optimizer)
        val_acc = evaluate(model)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(model.state_dict(), f"{tmp}/model.pt")
            tune.report(
                {"val_acc": val_acc, "loss": loss},
                checkpoint=Checkpoint.from_directory(tmp),
            )

tuner = tune.Tuner(
    train_mnist,
    param_space={
        "lr": tune.loguniform(1e-4, 1e-1),
        "hidden": tune.choice([64, 128, 256]),
        "dropout": tune.uniform(0.1, 0.5),
    },
    tune_config=tune.TuneConfig(
        metric="val_acc",
        mode="max",
        num_samples=50,
        scheduler=ASHAScheduler(
            max_t=100,           # max epochs
            grace_period=10,     # min epochs before stopping
            reduction_factor=2,  # halve trials at each rung
        ),
    ),
)
results = tuner.fit()
```

## Schedulers

| Scheduler | Best For | Key Idea |
|-----------|----------|----------|
| **ASHA** | Large search, limited budget | Aggressively kills bad trials early |
| **PBT** | Long training (LLM fine-tuning) | Mutates hyperparams mid-training from top performers |
| **MedianStopping** | Conservative early stopping | Stops below-median trials |
| **BOHB** | Bayesian + banding | Combines TPE search with HyperBand scheduling |

### PBT (Population-Based Training)

```python
from ray.tune.schedulers import PopulationBasedTraining

pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=5,      # every 5 iterations
    hyperparam_mutations={
        "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.uniform(0.0, 0.3),
    },
)
```

## Search Algorithms

### Optuna (Recommended Default)

```python
from ray.tune.search.optuna import OptunaSearch

search_alg = OptunaSearch(metric="val_loss", mode="min")

tuner = tune.Tuner(
    trainable,
    param_space={...},
    tune_config=tune.TuneConfig(
        search_alg=search_alg,
        num_samples=100,
    ),
)
```

### HyperOpt (TPE)

```python
from ray.tune.search.hyperopt import HyperOptSearch
search_alg = HyperOptSearch(metric="loss", mode="min")
```

## Integration with Ray Train

```python
from ray.train.torch import TorchTrainer
from ray import tune

trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
)

tuner = tune.Tuner(
    trainer,
    param_space={
        "train_loop_config": {
            "lr": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([16, 32, 64]),
        }
    },
    tune_config=tune.TuneConfig(metric="val_loss", mode="min", num_samples=20),
)
results = tuner.fit()
```

## Tips

| Tip | Why |
|-----|-----|
| Start with `num_samples=20` + ASHA | Fast signal on what matters |
| Use `loguniform` for learning rate | LR spans orders of magnitude |
| Use `choice` for architecture decisions | Discrete, not continuous |
| Set `grace_period` ≥ 5-10 epochs | Don't kill trials before they warm up |
| PBT for expensive training (>1h/trial) | Reuses compute from top performers |
| Always report checkpoints | Enables resume after failure |

## References

- Official docs: https://docs.ray.io/en/latest/tune/index.html
- Key concepts: https://docs.ray.io/en/latest/tune/key-concepts.html
- GitHub: https://github.com/ray-project/ray
