---
name: gradient-free-optimization
description: Gradient-free / black-box optimization for ML — Simulated Annealing, Tabu Search, Genetic Algorithm, Differential Evolution, Particle Swarm Optimization, CMA-ES, OpenAI ES, NSGA-II, Bayesian Optimization (Optuna TPE / GP-BO). Use when objective is non-differentiable or noisy (HPO, NAS, prompt search, RL policy search, combinatorial assignment, simulator-in-the-loop), or SGD plateaus on a multi-modal landscape.
---

# Gradient-Free Optimization for ML

## Why This Exists

SGD/Adam dominate modern ML because most losses are differentiable and gradients give you a cheap, unbiased descent direction. But a lot of real ML work isn't shaped like that:

- **Hyperparameters** are categorical/integer/log-scale and the validation metric is a black box (you can evaluate but not differentiate `learning_rate -> val_loss`).
- **Neural architecture search** picks discrete operators (conv vs attention, kernel size, layer count).
- **Prompt optimization** searches over strings.
- **RL policy search with non-differentiable rewards** (sparse, simulator-only, environment with stochastic dynamics) breaks vanilla policy gradient assumptions.
- **Combinatorial assignment** — routing, scheduling, allocation — has discrete decisions and constraints.
- **SGD plateaus on a multi-modal loss surface** and you need a global search to escape.

In all those cases you can **evaluate** `f(x)` but you **cannot differentiate** it (or the gradient is too noisy/biased to trust). Gradient-free / black-box optimization fills that gap with a small zoo of trajectory-based, population-based, swarm, and model-based methods. Each one shines in a different regime — this skill tells you which to reach for, and gives runnable code.

**Key insight**: you don't need 30 metaheuristics. In practice ~8 carry the load: Simulated Annealing, Tabu, GA, Differential Evolution, PSO, CMA-ES, OpenAI ES, and Bayesian Optimization (GP-BO / TPE). Plus NSGA-II for multi-objective and ACO for routing.

**Reach for this when**:
- you can evaluate but not differentiate the objective,
- you have a discrete or mixed (cont + discrete) search space,
- evaluations are expensive (minutes/hours each) so you want sample-efficient search,
- SGD plateaus and you suspect a multi-modal landscape,
- you need to optimize a *simulator* (robotics, control, RL).

**Don't reach for this when**:
- the loss is differentiable and you have backprop — SGD/Adam will dominate,
- search dimension > ~10K and only a black-box is available — progress is hard; consider surrogate gradients or a bilevel formulation,
- the objective is convex and smooth — use a proper convex solver (CVXPY, scipy.optimize).

---

## Decision Table

Pick by problem shape, *not* by which algorithm sounds coolest.

| Problem shape | Reach for | Why |
|---|---|---|
| Continuous, low-dim (≤ ~50), cheap eval | **CMA-ES** or PSO | CMA-ES adapts a covariance to the loss landscape; gold standard for black-box continuous |
| Continuous, high-dim (1K-1M), parallelizable evals (RL policies) | **OpenAI ES / NES** | Antithetic sampling + linear-time update; scales across hundreds of workers |
| Discrete / mixed search space (HPO with categoricals) | **TPE** (Optuna default) or **GA** | TPE handles categoricals natively; GA handles encodings |
| Single combinatorial trajectory (TSP, scheduling) | **Simulated Annealing** or **Tabu Search** | Cheap, escapes local optima with the right cooling / aspiration |
| Combinatorial routing / assignment with construction phase | **Ant Colony Optimization** | Pheromone trails encode partial-tour quality |
| Expensive evals (minutes / hours each) | **Bayesian Optimization** (GP-BO for cont, TPE for mixed) | Sample-efficient — surrogate-driven |
| Multi-objective / Pareto front | **NSGA-II** (pymoo) | Non-dominated sort + crowding distance |
| Noisy / stochastic objective (RL, sim) | **CMA-ES** with reevaluation, **ES** | Population-based methods average out noise |
| Constrained search | **CMA-ES with constraint handling**, NSGA-II, or penalty-based GA | All accept constraint penalties |
| HPO inside a Python training loop | **Optuna** (TPE / CMA-ES / NSGAII samplers) | Best ergonomics, distributed-friendly |
| HPO across a cluster | **Ray Tune** + Optuna or HyperOpt | Schedules and parallelizes trials |

If you're not sure: **start with Optuna's TPE sampler**. It works for continuous, integer, categorical, and log-scaled parameters out of the box, parallelizes via SQLite/Postgres storage, supports pruning, and gives you the right thing for most HPO tasks.

---

## Taxonomy

Four families. You'll mix-and-match in practice.

### 1. Trajectory-based (S-metaheuristics)

A single solution moves through the search space step by step. Cheap per iteration. Risk: getting stuck in local optima — escaped by accepting worsening moves probabilistically (SA) or remembering recent moves to forbid revisits (Tabu).

- **Hill Climbing** — accept only improving moves. Simple, gets stuck.
- **Simulated Annealing** — accept worsening moves with probability `exp(-Δf / T)`; cool `T` over time.
- **Tabu Search** — short-term memory list of forbidden moves; aspiration criterion lets you break tabu if the move beats global best.

### 2. Population-based / evolutionary (P-metaheuristics)

A population of candidate solutions evolves via selection, recombination, and mutation. Naturally explores in parallel; robust to noise; embarrassingly parallel evaluation.

- **Genetic Algorithm (GA)** — encoding (binary, gray, real, permutation) + selection (tournament, roulette) + crossover + mutation + elitism. Variants: real-coded GA (BGA), permutation GA, multi-objective (NSGA-II).
- **Differential Evolution (DE)** — perturb each candidate by scaled difference of two others; binomial or exponential crossover. Strategies: `rand/1/bin`, `best/2/exp`.
- **CMA-ES** — Covariance Matrix Adaptation Evolution Strategy. The de-facto best black-box continuous optimizer for moderate dim. Maintains a Gaussian over the search space, updates mean and covariance from elite samples.
- **OpenAI ES / Natural Evolution Strategies** — antithetic perturbations of a single mean, weighted-sum gradient estimate. Used by Salimans et al. (2017) for RL policy search at scale.

### 3. Swarm intelligence

Population of agents with local rules; collective behavior emerges. Distinct from GA in that there's no recombination — agents update positions/velocities based on neighbors.

- **Particle Swarm Optimization (PSO)** — each particle has position + velocity; updated toward personal best (cognitive) and swarm/neighborhood best (social). Variants: Global-best PSO, Local-best PSO (ring topology), Binary PSO (discrete), Adaptive PSO.
- **Ant Colony Optimization (ACO)** — agents construct solutions edge by edge; deposit pheromone proportional to solution quality; pheromone evaporates. Strong for routing / TSP / scheduling.
- **Artificial Bee Colony (ABC)** — employed bees / onlookers / scouts foraging metaphor; continuous and discrete variants.
- **Firefly, Bat, Cuckoo, Grey Wolf** — many more nature-inspired variants exist; most don't outperform the above on real benchmarks, despite the marketing.

### 4. Bayesian / model-based

Build a probabilistic surrogate of the objective and pick the next eval to maximize an acquisition function (expected improvement, UCB). Sample-efficient; the right choice when each evaluation is expensive.

- **GP-BO (Gaussian Process Bayesian Optimization)** — best for continuous, low-dim (≤ ~20). Implementations: BoTorch, scikit-optimize, GPyOpt, Ax.
- **TPE (Tree-structured Parzen Estimator)** — kernel density over good vs bad trials. Handles mixed search spaces. Optuna's default sampler. Cheaper than GP-BO at high trial counts.
- **SMAC** — random forests as surrogate. Strong for HPO with categoricals; AutoML and HyperBand foundations.

---

## Algorithms — Just Enough Detail to Use Them

### Simulated Annealing

```text
T = T0
while T > Tf and n < N:
    propose neighbor x'
    Δf = f(x') - f(x)
    if Δf < 0 (or Δf > 0 and random() < exp(-Δf / T)):
        x = x'
    T = cool(T)        # linear, geometric (T *= α), or log
```

- Cooling schedules: **geometric** `T_{i+1} = α * T_i` with `α ∈ [0.85, 0.99]` is the practical default. **Linear** `T_{i+1} = T_i - β`. **Logarithmic** `T_i = T_0 / log(1+i)` is provably convergent but glacially slow.
- Initial `T_0`: pick so initial bad-move acceptance is ~80% (run a short pilot).
- When `T → 0`, SA reduces to hill climbing.
- Use case: TSP, Sudoku, layout, scheduling, simulator parameter tuning. `scipy.optimize.dual_annealing` is a solid out-of-the-box implementation.

### Tabu Search

- Maintain a **tabu list** of recently used moves (or attributes), forbidding revisits.
- **Aspiration criterion**: override tabu if the move would beat the global best.
- Short-term memory (recent moves) + long-term memory (frequency of attributes — diversification).
- Use case: constraint satisfaction, vehicle routing, job-shop scheduling.

### Genetic Algorithm

```text
init population P
while not converged:
    parents = select(P)                 # tournament, roulette, rank
    children = crossover(parents)       # one-point, two-point, uniform, BLX-α
    children = mutate(children)         # bit-flip, gaussian, swap
    P = elitism(P, children)            # keep top-k
```

- **Encoding** matters: binary for combinatorial, real-valued for continuous, permutation for routing/scheduling.
- **Selection pressure** trade-off: tournament size 2-5 is standard.
- **Mutation rate** ~ `1/n` (one bit-flip per individual on average).
- Variants: real-coded GA, permutation GA (PMX/OX crossover), island model GA, **NSGA-II** for multi-objective.

### Differential Evolution

```text
for each x_i in population:
    pick distinct r1, r2, r3 != i
    v = x_{r1} + F * (x_{r2} - x_{r3})       # mutation, F ∈ [0.4, 1.0]
    u = crossover(x_i, v, CR)                # CR ∈ [0.5, 0.9]
    if f(u) < f(x_i): x_i = u
```

- Robust on continuous, low-to-moderate dim. Available as `scipy.optimize.differential_evolution`.

### PSO

```text
v_i ← w * v_i + c1 * r1 * (pbest_i - x_i) + c2 * r2 * (gbest - x_i)
x_i ← x_i + v_i
```

- `w` (inertia, ~0.7), `c1` (cognitive, ~1.5), `c2` (social, ~1.5).
- **Velocity clamping** prevents explosion: `|v| ≤ v_max`.
- Topology: **global best** (every particle sees `gbest` — fast, can converge prematurely) vs **local/ring best** (each particle sees only `k` neighbors — slower, more exploration).
- Discrete variant: **Binary PSO** updates a probability of bit=1 from velocity via sigmoid.

### CMA-ES

Maintains an `N(m, σ²C)` distribution; samples `λ` candidates per generation; updates `m` toward weighted mean of the top-`μ` (recombination); updates `C` to elongate along the descent direction; adapts `σ` from the cumulative path length.

- Practically the best black-box optimizer for continuous problems up to a few hundred dim.
- Single hyperparameter you usually tune: initial step size `σ0` (set to ~1/4 of the search range).
- Use `cma` PyPI package or `Optuna` `CmaEsSampler`.
- Hansen (2016) tutorial linked below is the canonical reference.

### OpenAI ES (NES variant)

```text
init θ                        # policy parameters, large
for generation:
    sample ε_1, ..., ε_n ~ N(0, I)             # antithetic: also use -ε_i
    R_i = f(θ + σ * ε_i)                        # rollout in environment
    θ ← θ + α * (1 / (n*σ)) * Σ_i R_i * ε_i    # gradient estimate
```

- Embarrassingly parallel: `n` workers each run one rollout per generation.
- **Rank-shape** `R_i` (centered, normalized ranks) for invariance to reward scale.
- **Antithetic sampling** `±ε_i` halves variance.
- Salimans et al. (2017) showed competitive performance with TRPO/A3C on Atari/MuJoCo using thousands of CPU cores.

### Ant Colony Optimization

For TSP-style problems:

```text
each ant builds a tour:
    p(edge i→j) ∝ τ_ij^α * η_ij^β            # τ pheromone, η = 1/distance
update: τ_ij ← (1-ρ) * τ_ij + Σ_k Δτ_ij^k    # evaporate + reinforce
```

- Knobs: pheromone influence `α`, heuristic influence `β`, evaporation `ρ`.
- Variants: Ant System (AS), Ant Colony System (ACS), Max-Min Ant System (MMAS).
- ACO beats SA on many routing problems; pairs well with local search (2-opt) on the constructed tour.

### Bayesian Optimization

```text
fit surrogate p(f | D)          # GP, RF, or TPE density
acquisition a(x) = EI / UCB / PI    # exploit-explore trade-off
x_next = argmax a(x)
y_next = f(x_next)
D = D ∪ {(x_next, y_next)}
```

- **GP-BO**: `BoTorch` / `scikit-optimize` / `Ax`. Best for continuous low-dim.
- **TPE**: `Optuna`, `HyperOpt`. Best for mixed/categorical search spaces, default for HPO.
- **SMAC3**: random-forest surrogate, native HyperBand integration, AutoML use.

---

## Code: Optuna HPO with CMA-ES Sampler

Most common ML use of GFO. Optuna handles trial scheduling, pruning, distributed storage, and visualization.

```python
import optuna
from optuna.samplers import CmaEsSampler, TPESampler, NSGAIISampler
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)


def objective(trial: optuna.Trial) -> float:
    # Continuous, integer, categorical, and log-scaled all in one search space
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 2, 10)
    lr = trial.suggest_float("learning_rate", 1e-4, 1.0, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    loss = trial.suggest_categorical("loss", ["log_loss", "exponential"])

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=subsample,
        loss=loss,
        random_state=0,
    )
    scores = cross_val_score(clf, X, y, cv=3, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# CMA-ES is great when the search space is mostly continuous
# Falls back to RandomSampler for categorical params automatically
study = optuna.create_study(
    direction="maximize",
    sampler=CmaEsSampler(seed=0),
    storage="sqlite:///hpo.db",  # parallelizable across workers / machines
    study_name="gbm_breast_cancer",
    load_if_exists=True,
)
study.optimize(objective, n_trials=100, n_jobs=4, show_progress_bar=True)

print("best params:", study.best_params)
print("best value:", study.best_value)

# For mixed/categorical-heavy spaces, switch sampler:
# sampler=TPESampler(multivariate=True, group=True)
# For multi-objective (e.g. accuracy + latency):
# study = optuna.create_study(directions=["maximize", "minimize"],
#                              sampler=NSGAIISampler())
```

Run distributed: just point `n_jobs` workers (or separate processes / machines) at the same `storage` URL. Optuna handles trial coordination.

---

## Code: PSO for Neural Network Weight Training (Khamis-style)

When backprop isn't an option (non-differentiable activations, simulator-in-the-loop, hardware constraints), you can train a small NN by treating the flat weight vector as a search problem. PSO finds a reasonable optimum on tiny networks; on real models you'd use ES instead.

```python
# pip install pyswarms scikit-learn
import numpy as np
import pyswarms as ps
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

X_raw, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X_raw)

n_inputs, n_hidden, n_classes = X.shape[1], 10, len(np.unique(y))
# weight vector layout: W1 (n_inputs * n_hidden) | b1 (n_hidden) |
#                       W2 (n_hidden * n_classes) | b2 (n_classes)
dimensions = n_inputs * n_hidden + n_hidden + n_hidden * n_classes + n_classes


def unpack(p: np.ndarray):
    i = 0
    W1 = p[i : i + n_inputs * n_hidden].reshape(n_inputs, n_hidden); i += n_inputs * n_hidden
    b1 = p[i : i + n_hidden]; i += n_hidden
    W2 = p[i : i + n_hidden * n_classes].reshape(n_hidden, n_classes); i += n_hidden * n_classes
    b2 = p[i : i + n_classes]
    return W1, b1, W2, b2


def forward(p: np.ndarray) -> float:
    """Negative log-likelihood loss for one parameter vector."""
    W1, b1, W2, b2 = unpack(p)
    a1 = np.tanh(X @ W1 + b1)
    logits = a1 @ W2 + b2
    # numerically stable softmax
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    nll = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
    return float(nll)


def swarm_loss(positions: np.ndarray) -> np.ndarray:
    """PSO calls with shape (n_particles, dimensions); returns (n_particles,)."""
    return np.array([forward(p) for p in positions])


options = {"w": 0.79, "c1": 0.9, "c2": 0.5}
optimizer = ps.single.GlobalBestPSO(n_particles=150, dimensions=dimensions, options=options)
cost, pos = optimizer.optimize(swarm_loss, iters=500)


def predict(p: np.ndarray) -> np.ndarray:
    W1, b1, W2, b2 = unpack(p)
    return np.argmax(np.tanh(X @ W1 + b1) @ W2 + b2, axis=1)


print(f"final loss: {cost:.4f}, accuracy: {(predict(pos) == y).mean():.3f}")

# Alternatives in PySwarms:
#   ps.single.LocalBestPSO(...)    # ring topology, more exploratory
#   ps.discrete.BinaryPSO(...)     # binary search spaces
```

This mirrors the Khamis Penguins example. Practical takeaways: (1) global-best PSO converges fastest on smooth losses; (2) tanh activation pairs well with PSO because gradients aren't needed; (3) for any model bigger than a few hundred parameters, switch to CMA-ES or backprop.

---

## Code: CMA-ES on a Black-Box Function

Best in class for continuous low-dim black-box optimization.

```python
# pip install cma
import cma
import numpy as np


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


x0 = np.zeros(10)
sigma0 = 0.5

es = cma.CMAEvolutionStrategy(
    x0,
    sigma0,
    {
        "maxiter": 500,
        "popsize": 20,            # default 4 + floor(3 * log(N))
        "bounds": [[-5] * 10, [5] * 10],
        "verbose": -9,
    },
)
es.optimize(rosenbrock)

print("best x:", es.result.xbest)
print("best f:", es.result.fbest)
print("evals:", es.result.evaluations)
```

Tips: set `sigma0` to ~1/4 of the search range; if the optimizer stalls, increase `popsize`; use `BoundaryHandler` for hard constraints; for noisy objectives set `noise_handling=True`.

---

## Code: Genetic Algorithm with DEAP

Useful when you have a custom encoding (permutation, tree, graph) where CMA-ES doesn't apply.

```python
# pip install deap
import random
from deap import base, creator, tools, algorithms

# Maximize a 0/1 knapsack-style sum of weights
N_ITEMS = 50
weights = [random.uniform(0, 1) for _ in range(N_ITEMS)]


def evaluate(individual):
    return (sum(w for w, b in zip(weights, individual) if b),)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.attr_bool,
    n=N_ITEMS,
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1 / N_ITEMS)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=100)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("max", max)
stats.register("avg", lambda v: sum(v) / len(v))

algorithms.eaSimple(
    pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=False
)

print("best individual:", hof[0])
print("best fitness:", hof[0].fitness.values[0])
```

For multi-objective use `creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))` and `algorithms.eaMuPlusLambda` with `tools.selNSGA2`. Or — easier — use `pymoo`:

```python
# pip install pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems import get_problem

problem = get_problem("zdt1")
algorithm = NSGA2(pop_size=100)
res = minimize(problem, algorithm, ("n_gen", 200), seed=1, verbose=False)
print("Pareto front shape:", res.F.shape)  # (n_solutions, n_objectives)
```

---

## Code: OpenAI ES Skeleton for Policy Search

For RL where the reward is non-differentiable (game score, simulator metric). This is the spirit of Salimans et al. (2017).

```python
import numpy as np


def rollout(theta: np.ndarray) -> float:
    """Run one episode in the environment with policy parameters theta.
    Replace with your env (gymnasium / dm_control / custom sim)."""
    # placeholder: pretend reward is -||theta - target||^2
    target = np.zeros_like(theta)
    return -float(np.sum((theta - target) ** 2))


def openai_es(
    f, n_params: int, n_workers: int = 50, sigma: float = 0.1,
    alpha: float = 0.01, n_generations: int = 200, seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n_params) * 0.1

    for gen in range(n_generations):
        # antithetic sampling: half +eps, half -eps
        eps = rng.normal(size=(n_workers // 2, n_params))
        eps_full = np.concatenate([eps, -eps], axis=0)

        # parallelizable: each rollout is independent (use ray / mp.Pool)
        rewards = np.array([f(theta + sigma * e) for e in eps_full])

        # rank-shape rewards: invariant to reward scale, robust to outliers
        ranks = np.argsort(np.argsort(rewards)).astype(float)
        ranks = ranks / (len(ranks) - 1) - 0.5

        # gradient estimate: weighted sum of perturbations
        grad = (ranks[:, None] * eps_full).sum(axis=0) / (n_workers * sigma)
        theta = theta + alpha * grad

        if gen % 20 == 0:
            print(f"gen {gen}: best reward {rewards.max():.4f}")

    return theta


best = openai_es(rollout, n_params=100, n_workers=50, n_generations=100)
```

For real RL: parallelize rollouts with Ray; use shared seeds + index-based reconstruction so workers only exchange `(seed_index, reward)` not full perturbation vectors (the trick that lets ES scale to thousands of cores in the original paper).

---

## Application Patterns in Modern ML

### Hyperparameter Optimization

The dominant use of GFO in ML. Recommended stack by trial budget:

| Trial budget | Stack |
|---|---|
| ≤ 50 trials, expensive evals | GP-BO via Ax / BoTorch / scikit-optimize |
| 50-1000 trials, mixed search space | **Optuna with TPE** (default) — or CMA-ES if mostly continuous |
| 1000+ trials, distributed | **Ray Tune + Optuna** + ASHA/HyperBand pruning |
| Categorical-heavy, AutoML pipelines | SMAC3 |
| Multi-objective (acc + latency + size) | Optuna `NSGAIISampler` |

Pair with **early stopping** (`HyperbandPruner`, `MedianPruner`) — most trials are bad, kill them at iteration 10% rather than 100%.

### Neural Architecture Search

- **AmoebaNet** (Real et al. 2019) — evolutionary NAS that beat RL-based controllers on ImageNet.
- **PSO-NAS / GA-NAS** — evolve cell topologies as discrete genomes.
- **NAS-Bench-101 / 201** — pre-tabulated benchmarks; use these to evaluate your search algorithm without training each candidate from scratch.
- For practitioner use today: gradient-based NAS (DARTS, ProxylessNAS) is more efficient when applicable; reach for evolutionary NAS for non-differentiable architecture choices (mixed precision, hardware-aware).

### Prompt Optimization

Discrete optimization over strings. Methods:
- **APE** (Automatic Prompt Engineer) — LLM proposes prompts, scores via task accuracy, picks the best.
- **DSPy** — bootstrap-fewshot + MIPROv2 do search over instructions and demonstrations using TPE-style proposal-and-score.
- **PromptBreeder** — evolutionary prompt mutation and crossover with LLM as the mutation operator.

If your search space is "set of discrete tokens" with no natural distance metric, GA with LLM-based mutation operators is the practical default.

### RL / Policy Search Without Gradients

When rewards are sparse, non-differentiable, or backprop through the environment is infeasible:
- **OpenAI ES** (Salimans et al. 2017) — competitive with TRPO/A3C, scales to thousands of CPU cores.
- **CMA-ES for control** — works well for low-dim controllers (PID, MPC residuals, small policies up to ~thousands of params).
- **Augmented Random Search (ARS)** — even simpler than ES; finite differences along Gaussian directions.
- **Population-Based Training (PBT)** — DeepMind's hybrid: each worker does SGD; periodically copy weights from better workers and perturb hyperparameters.

### Combinatorial Assignment / Scheduling / Routing

- **Vehicle routing**: ACO + 2-opt local search; or Google OR-Tools (CP-SAT) for exact small instances.
- **Job-shop scheduling**: Tabu search; or constraint programming.
- **Feature subset selection**: GA with bitmask encoding (see `../feature-selection/`).
- **Sensor placement**: GA or submodular greedy.

For pure combinatorial problems, *check first whether an exact solver (MILP, CP-SAT, SAT) handles your instance size before reaching for metaheuristics*. CP-SAT solves problems people used to throw GAs at.

### Simulator-in-the-loop / Robotics

- Robot morphology + controller co-evolution: GA with a real-valued/structured genome.
- Sim-to-real domain randomization parameter tuning: BO or CMA-ES.
- Each evaluation runs a sim — so you want sample efficiency. BO if eval is the dominant cost; CMA-ES if eval is cheap and the problem is high-dim continuous.

---

## Pitfalls & Anti-Patterns

- **Don't pretend metaheuristics replace SGD**. If the loss is differentiable and you have backprop, gradient methods will dominate by orders of magnitude. GFO is for the boundary cases.
- **Don't run 2000 trials of TPE on a search space where every trial takes 8 hours**. That's a 2-year run. Switch to BO with early-stopping pruners or switch to a cheaper proxy.
- **Don't use vanilla GA on continuous problems** when CMA-ES exists. CMA-ES dominates real-coded GA on virtually every continuous benchmark.
- **Don't tune the metaheuristic's own hyperparameters** for hours. Default settings (Optuna defaults, CMA-ES defaults) are well-chosen. Spend that time enlarging the search space or improving the eval signal instead.
- **Don't forget noise handling**. If `f(x)` is stochastic (RL reward, sim variance), `CmaEsSampler` and `cma` accept reevaluation; OpenAI ES is naturally noise-robust through population averaging; vanilla SA is *not* — wrap evals in averaged batches.
- **Don't ignore the no-free-lunch theorem**. Different algorithms exploit different structure. If your "novel firefly-bat-grey-wolf hybrid" beats CMA-ES on your benchmark, suspect overfitting to the benchmark before celebrating.
- **Don't trust GA papers that compare against weak baselines**. The relevant baseline for continuous problems is CMA-ES; for HPO it's TPE (or random search with HyperBand, which is a shockingly hard baseline to beat — Li et al. 2017).
- **Don't manually search 4 hyperparameters in a notebook**. Wire up Optuna in 10 lines. You'll find better configs and have a study artifact to reproduce later.
- **Don't skip parallelism**. Most GFO is embarrassingly parallel — Ray Tune / Optuna `n_jobs` / OpenAI ES with workers turns wall-clock hours into minutes.
- **Don't optimize on training set**. Use cross-validation or a held-out validation split inside the objective. Otherwise you'll overfit your HPO.

---

## Practical Recipe — "Which one do I run today?"

1. **Differentiable loss?** → SGD/Adam. Stop reading.
2. **HPO with mixed types and ≤ 1000 trials?** → Optuna + TPE.
3. **HPO with > 1000 trials, distributed cluster?** → Ray Tune + Optuna + ASHA pruning.
4. **HPO with very expensive evals (≥ 10 min/trial)?** → BO via Ax/BoTorch (continuous) or Optuna + TPE (mixed).
5. **Continuous black-box, dim ≤ 100, no constraints?** → CMA-ES via `cma` or Optuna's `CmaEsSampler`.
6. **RL policy search, non-differentiable reward, big cluster?** → OpenAI ES. ARS as a baseline.
7. **TSP / VRP / job-shop?** → Try Google OR-Tools first (it'll surprise you); fall back to ACO + 2-opt or Tabu.
8. **Multi-objective?** → NSGA-II via pymoo, or Optuna multi-objective.
9. **Architecture search?** → DARTS / ProxylessNAS if differentiable; AmoebaNet-style evolutionary NAS otherwise; or NAS-Bench for research.
10. **Prompt optimization?** → DSPy MIPROv2 or APE-style search; not the original 1990s GA.

---

## See Also

- `../training-workflow/` — Optuna for HPO is the dominant use of GFO; the deeper HPO walkthrough lives there.
- `../feature-selection/` — wrapper-based feature selection (RFE, GA-based subset search) overlaps with combinatorial GFO.
- `../experiment-tracking/` — log every trial; HPO studies are useless without a proper experiment log.
- `../inference-optimization/` — multi-objective HPO (accuracy vs latency) for production models.
- `../../ml-architectures/reinforcement-learning/` — ES / CMA-ES as alternatives to policy gradient when rewards are non-differentiable.
- `../../ml-architectures/ann/` — for neural-network weight-space optimization context.
- `../../ml-libraries/ray/` — Ray Tune for parallelizing HPO trials across a cluster.

---

## References

- Optuna documentation — https://optuna.readthedocs.io/en/stable/
- pycma (CMA-ES reference implementation) — https://github.com/CMA-ES/pycma
- PySwarms (PSO library) — https://github.com/ljvmiranda921/pyswarms
- DEAP (evolutionary algorithms) — https://github.com/DEAP/deap
- pymoo (multi-objective and many-objective optimization) — https://github.com/anyoptimization/pymoo
- Ray Tune — https://docs.ray.io/en/latest/tune/index.html
- SMAC3 (random-forest BO + HyperBand) — https://github.com/automl/SMAC3
- Salimans et al. (2017), *Evolution Strategies as a Scalable Alternative to Reinforcement Learning* — https://arxiv.org/abs/1703.03864
- Hansen (2016), *The CMA Evolution Strategy: A Tutorial* — https://arxiv.org/abs/1604.00772
- Khamis (2024), *Optimization Algorithms* (Manning) — https://www.manning.com/books/optimization-algorithms
- Khattab et al. (2023), *DSPy: Compiling Declarative Language Model Calls* — https://arxiv.org/abs/2310.02905
