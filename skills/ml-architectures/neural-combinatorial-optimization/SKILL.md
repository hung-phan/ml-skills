---
name: neural-combinatorial-optimization
description: Neural Combinatorial Optimization (NCO) — Pointer Networks, Attention Model, GNN+RL, diffusion-based, transformer heuristics for TSP, VRP, JSSP, MaxCut, SAT, knapsack. End-to-end learning, learning-to-configure, and learning-to-advise paradigms. Use when classical solvers (Concorde, Gurobi, OR-tools) are too slow at inference, you have many similar instances, or you want amortized millisecond-latency heuristics with controllable optimality gap.
---

# Neural Combinatorial Optimization (NCO)

## Why This Exists

**Problem.** Classical combinatorial optimization solvers — Concorde for TSP,
Gurobi/CPLEX for MILP, Google OR-Tools for routing — are extremely mature and
provably optimal (or with tight bounds). But they have three structural
weaknesses on production workloads:

1. **Latency.** Concorde solves a 1000-city TSP in seconds; a 10000-city
   instance can take hours. Branch-and-cut scales poorly to repeated, small,
   latency-sensitive queries (re-route every 50 ms in a dispatcher).
2. **Adaptation.** Switching objective ("minimize tour length subject to time
   windows AND carbon cost") often requires re-deriving cuts or rewriting
   constraint code. Exact solvers do not transfer.
3. **Distribution-specific structure.** Real instances are not adversarial
   worst cases. Real TSP instances from a fleet, real CVRP instances from one
   warehouse, real JSSP instances from one factory all share latent structure
   the solver ignores.

**Key insight.** If you face *many similar* instances and can simulate or
collect optimal-ish solutions, you can *amortize* the search: train a neural
policy once, then run it in milliseconds per instance. Bengio, Lodi & Prouvost
(2021) framed this in three modes:

1. **End-to-end learning** — the network outputs the solution directly
   (Pointer Network, Attention Model, GNN-decoder).
2. **Learning to configure** — the network picks hyperparameters or
   primal heuristics for a classical solver (cooling schedule for SA, ACO
   pheromone weights, Gurobi parameter tuning, branching variable order).
3. **Learning to advise / repeated calls** — a classical solver iteratively
   queries a learned model for a low-level decision (which variable to branch
   on, which node to expand, which neighborhood to destroy in LNS). Examples:
   Gasse et al. GNN for branch-and-bound, NeuroSAT.

**Reach for this when:**
- You have **many similar instances** (a fleet routes 100k packages/day from
  the same depots).
- You can **simulate** the environment or have access to a slow optimal
  solver to generate training labels.
- You need **ms-latency** at inference, not best-possible solution.
- A **fixed optimality gap** (say 2–5 %) is acceptable.
- Objective changes are frequent — learn-to-advise wraps existing solvers.

**Do NOT reach for this when:**
- A single high-stakes instance must be solved provably to optimality
  (cutting stock for an aerospace job — call Gurobi).
- Distribution shifts faster than you can retrain (one-off custom problems).
- Instances are tiny (<50 vars) — Concorde/Gurobi already finish in
  microseconds.

---

## Decision Table

| Situation | Recommended approach | Why |
|---|---|---|
| Single TSP, n ≤ 1000, optimality required | **Concorde** | Mature, provably optimal, fast at this size. |
| MILP with strict optimality and bound proof | **Gurobi / CPLEX / SCIP** | Battle-tested branch-and-cut; certificates. |
| Production CVRP, 100s nodes, ms latency, recurring distribution | **End-to-end NCO** (RL4CO Attention Model + POMO + beam search) | Amortizes search; <100 ms inference; 1–4 % gap. |
| MILP solver too slow, want to keep correctness guarantees | **Learning to advise** (GNN branching à la Gasse 2019) | Solver still proves optimality; ML only re-orders branches. |
| Generic black-box problem, no gradient, low budget | **Metaheuristics** (SA, GA, PSO) — see [gradient-free-optimization](../../ml-training/gradient-free-optimization/) | No training data needed. |
| JSSP, dispatching rules | **L2D (Learning to Dispatch) — GNN + PPO** | Dynamic disjunctive graph; ms decision per operation. |
| SAT decision / planning | **NeuroSAT, AlphaZero-style** | Self-play + tree search beats hand-crafted heuristics. |
| Hybrid: ML proposes, exact refines | **NCO + 2-opt / OR-tools local search** | Best of both — ML warm-starts, classical polishes. |
| Massive graph problems (MaxCut, MIS, MVC), n ≥ 10k | **GNN + RL** (S2V-DQN, ECO-DQN, DIFUSCO) | GNN captures locality; RL handles unlabeled. |

---

## The Three Bengio Paradigms

### 1. End-to-end learning

Network maps **instance → solution**. Trained with one of:
- **Supervised / imitation** — generate (instance, optimal solution) pairs
  with a slow oracle (Concorde for TSP) and minimize cross-entropy.
- **REINFORCE** — sample tour, reward = -length, update with policy
  gradient. No oracle needed.
- **Self-supervised / contrastive** — diffusion + denoising (DIFUSCO).

Examples: Pointer Networks (Vinyals 2015), Bello et al. RL Ptr-Net (2016),
Attention Model (Kool 2018), POMO (Kwon 2020), DIFUSCO (Sun & Yang 2023),
NeuOpt, MatNet.

### 2. Learning to configure

Network picks knobs for a classical algorithm. Less common as a research
topic but huge in practice — every Gurobi parameter-tuning service uses
this. Examples:
- Initial temperature / cooling schedule for SA.
- Pheromone evaporation rate for ACO.
- Population size, mutation/crossover probability for GA.
- `MIPGap`, `Heuristics`, `Cuts` parameters for Gurobi (SMAC, ConfigSpace).
- Selecting which warm-start to use.

### 3. Learning to advise (in-the-loop)

Solver calls the model **at every step**. The state is the partial solution
or solver tree, the action is the next decision (branch variable, node to
expand, neighborhood to destroy).

Canonical examples:
- **Gasse et al. (2019)** — GNN learns branch-and-bound variable selection,
  imitating strong branching expert. Drop-in for SCIP.
- **NeuroSAT** — learns SAT variable assignment heuristics.
- **Hottung 2020 — DLTS** for container pre-marshalling: tree search with
  DNN bounding.
- **Learning to Search (L2S)** — GNN proposes which neighborhood to destroy
  in Large Neighborhood Search.

This is the best fit when you must keep **certificates of optimality** —
the underlying solver still proves bounds; ML only reorders work.

---

## Architectures

### Pointer Networks (Vinyals 2015)

The grandparent. Encoder-decoder seq2seq with **content-based attention used
as a pointer** — at each decode step, the attention distribution over the
*input* sequence is the output. Output dictionary size = input length, so
it handles variable-sized problems (n cities).

Pseudo:
```
encode(x_1..x_n) → h_1..h_n
for t in 1..n:
    a_t = attention(decoder_state, h_1..h_n)   # softmax over inputs
    pointer_t = argmax(a_t)                    # pick a city
    decoder_state = update(decoder_state, h_{pointer_t})
```

Trained supervised on (point set, convex hull) or (cities, optimal TSP tour).
Works on n ≤ 50, struggles to generalize beyond.

### RL Pointer Network (Bello et al. 2016)

Same architecture as Vinyals but trained with **REINFORCE** using tour length
as reward. Removes the need for optimal labels; allows training on much
larger n. Adds a critic (value baseline) to reduce variance.

### Attention Model (Kool et al. 2018)

The current production-strength baseline for routing. Replaces RNN with a
**transformer encoder** + **single-head pointer decoder**, trained with
REINFORCE using a **rollout baseline** (greedy decode of an exponentially
moving-average policy as baseline). Scales to TSP100, CVRP100 with
1–4 % optimality gap; <100 ms inference. This is what RL4CO ships as default.

```python
# Conceptual Attention Model decode (per step)
# context = embed(graph) + embed(first_node) + embed(last_node)
# probs = softmax(C * tanh(W_q context + W_k node_embeds))   # masked, single-head
```

### POMO (Kwon et al. 2020)

Plug-in trick on top of Attention Model. Insight: for TSP, *every* node is a
valid starting point and the optimal tour is the same up to rotation. So
sample N parallel rollouts, one starting from each node, and use their
**shared baseline** (mean reward of the N) — much lower variance than
greedy-rollout baseline. Improves TSP/CVRP solution quality by ~1 %
absolute, at no extra cost. Always turn POMO on if your problem has
symmetry.

### GNN + RL for graph problems

For TSP/MaxCut/MIS/MVC where the **graph itself is the input**, swap the
transformer encoder for a **GNN encoder** (GCN, GIN, GAT, attention-GNN).
The decoder is still pointer-style (or pick-a-vertex).

- **S2V-DQN (Khalil et al. 2017)** — structure2vec embedding + DQN over
  vertex picks. Trained on small instances, generalizes to graphs 10x larger.
- **ECO-DQN** — local-move RL for MaxCut.
- **Joshi et al. (2019, 2021)** — supervised GNN + beam search for TSP.
  Pipeline: graph → GCN message passing → MLP per edge → probability of
  edge being in optimal tour → beam search to extract a tour.

Skeleton:
```python
import torch
import torch.nn as nn

class GNNEncoder(nn.Module):
    """Edge-aware GNN encoder for routing problems."""
    def __init__(self, hidden=128, num_layers=4):
        super().__init__()
        self.node_in = nn.Linear(2, hidden)        # 2-D coords
        self.edge_in = nn.Linear(1, hidden)        # distance feature
        self.layers = nn.ModuleList([
            EdgeMPNNLayer(hidden) for _ in range(num_layers)
        ])

    def forward(self, coords, dist):
        # coords: (B, N, 2)   dist: (B, N, N, 1)
        h = self.node_in(coords)            # (B, N, H)
        e = self.edge_in(dist)              # (B, N, N, H)
        for layer in self.layers:
            h, e = layer(h, e)
        return h, e


class EdgeMPNNLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.msg = nn.Linear(3 * hidden, hidden)
        self.upd = nn.GRUCell(hidden, hidden)
        self.eup = nn.Linear(3 * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, e):
        B, N, H = h.shape
        hi = h.unsqueeze(2).expand(B, N, N, H)
        hj = h.unsqueeze(1).expand(B, N, N, H)
        m = torch.relu(self.msg(torch.cat([hi, hj, e], -1)))   # (B,N,N,H)
        agg = m.sum(dim=2)                                     # (B,N,H)
        h_new = self.upd(agg.reshape(-1, H), h.reshape(-1, H)).reshape(B, N, H)
        e_new = torch.relu(self.eup(torch.cat([hi, hj, e], -1)))
        return self.norm(h_new), e_new
```
This encoder feeds a pointer decoder (one softmax per step, masking visited
nodes) trained via REINFORCE+POMO.

### Transformer-based heuristics (NeuOpt, MatNet)

Beyond simple pointer decode:
- **NeuOpt** — learns to *improve* an existing tour (operator selection in a
  local-search loop). Closer to RL-controlled metaheuristic.
- **MatNet (Kwon 2021)** — replaces position embeddings with **learned
  matrix encoding** for asymmetric problems (ATSP, FFSP). Encoder operates
  on a non-symmetric distance/cost matrix directly.

### Diffusion-based (DIFUSCO, T2T)

Treat the solution as a discrete distribution. Train a **denoising
diffusion model** to gradually denoise from random into a feasible solution.
- **DIFUSCO (Sun & Yang 2023)** — for TSP and MIS, predicts edge / node
  inclusion as Bernoulli; iterative denoising + greedy decoding. Beats
  Attention Model on TSP-500 and TSP-1000 (better extrapolation).
- **T2T** — text-to-tour analogy.
Strong **generalization to larger instances** is the main draw.

### AlphaZero-style for SAT / planning

Self-play MCTS + neural value+policy. For SAT it's NeuroSAT-like with a
search tree; for general planning, AlphaZero/Schrittwieser MuZero style.
Highest sample budget, highest ceiling.

### Imitation learning from optimal solvers

When you can afford to run Concorde once on small instances, generating
(graph, optimal tour) pairs and training a GNN with cross-entropy is
*surprisingly competitive* with RL and trains 5–20× faster. Joshi et al.
take this approach. The catch: extrapolation outside the training-size
distribution drops sharply.

---

## Problem Families and Representative Methods

| Problem | Definition | NCO methods of choice |
|---|---|---|
| **TSP** | Shortest tour visiting all cities once | Attention Model + POMO; DIFUSCO for large n |
| **VRP / CVRP** | Multiple vehicles, capacity constraints | Attention Model with masking; RLOR; Nazari et al. (2018) |
| **JSSP** (job-shop) | Schedule operations on machines, minimize makespan | **L2D (Zhang et al. 2020)** — disjunctive graph + GIN + PPO |
| **Bin packing** | Pack items into fewest bins / smallest bin | RL placement policy; constructive |
| **Knapsack** | Pick items max value, capacity ≤ C | Pointer Net, Attention Model |
| **MaxCut / MIS / MVC** | Graph-cut / independent-set problems | S2V-DQN, ECO-DQN, DIFUSCO |
| **SAT** | Boolean satisfiability | NeuroSAT, NeuroCore, AlphaZero+search |
| **MILP branching** | Branch-and-bound variable selection | **Gasse et al. (2019)** — bipartite GNN + imitation of strong branching |

---

## Evaluation

NCO papers report three numbers; insist on all three for any production
candidate:

1. **Optimality gap** vs. an exact solver:
   `(L_model - L_optimal) / L_optimal × 100 %`. Below 2 % is strong on TSP100.
2. **Inference time** per instance (single-shot, beam search width K, sample
   width N).
3. **Generalization** — train on n=50, test on n=100, 200, 500, 1000.
   Performance usually degrades; reporting only on the training distribution
   is suspicious.

Other diagnostics:
- **Optimality gap distribution** (worst-case 99th-percentile, not just mean).
- **Feasibility rate** — for constrained problems (CVRP capacity, JSSP
  precedence), check the fraction of returned solutions that are feasible.
- **Search budget** — beam K=1 (greedy) vs K=1280 (sampling). NCO papers
  usually report several settings.
- **Hardware** — single-GPU vs CPU; some NCO methods are GPU-bound, classical
  solvers CPU-bound.

Comparison points to *always* include:
- A classical exact solver (Concorde / OR-Tools) given equivalent wall-clock.
- A simple heuristic (nearest-neighbor + 2-opt) given equivalent wall-clock.
- The best published NCO baseline on the same dataset (Attention Model from
  Kool et al. is the Imagenet-equivalent; DIFUSCO for large-scale).

---

## Practical Caveats

- **Extrapolation to bigger instances is the open problem.** A model
  trained on n=50 typically loses ~5–10 % gap when tested on n=200, more
  beyond that. Diffusion methods (DIFUSCO) and explicit "scale-aware"
  embeddings help. If you must serve a wide range of n, train on a *mix*
  of sizes.

- **Beam search at inference is almost free.** The REINFORCE-trained policy
  is stochastic — sampling K=128 or 1280 trajectories and keeping the best
  costs only K× the inference time and typically improves gap by 1–3 %
  absolute. POMO does this implicitly.

- **Hybrid wins in production.** Pure end-to-end NCO is rarely deployed
  alone. Standard recipe:
    1. NCO produces a candidate solution.
    2. Cheap classical local search (2-opt, Or-opt, LKH-style) polishes.
    3. (Optional) Periodic exact solver run on small subproblems for
       certificate.
  Several frameworks bake this in (NeuOpt, RL4CO's `improve` envs).

- **Distributional drift is a silent killer.** Retrain on freshly sampled
  production instances every quarter; track training/eval distribution KL.

- **Action masking is mandatory** for hard constraints — don't try to learn
  feasibility with rewards alone; mask infeasible actions in the decoder
  softmax.

- **Scaling up is mostly an inference engineering problem.** Use FP16,
  CUDA graphs, batch inference. RL4CO uses `torch.compile` and TorchRL
  vectorized envs.

---

## Frameworks

| Framework | Best for | Notes |
|---|---|---|
| **RL4CO** ([github.com/ai4co/rl4co](https://github.com/ai4co/rl4co)) | Routing, scheduling, end-to-end | PyTorch Lightning + TorchRL; ships Attention Model, POMO, MatNet, NeuOpt, DIFUSCO; recommended starting point in 2025+. |
| **DGL / PyG** | Custom GNN encoders | Flexible message-passing, but you write the RL loop. |
| **OR-Tools** ([developers.google.com/optimization](https://developers.google.com/optimization)) | Classical baselines, hybrid LNS | Open-source CP-SAT, VRP solver. Pair with NCO as warm-start or local-search step. |
| **Gurobi** ([gurobi.com](https://www.gurobi.com/)) | Production MILP, baselines | Best-in-class commercial solver; use for ground-truth labels and as a hybrid backbone. |
| **Concorde** | Optimal TSP labels | Generates ground-truth TSP tours up to ~10000 cities. Use offline to build datasets. |
| **PySCIPOpt** | Learning-to-advise on B&B | Open MILP solver with hooks for custom branchers — required for Gasse-style work. |

---

## Code Examples

### 1. RL4CO Attention Model on TSP (recommended starting point)

```python
# pip install rl4co
from rl4co.envs import TSPEnv
from rl4co.models import AttentionModel, AttentionModelPolicy
from rl4co.utils.trainer import RL4COTrainer

# 1. Environment — generates random TSP instances on the fly
env = TSPEnv(generator_params=dict(num_loc=50))      # TSP-50

# 2. Policy — encoder = transformer; decoder = pointer with masking
policy = AttentionModelPolicy(
    env_name=env.name,
    embed_dim=128,
    num_encoder_layers=3,
    num_heads=8,
)

# 3. Model — REINFORCE with rollout (or shared POMO) baseline
model = AttentionModel(
    env, policy,
    baseline="rollout",        # or "shared" for POMO
    train_data_size=100_000,
    val_data_size=10_000,
    optimizer_kwargs={"lr": 1e-4},
)

# 4. Train
trainer = RL4COTrainer(max_epochs=100, devices=1, accelerator="gpu")
trainer.fit(model)

# 5. Inference: greedy or beam-search decode
td = env.reset(batch_size=[16])           # 16 fresh instances
out = policy(td, env, decode_type="greedy")
print(out["reward"].mean())               # negative tour length
```

To enable POMO, set `baseline="shared"` and `num_starts=N` (e.g. 50) — this
turns on the multi-trajectory data augmentation across N start nodes.

### 2. Hybrid: NCO proposes, 2-opt refines

```python
import numpy as np

def two_opt(tour, dist):
    """Standard 2-opt local search. Stops at first local optimum."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                a, b, c, d = tour[i-1], tour[i], tour[j-1], tour[j % n]
                # delta = d(a,c) + d(b,d) - d(a,b) - d(c,d)
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                if delta < -1e-9:
                    tour[i:j] = tour[i:j][::-1]
                    improved = True
    return tour


# NCO proposes
td = env.reset(batch_size=[1])
out = policy(td, env, decode_type="sampling", num_samples=128)
best = out["actions"][out["reward"].argmax()].cpu().numpy()

# Classical refines
coords = td["locs"][0].cpu().numpy()
dist = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
refined = two_opt(best.tolist(), dist)
print("NCO length :", -out["reward"].max().item())
print("NCO + 2opt :", sum(dist[refined[i], refined[(i+1)%len(refined)]]
                         for i in range(len(refined))))
```

Typical wins: 0.5–1.5 % absolute gap improvement over pure NCO at <50 ms
extra wall-clock per instance.

### 3. GNN encoder for graph-structured problems

See the `GNNEncoder` / `EdgeMPNNLayer` skeleton above. Plug it as the
`encoder` in any RL4CO routing model:

```python
from rl4co.models.zoo.am import AttentionModelPolicy

policy = AttentionModelPolicy(
    env_name="tsp",
    encoder=GNNEncoder(hidden=128, num_layers=4),    # custom GNN encoder
    embed_dim=128,
)
```

### 4. Imitation from Concorde labels (supervised)

```python
import torch.nn.functional as F

# (graph, optimal_tour) pairs from Concorde, batched
for batch in dataloader:
    coords, optimal = batch["coords"], batch["optimal"]  # (B,N,2), (B,N)
    log_probs = policy.forward_with_labels(coords, optimal)
    loss = F.nll_loss(log_probs.reshape(-1, log_probs.size(-1)),
                      optimal.reshape(-1))
    loss.backward()
    optimizer.step()
```
Trains 5–20× faster than REINFORCE; works up to ~n=100; weaker
generalization beyond.

---

## Mental Model: When to Choose What

```
                 ┌─────────────────────────────────────────────┐
                 │  Need provable optimality on this instance? │
                 └────────────────┬────────────────────────────┘
                                  │
                ┌─────yes─────────┴────────no─────┐
                ▼                                  ▼
         exact solver               ┌───────────────────────────┐
         (Concorde, Gurobi)         │ Many similar instances?    │
                                    └────────────┬───────────────┘
                                                 │
                                ┌────yes─────────┴────no────┐
                                ▼                            ▼
                ┌────────────────────────┐         metaheuristic
                │ Allowed to wrap solver │     (SA / GA / PSO / Tabu)
                │ or replace it?         │     see gradient-free-optimization
                └──────────┬─────────────┘
                           │
              ┌────wrap────┴───replace────┐
              ▼                            ▼
     learning-to-advise              end-to-end NCO
     (Gasse GNN-B&B,                 (RL4CO Attention Model
      LNS-with-RL,                    + POMO + beam search,
      branching/cutting nets)         + 2-opt refinement)
```

---

## Common Pitfalls

- **Training only on uniform random instances** — production data is rarely
  uniform. Sample synthetic data that *matches* your operating distribution
  (clustered cities, depot-heavy, time-window structure).

- **Forgetting to mask infeasible actions** — letting the decoder pick a
  visited city will silently hurt training. Use causal masks.

- **Comparing NCO to OR-Tools/Gurobi at default settings** — both classical
  solvers can be aggressively tuned; report the best a competitor with equal
  engineering effort would achieve.

- **Using only mean optimality gap** — production cares about p99 worst-case
  too. Plot the gap distribution.

- **Ignoring the constraint of constructive vs improvement** — constructive
  policies (Pointer/Attention/POMO) build a solution from scratch in O(n)
  steps. Improvement policies (NeuOpt, ECO-DQN) start from a feasible
  solution and edit it. Improvement methods extend better to large n but
  need a starting solution.

- **Underestimating beam search.** A fully greedy attention model often
  loses 2–4 % gap that a beam K=1280 would close. Always benchmark both.

---

## See Also

- [../gnn/](../gnn/) — Graph neural network encoders for graph-structured
  combinatorial problems (TSP, MaxCut, MIS, B&B).
- [../reinforcement-learning/](../reinforcement-learning/) — REINFORCE,
  PPO, A2C and the policy-gradient machinery NCO is built on.
- [../transformer/](../transformer/) — The Attention Model, MatNet and
  most modern NCO architectures rely on multi-head attention encoders.
- [../../ml-training/gradient-free-optimization/](../../ml-training/gradient-free-optimization/) —
  Metaheuristics (SA, GA, PSO, Tabu, ACO) and when classical search beats
  learned policies; NCO and metaheuristics are complementary.
- [../../ml-libraries/pytorch/](../../ml-libraries/pytorch/) — Core PyTorch
  tooling that all RL4CO/RL training rests on.

---

## References

Verified 2026-06-05 (`curl -sIL`, all return HTTP 200):

- RL4CO framework — https://github.com/ai4co/rl4co
- Vinyals, Fortunato, Jaitly. *Pointer Networks* (2015) — https://arxiv.org/abs/1506.03134
- Bello, Pham, Le, Norouzi, Bengio. *Neural Combinatorial Optimization with RL* (2016) — https://arxiv.org/abs/1611.09940
- Kool, van Hoof, Welling. *Attention, Learn to Solve Routing Problems!* (2018) — https://arxiv.org/abs/1803.08475
- Bengio, Lodi, Prouvost. *Machine Learning for Combinatorial Optimization: a Methodological Tour d'Horizon* (2018, EJOR 2021) — https://arxiv.org/abs/1811.06128
- Kwon, Choo, Kim, Yoon, Min, Gwon. *POMO: Policy Optimization with Multiple Optima for RL* (2020) — https://arxiv.org/abs/2010.16011
- Sun, Yang. *DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization* (2023) — https://arxiv.org/abs/2310.10709
- Gasse, Chételat, Ferroni, Charlin, Lodi. *Exact Combinatorial Optimization with GCNs* (2019, NeurIPS) — https://arxiv.org/abs/1906.01629
- Zhang, Song, Cao, Zhang, Tan, Chi. *Learning to Dispatch for JSSP via Deep RL* (L2D, 2020) — https://arxiv.org/abs/2010.16317
- Google OR-Tools — https://developers.google.com/optimization
- Gurobi Optimizer — https://www.gurobi.com/
