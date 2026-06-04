---
name: mixture-of-experts
description: Mixture of Experts (MoE) sparse architectures — routing, load balancing, and implementation patterns. Use when scaling model capacity without proportional inference cost (Mixtral, Switch Transformer, DeepSeek), implementing token-level expert routing, or debugging load-balance loss.
---

# Mixture of Experts (MoE)

## 1 — Why MoE Exists

**Problem**: Dense transformers scale capacity by increasing parameters, but every token must pass through ALL parameters — so doubling knowledge requires doubling inference cost. You want a model that knows more without computing more.

**Key insight**: Route each token to only K-of-N expert subnetworks via a learned router — the model has massive capacity (N experts) but constant per-token compute (K experts active). Specialization emerges naturally as experts learn complementary functions.

**Reach for this when**: You need to scale model capacity without proportional inference cost, serve multi-domain workloads where implicit expert specialization helps, or train frontier models (Mixtral, DeepSeek) at compute-efficient scaling ratios. Don't use for models <7B params or memory-constrained deployment (all experts must live in memory).

| Problem | Dense Solution | MoE Solution |
|---------|---------------|--------------|
| More knowledge | Bigger model, slower inference | More experts, same inference cost |
| Multi-domain | One model does everything | Experts specialize implicitly |
| Training efficiency | 4x compute for 2x perf | Near-linear capacity scaling |

---

## 2 — Core Insight: Sparse Activation via Learned Routing

Each token passes through a **router** (small linear layer) that selects K experts out of N. Only selected experts run. The router is trained end-to-end with the rest of the model.

```
Input Token → Router (linear + softmax) → Top-K selection → Selected Expert FFNs → Weighted sum → Output
```

Key properties:
- **Conditional computation**: different tokens activate different params
- **Specialization emerges**: experts learn complementary functions without explicit assignment
- **Constant FLOPs**: regardless of total expert count, each token uses K experts

---

## 3 — Router Types

### Top-K Routing (Standard)
Router outputs logits over N experts, selects top-K, normalizes their weights.

```python
# Standard top-k router
logits = self.gate(x)  # (batch, seq, num_experts)
weights, indices = torch.topk(logits, k=self.top_k, dim=-1)
weights = F.softmax(weights, dim=-1)
```

- **Top-1**: Switch Transformer. Simplest, fastest, but unstable training.
- **Top-2**: Mixtral, GShard. Better quality, 2x expert compute.

### Expert Choice Routing
Experts choose their top-K tokens (instead of tokens choosing experts). Guarantees perfect load balance.

```python
# Expert choice: transpose the routing problem
logits = self.gate(x)  # (batch*seq, num_experts)
# Each expert picks its top-C tokens
indices = torch.topk(logits.T, k=capacity, dim=-1).indices
```

- Used in: Expert Choice paper (Zhou et al., 2022)
- Pro: no auxiliary loss needed, perfect balance
- Con: variable tokens per expert complicates batching

### Hash Routing
Deterministic routing by token position/hash. No learned router.

```python
expert_idx = hash(token_position) % num_experts
```

- Used in: Hash layers (Roller et al., 2021)
- Pro: zero routing overhead, perfectly balanced
- Con: no learned specialization

---

## 4 — Load Balancing

Without balancing, routers collapse: send all tokens to 1-2 experts, others idle.

### Auxiliary Loss (Standard)
Penalizes uneven expert utilization. Added to main loss with coefficient α (typically 0.01).

```python
# f_i = fraction of tokens routed to expert i
# P_i = mean router probability for expert i
aux_loss = alpha * num_experts * sum(f_i * P_i for i in range(num_experts))
```

This encourages uniform routing: if expert i gets too many tokens (high f_i), the loss pushes P_i down.

### Capacity Factor
Hard cap on tokens per expert per batch. Overflow tokens are dropped or sent to a shared expert.

```
capacity = (tokens_per_batch / num_experts) * capacity_factor
# capacity_factor typically 1.0-1.5
```

### Z-Loss (DeepSeek)
Penalizes large router logits to prevent routing collapse:

```python
z_loss = beta * torch.mean(torch.logsumexp(logits, dim=-1) ** 2)
```

### Shared Expert (DeepSeek-V2)
One or more experts always activated for every token (dense path), remaining experts sparse-routed. Ensures baseline quality even with routing failures.

---

## 5 — PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (batch, seq, d_model)
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # (B*S, D)

        # Route
        logits = self.gate(x_flat)  # (B*S, num_experts)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)  # (B*S, top_k)
        weights = F.softmax(weights, dim=-1)

        # Dispatch to experts
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for i in range(self.num_experts):
                mask = (indices[:, k] == i)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[i](expert_input)
                    output[mask] += weights[mask, k].unsqueeze(-1) * expert_output

        # Auxiliary load balancing loss
        self.aux_loss = self._load_balance_loss(logits, indices)
        return output.view(B, S, D)

    def _load_balance_loss(self, logits, indices):
        # f_i: fraction of tokens per expert
        num_tokens = logits.shape[0]
        f = torch.zeros(self.num_experts, device=logits.device)
        for k in range(self.top_k):
            f.scatter_add_(0, indices[:, k], torch.ones(num_tokens, device=logits.device))
        f = f / (num_tokens * self.top_k)
        # P_i: mean probability per expert
        P = F.softmax(logits, dim=-1).mean(dim=0)
        return self.num_experts * (f * P).sum()
```

**Production note**: The loop-based dispatch above is pedagogical. Real implementations use `torch.scatter`/`torch.gather` with padded expert buffers for GPU efficiency (see Megablocks, Tutel, or Fairseq MoE).

---

## 6 — Real Examples

### Switch Transformer (2021)
- Top-1 routing, simplified MoE for pretraining
- 1.6T params, same compute as T5-Base
- 7x speedup over dense equivalent at same quality
- Key innovation: showed top-1 works (GShard used top-2)
- Paper: https://arxiv.org/abs/2101.03961

### Mixtral 8x7B (2024)
- 8 experts, top-2 routing per token
- 47B total params, 13B active per token
- Matches/beats Llama 2 70B at 6x less inference compute
- Every layer is MoE (no dense layers)
- Sliding window attention (4096 tokens)
- Paper: https://arxiv.org/abs/2401.04088

### DeepSeek-V2 (2024)
- 236B total params, 21B active
- Multi-head Latent Attention (MLA) + DeepSeekMoE
- 2 shared experts + 160 routed experts, top-6 routing
- Device-limited routing: restrict expert selection to same GPU
- 42.5x cheaper than DeepSeek 67B at higher quality
- Paper: https://arxiv.org/abs/2405.04434

### Architecture Comparison

| Model | Total Params | Active Params | Experts | Top-K | Special |
|-------|-------------|---------------|---------|-------|---------|
| Switch-Base | 7.4B | 0.2B | 128 | 1 | Simplified routing |
| Mixtral 8x7B | 47B | 13B | 8 | 2 | Every layer MoE |
| DeepSeek-V2 | 236B | 21B | 2+160 | 6 | Shared + routed experts |
| Grok-1 | 314B | ~86B | 8 | 2 | xAI, top-2 |

---

## 7 — When to Use MoE vs Dense

### Use MoE When:
- **Scaling capacity** without proportional inference cost
- **Multi-task/multi-domain** workloads where specialization helps
- **Inference budget is fixed** but you want more knowledge
- **Pretraining at scale** (1T+ tokens, large clusters)

### Use Dense When:
- **Small models** (<1B) — routing overhead dominates
- **Single-task** fine-tuning — specialization adds complexity without benefit
- **Memory-constrained** — MoE needs all experts in memory even if few activate
- **Batch size 1** inference — can't amortize routing cost
- **Simplicity** — dense is easier to train, debug, quantize, deploy

### Decision Matrix

| Factor | Favors MoE | Favors Dense |
|--------|-----------|--------------|
| Model scale | >7B total params | <7B params |
| Domains covered | Multi-domain | Single domain |
| Inference latency | Batch serving | Single-request |
| Memory budget | Can hold all experts | Tight memory |
| Training data | >500B tokens | <100B tokens |
| Deployment | Dedicated infra | Commodity GPUs |

---

## 8 — Gotchas

1. **All experts in memory**: 8x7B means 47B params in VRAM even though only 13B activate. MoE doesn't save memory, only compute.
2. **Expert parallelism required**: distributing experts across GPUs adds all-to-all communication. Latency-sensitive apps need careful placement.
3. **Training instability**: routers can collapse early. Use auxiliary loss from step 0, not added later.
4. **Fine-tuning fragility**: experts over-specialize during fine-tuning. Use lower LR for router, or freeze it.
5. **Quantization is harder**: different experts have different weight distributions. Per-expert quantization needed.
6. **Token dropping**: capacity factor overflow means some tokens skip the expert layer entirely. Acceptable in training, problematic in inference.
7. **Batch size sensitivity**: small batches mean few tokens per expert, poor GPU utilization. MoE shines at large batch sizes.

---

## 9 — References

- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961) — Fedus et al., 2021
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Jiang et al., 2024
- [Mixtral 8x7B (HuggingFace model card)](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) — weights, config, usage
- [DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model](https://arxiv.org/abs/2405.04434) — DeepSeek-AI, 2024
- [DeepSeek-MoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.04088) — DeepSeek-AI, 2024
- [Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538) — Shazeer et al., 2017 (original MoE routing paper)
- [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668) — Lepikhin et al., 2020
- [Expert Choice Routing](https://arxiv.org/abs/2202.09368) — Zhou et al., 2022
- [Megablocks: Efficient Sparse Training with MoE](https://arxiv.org/abs/2211.15841) — Gale et al., 2022
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906) — Zoph et al., 2022
