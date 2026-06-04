---
name: rnn
description: Recurrent neural network architectures (Vanilla RNN, LSTM, GRU), bidirectional models, seq2seq with attention, teacher forcing, packed sequences, and gradient clipping. Use when modeling sequential data with PyTorch nn.LSTM/nn.GRU or when transformer overhead is unjustified for short sequences.
---

## Why This Exists

**Problem**: Sequential data (speech, text, time series) has temporal dependencies — the meaning of a token depends on what came before. MLPs treat each input independently and can't model this ordering or variable-length sequences.

**Key insight**: RNNs maintain a hidden state that accumulates information across timesteps, giving the network 'memory' of past inputs. LSTMs/GRUs add gating mechanisms to control what to remember and forget, solving the vanishing gradient problem for sequences up to ~500 tokens.

**Reach for this when**: You have short-to-medium sequences (<500 tokens) and need the simplest recurrent baseline, or you need streaming/online processing with O(1) memory per step. For longer sequences or parallelizable training, prefer Transformers; for ultra-long sequences with linear scaling, consider Mamba/SSMs.


# RNN Architectures Skill

Reference for Recurrent Neural Network architectures, patterns, and PyTorch implementation.

---

## 1. Vanilla RNN

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)
y_t = W_hy · h_t
```

**Vanishing Gradient Problem:** During BPTT (Backpropagation Through Time), gradients are multiplied by `W_hh` at each timestep. If largest singular value of `W_hh` < 1, gradients vanish exponentially with sequence length. If > 1, they explode. This makes vanilla RNNs unable to learn long-range dependencies (typically fails beyond ~10-20 timesteps).

**Mitigation:** Gradient clipping (for exploding), LSTM/GRU (for vanishing).

---

## 2. LSTM (Long Short-Term Memory)

Introduces a **cell state** `C_t` (highway for gradients) and three gates:

```
# Forget gate - what to discard from cell state
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

# Input gate - what new info to store
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)

# Cell state update
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

# Output gate - what to output from cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

**Why it solves vanishing gradients:** Cell state `C_t` has additive updates (not multiplicative), so gradients flow through the forget gate multiplication only — if `f_t ≈ 1`, gradients pass unchanged across many timesteps.

**Parameter count:** For input size `x` and hidden size `h`: `4 * ((x + h) * h + h)` = 4 gates × (weights + bias).

---

## 3. GRU (Gated Recurrent Unit)

Simplified LSTM with 2 gates (fewer params, often comparable performance):

```
# Reset gate - how much past to forget
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)

# Update gate - blend between old and new (combines forget+input gates)
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

# Candidate hidden state
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t] + b)

# Final hidden state
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**vs LSTM:** No separate cell state. `z_t` acts as both forget and input gate (coupled). Fewer parameters (~75% of LSTM). Generally preferred when data is limited or for faster training. LSTM often better for very long sequences or when precise memory control matters.

---

## 4. Bidirectional RNNs

Process sequence in both directions, concatenate hidden states:

```python
# PyTorch
rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
# output shape: (seq_len, batch, 2 * hidden_size)
# h_n shape: (2 * num_layers, batch, hidden_size)

# Separate directions:
# h_n[0] = forward final, h_n[1] = backward final (for 1 layer)
# For stacked: h_n[2*i] = forward layer i, h_n[2*i+1] = backward layer i
```

**Use when:** Full sequence is available at inference (classification, tagging). **Don't use for:** Autoregressive generation (can't see future).

---

## 5. Sequence-to-Sequence (Encoder-Decoder)

```
Encoder: x_1..x_T → h_T (context vector)
Decoder: h_T → y_1..y_T' (autoregressively)
```

**Bottleneck problem:** Entire input compressed into fixed-size `h_T`. Fails for long sequences → solved by attention.

---

## 6. Attention Mechanisms

### Bahdanau (Additive) Attention

```
# Alignment score
e_ij = v^T · tanh(W_1 · s_{i-1} + W_2 · h_j)
# Attention weights
α_ij = softmax(e_ij) over j
# Context vector
c_i = Σ_j α_ij · h_j
```

### Luong (Multiplicative) Attention

```
# Score variants:
dot:     e_ij = s_i^T · h_j
general: e_ij = s_i^T · W · h_j
concat:  e_ij = v^T · tanh(W · [s_i; h_j])
```

---

## 7. Teacher Forcing

During training, feed **ground truth** `y_{t-1}` as decoder input at step `t` instead of model's own prediction.

```python
for t in range(target_len):
    output, hidden = decoder(input, hidden)
    if random.random() < teacher_forcing_ratio:  # scheduled sampling
        input = target[t]  # ground truth
    else:
        input = output.argmax(dim=-1)  # model prediction
```

**Pros:** Faster convergence, more stable training.
**Cons:** Exposure bias (train/test mismatch). Mitigate with **scheduled sampling** (decay ratio over epochs).

---

## 8. Packed Sequences in PyTorch

For variable-length sequences in a batch (avoids computing on padding):

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# lengths must be sorted descending (or use enforce_sorted=False)
# x shape: (batch, max_seq_len, input_size) with batch_first=True
packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
output_packed, (h_n, c_n) = lstm(packed)
output, output_lengths = pad_packed_sequence(output_packed, batch_first=True)
# output shape: (batch, max_seq_len, hidden_size)
```

**Key:** `h_n` correctly contains the final hidden state at each sequence's actual last timestep (not the padded end).

---

## 9. Sequence Patterns

### Many-to-One (Classification)
```python
output, (h_n, c_n) = lstm(x)  # x: (batch, seq_len, input)
# Use final hidden state
logits = fc(h_n[-1])  # h_n[-1]: (batch, hidden)
# Or with bidirectional: cat forward[-1] and backward[-1]
logits = fc(torch.cat([h_n[-2], h_n[-1]], dim=-1))
```

### Many-to-Many (Same length: Tagging)
```python
output, _ = lstm(x)  # output: (batch, seq_len, hidden)
logits = fc(output)   # logits: (batch, seq_len, num_classes)
```

### Many-to-Many (Different length: Translation)
```python
encoder_outputs, hidden = encoder(src)
for t in range(max_target_len):
    output, hidden, attn = decoder(input, hidden, encoder_outputs)
```

### One-to-Many (Generation)
```python
hidden = fc_init(cnn_feature).unsqueeze(0)  # (1, batch, hidden)
input = start_token
outputs = []
for t in range(max_len):
    output, hidden = lstm(input.unsqueeze(1), hidden)
    outputs.append(output.squeeze(1))
    input = output.argmax(-1)
```

---

## 10. Gradient Clipping

```python
# Clip by global norm (most common, preserves gradient direction)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# Clip by value (per-element)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

**Typical max_norm:** 1.0–5.0 for LSTMs. Must be called after `loss.backward()` and before `optimizer.step()`.

---

## 11. Hidden State Initialization

```python
# Zero init (default, most common)
h_0 = torch.zeros(num_layers * num_directions, batch, hidden_size)

# Learned init (trainable parameter)
self.h_0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
# Expand for batch: h_0.expand(-1, batch_size, -1).contiguous()

# Encoder-to-decoder (seq2seq)
decoder_h_0 = self.fc_bridge(encoder_h_n)
```

---

## 12. Stacked (Multi-Layer) RNNs

```python
lstm = nn.LSTM(input_size=256, hidden_size=512, num_layers=3, dropout=0.3)
# dropout applied between layers (not on last layer's output)

# h_n shape: (num_layers * num_directions, batch, hidden_size)
# To get layer i: h_n[i] (unidirectional) or h_n[2*i:2*i+2] (bidirectional)
```

**Typical depth:** 2–4 layers for most tasks. Diminishing returns beyond 4. Use residual connections for >3 layers.

---

## 13. When to Use RNN vs Transformer

| Factor | RNN (LSTM/GRU) | Transformer |
|--------|---------------|-------------|
| Sequence length | Short-medium (<500) | Any length (with efficient attention) |
| Training data | Small-medium datasets | Large datasets (data hungry) |
| Inference | Streaming/online (one token at a time) | Needs full sequence (or KV cache) |
| Memory | O(1) per step | O(n²) attention or O(n) linear |
| Long-range deps | Struggles >100 tokens | Handles well |
| Parameter count | Compact | Large |
| Parallelization | Sequential (slow training) | Fully parallel |

**Use RNN when:** Streaming/online inference, tiny models, embedded/edge, very small data, or when you need constant-memory inference over unbounded streams.

**Use Transformer when:** Offline processing, large data available, long-range dependencies critical, training speed matters.

---

## 14. PyTorch nn.LSTM / nn.GRU Patterns

### Shape Reference

```python
lstm = nn.LSTM(
    input_size=128,      # features per timestep
    hidden_size=256,     # hidden state dimension
    num_layers=2,
    batch_first=True,    # input/output: (batch, seq, feature)
    bidirectional=True,
    dropout=0.2          # between layers (not last)
)

# Input
x.shape        = (batch, seq_len, 128)          # batch_first=True

# Output
output.shape   = (batch, seq_len, 256 * 2)      # 2 for bidirectional
h_n.shape      = (2 * 2, batch, 256)            # (num_layers*directions, batch, hidden)
c_n.shape      = (2 * 2, batch, 256)            # same as h_n (LSTM only)
```

### Full Training Example

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SeqClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, lengths):
        emb = self.dropout(self.embed(x))
        packed = pack_padded_sequence(emb, lengths.cpu(),
                                     batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers*2, batch, hidden) — cat last layer fwd+bwd
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, hidden*2)
        return self.fc(self.dropout(hidden))

# Training loop essentials
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for batch in dataloader:
    optimizer.zero_grad()
    logits = model(batch.ids, batch.lengths)
    loss = nn.functional.cross_entropy(logits, batch.labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
```

---

## 15. Common Pitfalls

1. **Forgetting `contiguous()`** after view/transpose on hidden states
2. **Not detaching hidden** between TBPTT chunks: `h = h.detach()` to prevent graph accumulation
3. **Wrong h_n indexing** for bidirectional stacked: forward layer `i` = `h_n[2*i]`, backward = `h_n[2*i+1]`
4. **Not sorting for pack_padded_sequence** (use `enforce_sorted=False` for simplicity)
5. **Feeding packed output to Linear** without `pad_packed_sequence` first
6. **Initializing hidden on wrong device**: always `.to(x.device)`

---

## 16. Truncated BPTT (for very long sequences)

```python
hidden = None
for chunk in sequence.chunks(tbptt_len):
    if hidden is not None:
        hidden = tuple(h.detach() for h in hidden)  # break graph
    output, hidden = lstm(chunk, hidden)
    loss = criterion(output, targets_chunk)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Typical `tbptt_len`: 35–200 timesteps. Trade-off: shorter = faster but loses longer context gradients.

## When to Use

| ✅ Use RNN (LSTM/GRU) | ❌ Don't Use |
|---|---|
| Short sequences (<500 tokens) | Long documents (>1K tokens, use Transformer) |
| Streaming/real-time inference (one token at a time) | When parallelism matters for training speed |
| Tiny models for edge/mobile deployment | Large-scale pretraining (use Transformer) |
| Time series with clear recurrent structure | When attention over full context is needed |
| Sequential decision making (online learning) | Static feature extraction from sequences |

**Typical domains**: Time series forecasting, speech recognition (CTC), real-time anomaly detection, embedded NLP, music generation.

**Decision rule**: If sequence < 500 AND model must be tiny AND you need streaming → RNN. Otherwise → Transformer or Mamba.

---

## References

- [Understanding LSTM Networks (Olah, 2015)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — Visual LSTM explainer
- [PyTorch Recurrent Layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers) — LSTM, GRU, RNN API
