---
name: gnn
description: Graph neural networks for node, edge, and graph-level tasks — message passing, GCN/GAT/GraphSAGE, PyG and DGL patterns. Use when modeling graph-structured data (social networks, molecules, knowledge graphs, recommendations) where relationships matter as much as node features.
---

## Why This Exists

**Problem**: Many real-world systems are naturally graphs — social networks, molecules, knowledge bases, recommendation systems, traffic networks — where relationships between entities matter as much as entity features. Standard neural nets (MLPs, CNNs, RNNs) can't operate on irregular graph-structured data.

**Key insight**: Propagate information along edges via message passing — each node aggregates features from its neighbors, creating representations that encode both local features and graph topology. Stack L layers = each node sees L-hop neighborhood.

**Reach for this when**: Your data has explicit relational structure (edges between entities), you need to classify nodes/edges/whole graphs, or you want to reason over molecular structures, knowledge graphs, or social networks. If your data is a grid (image) or sequence (text), CNNs/Transformers are simpler and faster.


# Graph Neural Networks (GNN)

## Core Concepts

### Message Passing Framework

All GNNs follow the message-passing paradigm:

```python
# For each node v, at layer l:
# 1. AGGREGATE: collect messages from neighbors N(v)
# 2. UPDATE: combine with node's own representation

h_v^(l+1) = UPDATE(h_v^(l), AGGREGATE({h_u^(l) : u ∈ N(v)}))
```

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class CustomMP(MessagePassing):
    """Generic message passing layer."""
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # 'add', 'mean', 'max'
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out
```

---

## Architectures

### GCN (Graph Convolutional Network)

```python
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden_ch)
        self.conv2 = GCNConv(hidden_ch, out_ch)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

**When to use**: Homogeneous graphs, semi-supervised node classification, citation networks.

### GAT (Graph Attention Network)

```python
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, heads=8, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(in_ch, hidden_ch, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_ch * heads, out_ch, heads=1,
                             concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

**When to use**: Graphs where neighbor importance varies (social networks, heterogeneous interactions).

### GraphSAGE (Sample and Aggregate)

```python
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hidden_ch)
        self.conv2 = SAGEConv(hidden_ch, out_ch)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
```

**Mini-batch training (large graphs)**:

```python
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],
    batch_size=1024,
    input_nodes=data.train_mask,
)

for batch in loader:
    out = model(batch.x, batch.edge_index)
    loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
```

**When to use**: Large-scale graphs (millions of nodes), inductive settings, production recommender systems.

### GIN (Graph Isomorphism Network)

Maximally powerful among 1-WL test GNNs. Best for graph-level classification.

```python
from torch_geometric.nn import GINConv, global_add_pool

class GIN(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, num_layers=5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(in_ch if i == 0 else hidden_ch, hidden_ch),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_ch, hidden_ch),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(torch.nn.BatchNorm1d(hidden_ch))
        self.lin = torch.nn.Linear(hidden_ch, out_ch)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        x = global_add_pool(x, batch)
        return self.lin(x)
```

---

## Task Types

### Node-Level (Classification/Regression)

```python
out = model(data)
loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
```

### Edge-Level (Link Prediction)

```python
from torch_geometric.utils import negative_sampling

def link_pred_loss(z, pos_edge_index, neg_edge_index=None):
    if neg_edge_index is None:
        neg_edge_index = negative_sampling(pos_edge_index, num_nodes=z.size(0))
    pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
    neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    return pos_loss + neg_loss
```

### Graph-Level (Classification/Regression)

```python
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader

x = global_mean_pool(x, batch)  # [num_graphs, hidden]
out = self.classifier(x)

loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

---

## Heterogeneous Graphs

```python
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv
from torch_geometric.data import HeteroData

data = HeteroData()
data['user'].x = torch.randn(1000, 64)
data['item'].x = torch.randn(5000, 128)
data['user', 'buys', 'item'].edge_index = ...
data['user', 'rates', 'item'].edge_index = ...

class HeteroGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = HeteroConv({
            ('user', 'buys', 'item'): SAGEConv((-1, -1), 64),
            ('user', 'rates', 'item'): GATConv((-1, -1), 64, add_self_loops=False),
            ('item', 'similar', 'item'): GCNConv(-1, 64),
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return x_dict
```

---

## Use-Case Guide

| Domain | Task | Architecture |
|--------|------|--------------|
| **Social Networks** | Community detection | GraphSAGE + sampling |
| **Molecules** | Property prediction | GIN + global_add_pool |
| **Recommendation** | User-item scoring | HeteroGNN (LightGCN) |
| **Knowledge Graphs** | Link prediction | R-GCN, CompGCN |
| **Traffic/Maps** | Flow prediction | Temporal GNN |
| **Fraud Detection** | Anomaly node detection | GAT + heterogeneous |
| **Proteins** | Structure prediction | Equivariant GNN (EGNN) |

---

## Performance Tips

1. **Over-smoothing**: Limit to 2-3 layers. Use residual connections or JumpingKnowledge for deeper.
2. **Large graphs**: Use `NeighborLoader` (PyG) or `MultiLayerNeighborSampler` (DGL).
3. **Feature normalization**: Apply `LayerNorm` or `BatchNorm` between GNN layers.
4. **Edge features**: Use `NNConv` or `TransformerConv`.
5. **Dropout**: Apply to both features and attention coefficients (GAT).
6. **Learning rate**: Start at 0.01 for GCN/SAGE, 0.005 for GAT.

```python
from torch_geometric.nn import JumpingKnowledge

class DeepGNN(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, num_layers=6):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_ch, hidden_ch))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_ch, hidden_ch))
        self.jk = JumpingKnowledge(mode='cat')
        self.lin = torch.nn.Linear(hidden_ch * num_layers, out_ch)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        xs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs.append(x)
        x = self.jk(xs)
        return self.lin(x)
```

---

## Architecture Selection Flowchart

```
Is graph homogeneous?
├── YES
│   ├── Need inductive (new nodes)? → GraphSAGE
│   ├── Need attention/interpretability? → GAT
│   ├── Graph-level task? → GIN + pool
│   └── Simple baseline? → GCN
└── NO (heterogeneous)
    ├── Multiple node types? → HeteroConv / R-GCN
    ├── Knowledge graph? → CompGCN / RotatE
    └── Temporal? → TGN / EvolveGCN

Is graph large (>100K nodes)?
├── YES → NeighborLoader + GraphSAGE/ClusterGCN
└── NO → Full-batch GCN/GAT fine
```

---

## References

- [Graph Convolutional Networks (Kipf & Welling, 2017)](https://arxiv.org/abs/1609.02907) — Semi-supervised GCN
- [Graph Attention Networks (Veličković et al., 2018)](https://arxiv.org/abs/1710.10903) — Attention-based message passing
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io) — GNN library for PyTorch
