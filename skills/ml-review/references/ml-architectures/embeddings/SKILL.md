---
name: embeddings
description: Embedding models for retrieval and similarity — sentence-transformers, BGE, E5, OpenAI/Voyage APIs, plus rerankers and cross-modal (CLIP) matching. Use when building RAG retrieval, semantic search, deduplication, or any pipeline that maps unstructured data to dense vectors.
---

# Embeddings

## 1. Why This Exists

Text, images, and code are variable-length unstructured data. You can't compute "distance" between two paragraphs or sort images by similarity without a shared numeric representation.

Embeddings solve this by mapping inputs to fixed-size dense vectors (typically 384–4096 dimensions) where geometric proximity encodes semantic similarity. This enables:

- **Retrieval**: find the 10 most relevant documents for a query in <10ms over millions of vectors
- **Clustering**: group similar items without labels
- **Classification**: use vectors as features for downstream models
- **Deduplication**: detect near-duplicates via cosine threshold
- **Cross-modal search**: match text queries to images (CLIP/SigLIP)

| Without Embeddings | With Embeddings |
|---|---|
| BM25 keyword match (misses synonyms) | Semantic similarity captures meaning |
| O(N) pairwise comparison | ANN index: O(log N) or sublinear |
| Modality-locked (text↔text only) | Cross-modal (text↔image↔code) |

## 2. Text Embeddings

### Bi-Encoder vs Cross-Encoder

| Property | Bi-Encoder | Cross-Encoder |
|---|---|---|
| Architecture | Encode query & doc independently | Joint attention over (query, doc) pair |
| Speed | O(1) per doc (precompute) | O(N) -- must score every pair |
| Quality | Good (MTEB ~65-70) | Best (MTEB ~72-75) |
| Use case | First-stage retrieval | Reranking top-K |
| Scalability | Millions of docs | Top 20-100 candidates |

### Training Objectives

**Contrastive Loss (InfoNCE)**:
```
L = -log( exp(sim(q, d+)/τ) / Σ exp(sim(q, di)/τ) )
```
- `sim` = cosine similarity
- `τ` = temperature (typically 0.05-0.1)
- `d+` = positive, `di` = in-batch negatives + hard negatives

**Hard Negative Mining**: Random negatives are too easy. Mine hard negatives from:
1. BM25 top-K that aren't relevant (lexical overlap, wrong semantics)
2. Previous model's top-K false positives
3. Cross-encoder scores to find borderline cases

**Matryoshka Representation Learning (MRL)**: Train with loss applied at multiple dimension prefixes (e.g., 64, 128, 256, 768). At inference, truncate to smaller dims with minimal quality loss -- saves storage and speeds search.

## 3. Code Examples

### Sentence-Transformers Encode + Cosine

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

docs = ["embeddings map text to vectors", "neural networks learn representations"]
query = "how do embeddings work?"

doc_embeds = model.encode(docs, normalize_embeddings=True)
query_embed = model.encode(query, normalize_embeddings=True)

# Cosine similarity (dot product when normalized)
scores = query_embed @ doc_embeds.T
print(scores)  # [0.82, 0.61]
```

### FAISS Index (CPU, Flat + IVF)

```python
import faiss
import numpy as np

d = 768  # dimension
embeddings = np.random.randn(100_000, d).astype("float32")
faiss.normalize_L2(embeddings)

# Exact search (brute force, best recall)
index_flat = faiss.IndexFlatIP(d)
index_flat.add(embeddings)

# Approximate search (IVF, 10x faster, ~98% recall)
nlist = 256
quantizer = faiss.IndexFlatIP(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
index_ivf.train(embeddings)
index_ivf.add(embeddings)
index_ivf.nprobe = 16  # search 16 of 256 clusters

query = np.random.randn(1, d).astype("float32")
faiss.normalize_L2(query)
distances, indices = index_ivf.search(query, k=10)
```

### Milvus Insert + Search

```python
from pymilvus import MilvusClient
import numpy as np

client = MilvusClient(uri="http://localhost:19530")

# Create collection with auto-index
client.create_collection(
    collection_name="docs",
    dimension=768,
    metric_type="COSINE",
)

# Insert
vectors = np.random.randn(1000, 768).tolist()
data = [{"id": i, "vector": vectors[i], "text": f"doc_{i}"} for i in range(1000)]
client.insert(collection_name="docs", data=data)

# Search
query_vector = np.random.randn(1, 768).tolist()
results = client.search(
    collection_name="docs",
    data=query_vector,
    limit=10,
    output_fields=["text"],
)
```

### Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "what is machine learning?"
candidates = ["ML is a subset of AI", "Python is a language", "Neural nets learn patterns"]

# Score all pairs jointly
scores = reranker.predict([(query, doc) for doc in candidates])
# Rerank by score
ranked = sorted(zip(scores, candidates), reverse=True)
```

## 4. Model Comparison

| Model | Dims | MTEB Avg | Max Tokens | Notes |
|---|---|---|---|---|
| BAAI/bge-base-en-v1.5 | 768 | 63.6 | 512 | Best open-source base size |
| BAAI/bge-large-en-v1.5 | 1024 | 64.2 | 512 | Slightly better, 2x compute |
| intfloat/e5-large-v2 | 1024 | 62.0 | 512 | Strong zero-shot |
| Alibaba/gte-large-en-v1.5 | 1024 | 65.4 | 8192 | Long context, high quality |
| nomic-embed-text-v1.5 | 768 | 62.3 | 8192 | MRL-trained, truncatable to 256d |
| text-embedding-3-large (OpenAI) | 3072 | 64.6 | 8191 | API-only, MRL (truncate to 256/1024) |
| embed-english-v3.0 (Cohere) | 1024 | 64.5 | 512 | API-only, input_type parameter |
| voyage-large-2 (Voyage AI) | 1024 | 66.6 | 16000 | Top API model for code+text |

**Choosing a model**:
- Budget-constrained, self-hosted → BGE-base (768d, 110M params)
- Quality-first, self-hosted → GTE-large or BGE-large
- Long documents (>512 tokens) → GTE or Nomic (8K context)
- API acceptable → Voyage or OpenAI text-embedding-3-large
- Need dim reduction → Nomic or OpenAI (MRL-trained)

## 5. Fine-Tuning Embeddings

### Sentence-Transformers Trainer

```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import load_dataset

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Dataset: (anchor, positive) pairs -- negatives are in-batch
dataset = load_dataset("your-org/retrieval-pairs", split="train")

# Standard contrastive loss
loss = MultipleNegativesRankingLoss(model)

# Or Matryoshka loss (train multiple dim prefixes)
matryoshka_loss = MatryoshkaLoss(
    model, loss, matryoshka_dims=[768, 512, 256, 128, 64]
)

args = SentenceTransformerTrainingArguments(
    output_dir="./finetuned-bge",
    num_train_epochs=3,
    per_device_train_batch_size=128,  # larger batch = more negatives
    learning_rate=2e-5,
    warmup_ratio=0.1,
    bf16=True,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    loss=matryoshka_loss,
)
trainer.train()
```

### Hard Negative Mining During Training

```python
from sentence_transformers.util import mine_hard_negatives

# Mine hard negatives using the current model
dataset_with_negatives = mine_hard_negatives(
    dataset=dataset,
    model=model,
    num_negatives=5,      # negatives per positive
    range_min=10,         # skip top-10 (likely true positives)
    range_max=100,        # mine from rank 10-100
)
```

### When to Fine-Tune

| Scenario | Action |
|---|---|
| Domain vocab differs (legal, medical) | Fine-tune on domain pairs |
| Short queries → long docs | Fine-tune with asymmetric pairs |
| Performance plateau on your eval set | Mine hard negatives, retrain |
| Need smaller dims for cost | Train with Matryoshka loss |
| General English retrieval | Use pretrained BGE/GTE (already strong) |

## 6. Image & Cross-Modal Embeddings

### CLIP / SigLIP

Map images and text to the **same** vector space. A text query finds relevant images (and vice versa).

```python
from sentence_transformers import SentenceTransformer
from PIL import Image

# SigLIP via sentence-transformers (better than CLIP on fine-grained)
model = SentenceTransformer("google/siglip-base-patch16-224")

images = [Image.open("cat.jpg"), Image.open("dog.jpg")]
texts = ["a photo of a cat", "a photo of a dog"]

img_embeds = model.encode(images, normalize_embeddings=True)
txt_embeds = model.encode(texts, normalize_embeddings=True)

# Cross-modal similarity
scores = txt_embeds @ img_embeds.T  # text-to-image matching
```

| Model | Image Dims | Text Dims | Training | Best For |
|---|---|---|---|---|
| openai/clip-vit-large-patch14 | 768 | 768 | Contrastive (softmax) | General cross-modal |
| google/siglip-base-patch16-224 | 768 | 768 | Sigmoid (per-pair) | Fine-grained, batch-flexible |
| laion/CLIP-ViT-bigG-14-laion2B | 1280 | 1280 | Contrastive on 2B pairs | Highest quality open |

**SigLIP vs CLIP**: SigLIP uses sigmoid loss (independent per pair) instead of softmax (requires all pairs in batch). This means SigLIP works with any batch size and handles fine-grained distinctions better.

## 7. Retrieval Patterns

### Bi-Encoder → Cross-Encoder Pipeline

```
Query → Bi-Encoder → ANN Index (top 100) → Cross-Encoder Rerank → Top 10
         ~1ms            ~5ms                    ~50ms
```

This is the standard production pattern. Bi-encoder handles scale; cross-encoder handles quality.

### Hybrid Search (Sparse + Dense)

Combine BM25 (exact keyword match) with dense embeddings (semantic match):

```python
# Milvus hybrid search
from pymilvus import AnnSearchRequest, RRFRanker

# Dense vector search
dense_req = AnnSearchRequest(
    data=[query_embedding],
    anns_field="dense_vector",
    param={"metric_type": "COSINE", "params": {"nprobe": 16}},
    limit=100,
)

# Sparse vector search (BM25 or SPLADE)
sparse_req = AnnSearchRequest(
    data=[sparse_query],
    anns_field="sparse_vector",
    param={"metric_type": "IP"},
    limit=100,
)

# Reciprocal Rank Fusion
results = collection.hybrid_search(
    reqs=[dense_req, sparse_req],
    ranker=RRFRanker(k=60),
    limit=10,
)
```

### When to Use Hybrid

| Signal | Use Dense Only | Use Hybrid |
|---|---|---|
| Queries have entity names (product IDs, codes) | ❌ | ✅ |
| Pure semantic ("things like X") | ✅ | ✅ (no harm) |
| Domain with specialized vocab | ❌ | ✅ |
| Cold start (no training data) | ✅ | ✅ |
| Latency budget <5ms | ✅ | ❌ (2 searches) |

## 8. Gotchas

- **Always normalize** embeddings before cosine similarity. Without normalization, dot product ≠ cosine. Most models have `normalize_embeddings=True` flag.
- **Prefix instructions matter**: BGE models expect `"Represent this sentence: "` prefix for docs, `"Represent this sentence for searching: "` for queries. E5 uses `"query: "` / `"passage: "`. Check model card.
- **Dimension mismatch**: You cannot mix 768d and 1024d vectors in the same index. Pick one model and stick with it (or retrain with MRL for flexibility).
- **Batching**: Encode in batches of 256-1024 for GPU utilization. Single-item encode wastes 90%+ of GPU compute.
- **Token truncation**: Most models silently truncate at 512 tokens. For long docs, chunk into overlapping windows (256 tokens, 64 overlap) and store per-chunk vectors.
- **Index type matters**: FLAT = exact, IVF = fast approximate, HNSW = best recall/speed tradeoff for <10M vectors, DiskANN = billion-scale.
- **Don't fine-tune on <1K pairs**: Contrastive learning needs diversity. Under 1K pairs, you'll overfit. Use pretrained or augment with LLM-generated pairs.

## 9. References

1. [Sentence-Transformers Documentation](https://www.sbert.net/) -- Training, loss functions, model hub
2. [Sentence-Transformers GitHub](https://github.com/UKPLab/sentence-transformers) -- Source code and examples
3. [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084) -- Reimers & Gurevych, 2019: siamese BERT for semantic similarity
4. [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) -- Massive Text Embedding Benchmark
5. [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki) -- Index types, GPU support, quantization
6. [Milvus Documentation](https://milvus.io/docs) -- Managed vector DB, hybrid search, schema design
5. [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) -- Train once, truncate dims at inference
6. [BGE Paper (BAAI)](https://arxiv.org/abs/2309.07597) -- C-Pack training, instruction-tuned retrieval
7. [SigLIP Paper](https://arxiv.org/abs/2303.15343) -- Sigmoid loss for vision-language, replaces CLIP softmax
8. [Nomic Embed](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) -- Long-context, MRL, fully open weights+data
9. [Mining Hard Negatives (sentence-transformers)](https://www.sbert.net/docs/package_reference/util.html#sentence_transformers.util.mine_hard_negatives) -- API reference
10. [HNSW Paper](https://arxiv.org/abs/1603.09320) -- Hierarchical Navigable Small World graphs for ANN
