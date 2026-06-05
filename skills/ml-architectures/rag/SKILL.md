---
name: rag
description: Retrieval-Augmented Generation — patterns for grounding LLMs in external knowledge. Term-based (BM25) vs embedding-based vs hybrid retrieval, chunking strategies, rerankers, query rewriting (HyDE, multi-query), faithfulness evaluation. Use when building knowledge-grounded LLM apps, deciding between RAG and long-context, debugging hallucinations or stale answers, or evaluating retrieval quality.
---

# Retrieval-Augmented Generation (RAG)

## Why This Exists

**Problem.** LLMs hallucinate when they don't know. Three common failure modes:

1. **Stale knowledge.** Training cuts off; the world keeps moving. Ask about events, prices, schemas, or staff after the cutoff and you get fiction.
2. **Private/long-tail facts.** Your customer's invoice number, your codebase's APIs, a 2018 internal RFC — none of this was in pretraining.
3. **Per-user context.** A general model doesn't know which Slack thread the user means by "that bug."

Naive responses don't fix this:

- **Finetuning** teaches the model *form* (style, format, refusal behavior, domain phrasing), not *facts*. New facts can be memorized through finetuning, but updating one fact means retraining; retrieval lets you `INSERT` a row.
- **Long context** ($\geq$ 200K tokens) helps, but every extra token costs money and latency, and models still focus on the wrong parts of giant contexts ("lost in the middle"). Anthropic's own guidance: under ~200K tokens of corpus, just stuff it all in; over that, retrieve.

**Key insight (Huyen).** *"Finetuning is for form, RAG is for facts."*

**Reach for RAG when:**

- Your knowledge changes after the training cutoff or has to be editable in seconds.
- You need per-user / per-tenant context isolation.
- The model fails on long-tail names, IDs, error codes, or private documents.
- Your corpus is bigger than the context window, *or* fits but you'd rather pay for retrieval than for tokens.

**Don't reach for RAG when:**

- Corpus fits in context (Anthropic's 200K rule of thumb) and stays fresh enough — just include it.
- The failure is behavioral (wrong tone, won't follow format, refuses safe asks) — that's prompting or finetuning.
- You need answers from structured tables — use SQL agents (text-to-SQL), not chunked retrieval over CSVs.
- Latency budget < 100ms end-to-end and corpus is small — retrieve once, cache, or precompute.

---

## RAG Architecture

A RAG system has two pipelines that share an index:

```
INDEXING (offline, run once per corpus update)
  raw docs → loader → chunker → embedder → vector DB
                              ↘ tokenizer → BM25 index

QUERY (online, run per request)
  user query → query rewriter → retriever(s) → reranker → prompt assembly → LLM → answer
                                  ↑ vector DB
                                  ↑ BM25 index
```

| Component   | Job                                             | Common choices                                      |
|-------------|-------------------------------------------------|-----------------------------------------------------|
| Loader      | Parse PDF/HTML/MD/code into text                | `unstructured`, `pypdf`, `trafilatura`, LlamaParse  |
| Chunker     | Split docs into retrievable units               | recursive char, semantic, doc-structure-aware       |
| Embedder    | Map chunk → dense vector                        | `bge-large`, `e5-large-v2`, `text-embedding-3-large`|
| Vector DB   | Index + ANN search                              | FAISS, Qdrant, Weaviate, Milvus, pgvector, Pinecone |
| Lex index   | Inverted index for term match                   | Elasticsearch, OpenSearch, tantivy, `rank_bm25`     |
| Retriever   | Score and return top-K                          | dense, BM25, hybrid (RRF / weighted)                |
| Reranker    | Re-score top-K with stronger model              | `bge-reranker-v2-m3`, Cohere Rerank, ColBERT        |
| Generator   | Compose answer from query + retrieved chunks    | any chat LLM                                        |

The retriever has two functions: **indexing** (preprocess corpus) and **querying** (find relevant docs at request time). Quality of the whole system is bounded by retrieval quality — a perfect generator over wrong docs still hallucinates.

---

## Retrieval Algorithms

### Term-based (sparse, lexical)

Score documents by token-level overlap with the query. Cheap, fast, no training, strong baseline.

- **TF-IDF.** Score = (term frequency in doc) × log(N / docs containing term). Rare informative terms outweigh stop-words.
- **BM25 (Okapi).** TF-IDF normalized for document length, with TF saturation. Variants: **BM25+** (handles edge case of long-doc bias), **BM25F** (per-field weights — weight title higher than body).
- **Implementations.** Elasticsearch / OpenSearch (production), tantivy (Rust), `rank_bm25` (Python, in-memory, fine up to ~100K small docs).

**Strengths.** Exact-match recall (error codes, product SKUs, function names like `EADDRNOTAVAIL`), zero training cost, sub-ms latency, interpretable. Strong baseline that consistently competes with dense on domain-shifted data (BEIR benchmark).

**Weaknesses.** No semantics — "transformer architecture" matches the movie. Vocabulary mismatch — query "car" misses doc "automobile."

### Embedding-based (dense)

Bi-encoder maps query and docs into the same vector space; similarity = cosine or dot product. Index uses approximate nearest neighbor (ANN) for sublinear search.

| ANN algorithm        | How it works                                        | Trade-off                            |
|----------------------|-----------------------------------------------------|--------------------------------------|
| Flat (brute force)   | Compare query to all vectors                        | Exact, slow; OK up to ~100K vectors  |
| **HNSW**             | Multi-layer graph of neighbors                      | Fast, high recall; high memory       |
| **IVF**              | K-means partitions; search top-N clusters           | Tunable speed/recall; fast build     |
| **IVF-PQ**           | IVF + product quantization (compress vectors)       | Big memory savings; slight recall hit|
| **ScaNN**            | Anisotropic quantization (Google)                   | Best on benchmarks for large corpora |
| **LSH**              | Hash similar vectors to same bucket                 | Simple; lower recall than HNSW       |

For most workloads, start with **HNSW** (high recall, simple to tune) or **IVF-PQ** (when memory matters at >10M vectors). FAISS implements all of the above; `faiss.index_factory("IVF1024,PQ32")` builds a typical large-scale index.

**Strengths.** Captures synonyms and paraphrases. Improves with finetuning. Handles natural-language queries.

**Weaknesses.** Cost (embedding inference, storage, vector search). Loses exact-match precision. Reindexing is expensive if you change the embedding model. Domain shift hurts hard — out-of-domain dense retrieval often loses to BM25.

### Sparse embeddings (SPLADE)

SPLADE generates **sparse** vectors via BERT + L1 regularization that pushes most dimensions to zero. The non-zero dimensions correspond to vocabulary terms — interpretable, indexable in an inverted file, and competitive with dense retrieval. Best of both: term-grounded but learned.

Use SPLADE when you want dense-quality retrieval but with the inverted-index economics of BM25 (very large corpora, distributed search).

### Hybrid retrieval

Run BM25 and dense in parallel, fuse their rankings. The standard fusion: **reciprocal rank fusion (RRF)**.

$$\text{RRF}(d) = \sum_{i=1}^{n} \frac{1}{k + \text{rank}_i(d)}$$

where $k$ is typically 60. Documents ranked high by both retrievers float to the top.

```python
import numpy as np
from collections import defaultdict

def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """rankings: list of ordered doc-id lists, one per retriever (highest-ranked first)."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Example:
bm25_top = ["d3", "d1", "d7", "d2"]
dense_top = ["d1", "d4", "d3", "d9"]
fused = reciprocal_rank_fusion([bm25_top, dense_top])
# d1 and d3 win (ranked highly by both)
```

**Weighted-sum fusion** (alternative): normalize each retriever's scores to [0,1], then `score = α·dense + (1−α)·bm25`. RRF is more robust because raw scores are not comparable across retrievers.

### Rerankers (cross-encoders)

A bi-encoder is fast (encode independently) but loses query↔doc interaction. A **cross-encoder** takes `(query, doc)` jointly and outputs a score — much higher precision, but O(N) per query (no precomputed index).

**Standard pattern.** Retrieve top-100 with cheap retriever (BM25 or dense or hybrid) → rerank with cross-encoder → keep top-5 for the prompt. The quality lift is usually large; on BEIR-class benchmarks, reranking top-100 is often worth more than switching the base retriever.

| Reranker                        | Notes                                          |
|---------------------------------|------------------------------------------------|
| `BAAI/bge-reranker-v2-m3`       | Open, multilingual, strong default             |
| Cohere Rerank API               | Easy, paid, strong on English                  |
| `cross-encoder/ms-marco-MiniLM` | Tiny, CPU-friendly                             |
| **ColBERT / ColBERTv2**         | Late-interaction; per-token vectors; both fast and cross-encoder-quality |

ColBERT stores per-token embeddings and computes MaxSim at query time — better recall than bi-encoders, faster than full cross-encoders. Worth it when you need both speed and precision at scale.

---

## Decision Table: Which Retrieval Approach?

| Situation                                      | Pick                                    | Why                                       |
|------------------------------------------------|-----------------------------------------|-------------------------------------------|
| <10K docs, simple FAQ, English                 | BM25 only                               | Cheap, fast, good enough                  |
| Names, IDs, error codes, code search           | BM25 (or hybrid)                        | Embeddings blur exact tokens              |
| Natural-language Qs over prose                 | Dense (HNSW)                            | Semantic match wins                       |
| Mixed (most production cases)                  | Hybrid: BM25 + dense + RRF              | Robust to query type                      |
| Top-K precision matters (top-3 context window) | Hybrid + cross-encoder rerank           | Reranker squeezes the most from candidates|
| 100M+ docs, latency-critical                   | SPLADE or IVF-PQ (+ rerank top-50)      | Inverted-index economics                  |
| Multilingual                                   | `bge-m3` or `e5-multilingual` + rerank  | Multi-vec models built for it             |
| Highly specialized domain (medical, legal)     | Hybrid + finetuned embedder + rerank    | OOD breaks generic dense models           |

---

## Chunking Strategies

How you split docs determines what the retriever can see. Bad chunking caps system quality.

| Strategy                       | Description                                              | When to use                                 |
|--------------------------------|----------------------------------------------------------|---------------------------------------------|
| Fixed-size (chars/tokens)      | Split every N units                                      | Quick prototype; uniform content            |
| **Recursive character**        | Try paragraph → sentence → word splits until under size  | Default for prose; LangChain's standard     |
| Sentence-aware                 | NLTK/spaCy sentence boundaries                           | When sentence integrity matters             |
| **Sliding window with overlap**| Fixed size + 10–20% overlap                              | Preserves cross-boundary context            |
| Document-structure-aware       | Split on `<h1>` / `##` / code-block / table boundaries   | Markdown, HTML, technical docs              |
| Code-aware                     | tree-sitter splits at function/class boundaries          | Code search                                 |
| **Semantic chunking**          | Split where embedding similarity drops between sentences | Long unstructured prose; cost more to index |
| Q&A pair                       | Each (question, answer) is a chunk                       | FAQs, support tickets                       |

### Chunk-size trade-offs

- **Smaller chunks (128–256 tokens):** higher recall, more diverse evidence in context, but lose surrounding context and double indexing cost. Better for embedding-based retrieval where you'll fetch many.
- **Larger chunks (512–1024 tokens):** more self-contained, fewer chunks to index, but coarser retrieval — irrelevant material rides along.
- **Overlap (10–20% of chunk size):** prevents splits like "I left my wife" / "a note." Set 50–200 char overlap on 1000-char chunks.

| Document type           | Suggested chunk size  | Overlap   | Splitter                          |
|-------------------------|-----------------------|-----------|-----------------------------------|
| FAQ / support articles  | per Q&A pair          | none      | structural                        |
| Long-form prose / books | 512–1024 tokens       | 10–15%    | recursive character or semantic   |
| Markdown technical docs | section / sub-section | none      | structure-aware (`MarkdownSplit`) |
| Source code             | function / class      | none      | tree-sitter / language-aware      |
| Chat transcripts        | per turn or N-turn    | 1 turn    | role-aware                        |
| Tables                  | one row + headers     | none      | row-level (or text-to-SQL)        |

**Constraint:** `chunk_size <= min(embedder_max_tokens, generator_context_window / K)` where K is the number of chunks you'll stuff in.

---

## Embedding Model Selection

| Model                              | Dim   | Notes                                            |
|------------------------------------|-------|--------------------------------------------------|
| `BAAI/bge-large-en-v1.5`           | 1024  | Strong English open-weight default               |
| `BAAI/bge-m3`                      | 1024  | Multilingual + sparse + dense in one model       |
| `intfloat/e5-large-v2`             | 1024  | Strong English, requires `query:` / `passage:` prefix |
| `intfloat/multilingual-e5-large`   | 1024  | Multilingual                                     |
| `Alibaba-NLP/gte-large-en-v1.5`    | 1024  | Long-context (8K), strong                        |
| OpenAI `text-embedding-3-large`    | 3072* | API; supports Matryoshka truncation              |
| OpenAI `text-embedding-3-small`    | 1536* | Cheaper, surprisingly strong                     |
| Cohere `embed-v3`                  | 1024  | Multilingual; compressed-int8 mode               |
| Voyage `voyage-3` / `voyage-code-2`| 1024  | Strong on retrieval; code-specialized variant    |

*OpenAI v3 supports the `dimensions` parameter — truncate on read with little quality loss.

**Selection criteria:**

1. **Language coverage** — match the corpus and queries.
2. **MTEB retrieval score** — use the leaderboard, but evaluate on *your* data.
3. **Dimension** — larger ≠ better; trade against storage (1B × 1024 × 4B = 4 TB).
4. **Max sequence length** — 512 is common; 8K models like `gte-large-en-v1.5` for long chunks.
5. **License & deployment** — open-weight (BGE/E5/GTE) for sovereignty, API for ease.
6. **Domain finetuning** — for specialized corpora (legal, medical, code), finetune with `sentence-transformers` MultipleNegativesRanking on (query, positive) pairs. A few thousand pairs typically beat any off-the-shelf model on your domain.

---

## Query Rewriting and Expansion

The user's literal query is often a poor retrieval input.

### Conversational rewriting

```
User: When was the last time John Doe bought something from us?
AI:   January 3, 2030.
User: How about Emily Doe?           ← unintelligible to a retriever in isolation
```

Rewrite with an LLM: *"Given the conversation, rewrite the last user input as a standalone search query."* → `When was the last time Emily Doe bought something from us?`

### HyDE (Hypothetical Document Embeddings)

Have the LLM hallucinate a *plausible answer document* to the query, then embed *that* and retrieve neighbors. Works because docs and answers live closer in embedding space than questions and answers.

```python
def hyde_retrieve(query: str, llm, embedder, vector_db, k: int = 5):
    hypothetical = llm.generate(f"Write a paragraph that answers: {query}")
    return vector_db.search(embedder.encode(hypothetical), k=k)
```

Best when queries are short/keyword-y and docs are long-form prose. Adds one LLM call of latency.

### Multi-query

Have the LLM generate N paraphrases of the query, retrieve for each, fuse with RRF. Improves recall when phrasing matters.

### Sub-question decomposition

Break a complex query into atomic sub-questions, retrieve for each, then have the generator compose. Critical for multi-hop questions like *"Compare the revenue growth of Acme and Globex in 2023."*

---

## Advanced Patterns

### Contextual retrieval (Anthropic, 2024)

Plain chunks lose context — chunk #47 of a contract doesn't know it's clause 12 of the Acme MSA. Anthropic's fix: prepend a 50–100 token LLM-generated context to each chunk *before* indexing.

```
<document>{full_document}</document>
Here is the chunk we want to situate within the whole document:
<chunk>{chunk}</chunk>
Please give a short succinct context to situate this chunk within the
overall document for the purposes of improving search retrieval.
Answer only with the succinct context and nothing else.
```

Anthropic reported ~35% reduction in retrieval failures with this + hybrid + reranking. Cost: one cheap LLM call per chunk at index time (use prompt caching to keep it under control).

### Multi-vector / late-interaction (ColBERT)

Instead of one vector per chunk, store one per token. At query time, for each query token, find max similarity to any doc token, sum across query tokens. Captures fine-grained matches that a single pooled vector loses. ColBERTv2 + PLAID makes this production-feasible.

### Graph-RAG

Build a knowledge graph from the corpus (entities + relations); retrieve subgraphs instead of (or alongside) chunks. Microsoft GraphRAG reports better answers on questions that require synthesizing across many documents. Heavy: needs an extraction pipeline.

### Agentic RAG

Let an agent decide which tool to use per query — vector search, BM25, SQL, web search, calculator. The "retriever" is no longer a fixed step but a tool the LLM picks. See `../agents/` for tool-selection patterns and ReAct loops.

---

## Evaluation

Evaluate the **retriever** and the **end-to-end system** separately. Failures look different.

### Retrieval metrics

Curate an eval set: queries paired with relevance labels for candidate docs. Then:

| Metric          | What it measures                                              |
|-----------------|---------------------------------------------------------------|
| **Recall@k**    | Fraction of relevant docs found in top-k                      |
| **Precision@k** | Fraction of top-k that are relevant                           |
| **MRR**         | Mean reciprocal rank of first relevant doc                    |
| **nDCG@k**      | Position-discounted relevance (gold for graded relevance)     |
| **Hit rate@k**  | Was at least one relevant doc in top-k? (binary)              |

Recall is hard to compute exactly — you'd need every doc-query pair labeled. In practice approximate it: pool top-K from several strong retrievers and label only that pool.

### End-to-end (RAG triad)

The Ragas / TruLens "RAG triad":

1. **Context relevance** — were the retrieved chunks relevant to the query?
2. **Faithfulness / groundedness** — does the answer come from the context (no hallucination)?
3. **Answer relevance** — does the answer actually address the question?

Each is scored by an LLM-as-judge (GPT-4-class). Faithfulness is the headline metric for RAG — a relevant-looking answer that's not grounded in the context defeats the purpose.

### Synthetic eval set generation

You usually don't have labeled queries. Generate them:

```
For each chunk:
  prompt the LLM: "Generate 3 questions a user might ask whose
  answer is in this chunk. Return JSON."
  → (chunk, question) pairs serve as gold (chunk, query) eval data
```

Filter aggressively: drop ambiguous questions, near-duplicates, questions answerable from any chunk. Hand-review a sample. See `../../ml-training/llm-evaluation/` for the depth on LLM-as-judge calibration, golden-set construction, and bias mitigation.

---

## Code: End-to-End RAG (Raw)

Build it from primitives once before reaching for LangChain or LlamaIndex — you'll understand what they hide.

### Indexing (BM25 + dense + FAISS)

```python
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load and chunk
documents = [...]  # list of (doc_id, text)
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
chunks = []  # list of {"id": str, "text": str, "doc_id": str}
for doc_id, text in documents:
    for i, chunk in enumerate(splitter.split_text(text)):
        chunks.append({"id": f"{doc_id}#{i}", "text": chunk, "doc_id": doc_id})

texts = [c["text"] for c in chunks]

# 2. Dense index (FAISS HNSW)
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
dim = embeddings.shape[1]
dense_index = faiss.IndexHNSWFlat(dim, 32)  # M=32 neighbors per node
dense_index.hnsw.efConstruction = 200
dense_index.add(embeddings.astype("float32"))

# 3. Sparse index (BM25)
tokenized = [t.lower().split() for t in texts]  # use a real tokenizer in prod
bm25 = BM25Okapi(tokenized)
```

### Querying (hybrid + rerank)

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def hybrid_search(query: str, k_dense: int = 50, k_bm25: int = 50,
                  k_final: int = 5) -> list[dict]:
    # Dense top-K
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    _, dense_ids = dense_index.search(q_emb, k_dense)
    dense_ranking = [chunks[i]["id"] for i in dense_ids[0]]

    # BM25 top-K
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ids = np.argsort(bm25_scores)[::-1][:k_bm25]
    bm25_ranking = [chunks[i]["id"] for i in bm25_ids]

    # RRF fusion
    fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=60)
    candidate_ids = [doc_id for doc_id, _ in fused[:50]]
    candidates = [c for c in chunks if c["id"] in set(candidate_ids)]

    # Cross-encoder rerank
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:k_final]]
```

### Generation

```python
def answer(query: str, llm) -> str:
    docs = hybrid_search(query)
    context = "\n\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(docs))
    prompt = f"""Answer using ONLY the context. If the context does not contain
the answer, say "I don't know." Cite sources by [number].

Context:
{context}

Question: {query}
Answer:"""
    return llm.generate(prompt)
```

Notes:
- Always normalize embeddings and use inner product (or cosine) — mixing metrics silently wrecks recall.
- Tokenize for BM25 with a real tokenizer (NLTK, spaCy, or Lucene's analyzer) — `.split()` is a placeholder.
- Cross-encoder reranks 50 → 5; doing it on 1000 candidates is wasteful and slow.
- Force the model to say "I don't know" — without that, faithfulness collapses.

### BM25 baseline (minimal)

```python
from rank_bm25 import BM25Okapi

corpus = ["fancy printer A300 specifications",
          "transformer architecture attention is all you need",
          "transformers the movie 2007 michael bay"]
bm25 = BM25Okapi([d.lower().split() for d in corpus])
print(bm25.get_top_n("transformer architecture".lower().split(), corpus, n=2))
# → ['transformer architecture attention is all you need', 'transformers the movie ...']
```

---

## Framework Choice: LangChain vs LlamaIndex vs DIY

| Framework      | Strengths                                                         | Use when                                  |
|----------------|-------------------------------------------------------------------|-------------------------------------------|
| **DIY** (above)| Full control, no abstraction tax, easy to debug                   | You're learning, or you have a tight loop |
| **LlamaIndex** | Best-in-class indexing primitives, query engines, retrievers      | Doc-heavy RAG, complex query routing      |
| **LangChain**  | Broad integrations, agents, tooling, examples                     | Multi-tool agentic flows, fast prototyping|
| Haystack       | Production-leaning, strong eval                                   | Enterprise pipelines                      |

Both LangChain and LlamaIndex churn rapidly — pin versions and verify the tutorial you're reading matches your install.

---

## Common Failure Modes

| Symptom                                            | Likely cause                                  | Fix                                              |
|----------------------------------------------------|-----------------------------------------------|--------------------------------------------------|
| Answer ignores retrieved context                   | Prompt doesn't enforce grounding              | "Answer ONLY from context; else say I don't know"|
| Wrong docs retrieved on exact-token queries        | Pure dense retrieval                          | Add BM25 + RRF                                   |
| Top-1 is right, top-5 mixes garbage                | Decent retriever, no reranker                 | Add cross-encoder rerank on top-50               |
| Recall low across the board                        | Bad chunking (too big, no overlap)            | Recursive splitter, 256–512 tokens, 10–15% overlap|
| Worked in dev, broke in prod                       | Prod corpus is OOD vs benchmark embedder      | Finetune embedder; or contextual retrieval        |
| Context too long, answers slow / wrong             | Stuffing top-20 chunks                        | Rerank → top-3-5; smaller chunks                 |
| Stale answers                                      | Index not refreshed                           | Incremental indexing pipeline; TTLs              |
| "I don't know" all the time                        | Reranker filtering everything; threshold too high | Lower threshold; inspect reranker scores      |
| Hallucinates citations                             | Generator inventing source IDs                | Force structured output with allowed IDs         |

---

## See Also

- `../embeddings/` — canonical home for embedding models, bi-encoders, cross-encoders, and CLIP/SigLIP for multimodal retrieval
- `../agents/` — agentic RAG, tool selection, ReAct, planner/executor patterns
- `../../ml-training/prompt-engineering/` — prompt templates for grounded answering, citation, refusal
- `../../ml-training/llm-evaluation/` — LLM-as-judge calibration, faithfulness/answer-relevance scoring, eval-set construction
- `../../ml-libraries/huggingface/` — `sentence-transformers`, `transformers`, model hub
- `../../ml-libraries/litellm/` — unified LLM client (OpenAI/Anthropic/local) for the generator step

---

## References

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020): https://arxiv.org/abs/2005.11401
- Gao et al., *Retrieval-Augmented Generation for Large Language Models: A Survey* (2023): https://arxiv.org/abs/2312.10997
- Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE, 2022): https://arxiv.org/abs/2212.10496
- Formal et al., *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking* (2021): https://arxiv.org/abs/2107.05720
- Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT* (2020): https://arxiv.org/abs/2004.12832
- Anthropic, *Introducing Contextual Retrieval* (2024): https://www.anthropic.com/news/contextual-retrieval
- LlamaIndex: https://github.com/run-llama/llama_index
- LangChain RAG tutorial: https://python.langchain.com/docs/tutorials/rag/
- Ragas (RAG evaluation): https://github.com/explodinggradients/ragas
- FAISS (Facebook AI Similarity Search): https://github.com/facebookresearch/faiss
- MTEB Leaderboard (embedding benchmark): https://huggingface.co/spaces/mteb/leaderboard
