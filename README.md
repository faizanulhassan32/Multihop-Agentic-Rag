# Multi-Hop Agentic RAG

A locally running, agentic Retrieval-Augmented Generation (RAG) system designed to answer complex questions requiring information retrieval and reasoning across multiple documents.

## Overview

Traditional RAG systems often struggle with questions that require multiple retrieval steps and connecting information from different sources.

This project addresses these challenges through a multi-hop retrieval workflow that decomposes complex questions, retrieves relevant evidence, and generates answers using the collected context.

The system runs locally using Ollama, without relying on paid hosted inference APIs.

## Problems Addressed

- **Multi-hop reasoning:** Single-step retrieval can miss information spread across multiple documents.
- **Retrieval limitations:** Dense search can overlook exact keywords, names, and uncommon terms.
- **Inference latency:** Multiple sequential LLM calls increase response time.
- **Local resource constraints:** Running larger models locally requires efficient model and workflow management.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Agent orchestration | LangGraph, LangChain |
| Local inference | Ollama |
| Answer generation | Llama 3 |
| Supervisor | Qwen 2.5 1.5B |
| Embeddings | nomic-embed-text |
| Vector database | ChromaDB |
| Sparse retrieval | SQLite FTS5 BM25 |
| Frontend | React |
| Dataset | HotpotQA |

## Data Ingestion

The project uses a 500-example sample of the HotpotQA distractor dataset for development and evaluation.

The ingestion pipeline:

1. Loads and processes the dataset.
2. Prepares document chunks and assigns stable IDs.
3. Generates embeddings using the local Ollama embedding model.
4. Stores dense vectors in ChromaDB.
5. Builds a persistent SQLite FTS5 BM25 index using the same chunks and IDs.
6. Synchronizes both indexes and verifies ID consistency.

**Current ingestion results:**

| Metric | Count |
|---|---:|
| Dataset sample | 500 examples |
| Indexed chunks | 5,348 |
| ChromaDB documents | 5,348 |
| BM25 documents | 5,348 |
| Matching chunk IDs | Exact match |

**Storage locations:**
- ChromaDB: `backend/data/chroma/hotpotqa_distractor_500`
- BM25: `backend/data/bm25/index.sqlite3`

## Agent Workflow

The LangGraph workflow includes:

1. **Supervisor classification:** Handles request classification, scope, intent, and invalid or potentially unsafe inputs.
2. **Question decomposition:** Breaks complex questions into smaller sub-questions.
3. **Parallel retrieval:** Retrieves supporting documents for the sub-questions.
4. **Follow-up retrieval:** Performs additional retrieval when required.
5. **Final answer generation:** Generates an answer using the collected evidence.
6. **Supervisor finalization:** Handles the final response.

## Performance Optimization

The original workflow used multiple sequential LLM calls for intermediate reasoning and answer generation.

The optimization introduced:

- Parallel retrieval.
- Reduced intermediate LLM calls.
- Reuse of initialized LLM instances.
- Configurable retrieval and follow-up limits.
- Token-by-token streaming of final answers.

**Reported performance:**

| Metric | Before | After |
|---|---:|---:|
| Average latency | 17.669s | 6.995s |
| Median latency | 16.531s | 7.475s |
| P95 latency | 28.023s | 11.902s |
| LLM calls/query | 6-12 | 3 |

Results are based on a small development evaluation of six questions and three additional warm executions.

## Hybrid Retrieval

The agent runs Chroma dense search and persistent SQLite FTS5 BM25 search in parallel, then combines their ranked results using Reciprocal Rank Fusion (RRF).

Results are deduplicated using stable chunk IDs. Both retrieval backends expose these IDs through `Document.metadata["id"]`.

Defaults can be overridden in `backend/.env`:

- `DENSE_RETRIEVAL_K=10`
- `BM25_RETRIEVAL_K=10`
- `HYBRID_RETRIEVAL_K=5`
- `RRF_K=60`

The implementation includes backend failure handling and has been tested for ranking, deduplication, parallel execution, stable IDs, persistence, synchronization, and graph integration.

**Current implementation status:**
- Local ingestion and persistent indexes: Complete
- Supervisor integration: Complete
- Latency optimization and streaming: Complete
- Hybrid retrieval with BM25 and RRF: Complete