# Multi-Hop Agentic RAG

A locally running, agentic Retrieval-Augmented Generation (RAG) system designed to answer complex questions requiring information retrieval and reasoning across multiple documents.

## Overview

Traditional RAG systems often struggle with questions that require multiple retrieval steps and connecting information from different sources.

This project addresses these challenges through a multi-hop retrieval workflow that decomposes complex questions, retrieves supporting evidence, and generates answers using the collected context.

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
| Reranking | BGE Reranker Base |
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

Dataset, embeddings, indexes, and other local artifacts are excluded from Git.

## Agent Workflow

The current LangGraph workflow includes:

1. **Supervisor classification:** Handles request classification, scope, intent, and invalid or potentially unsafe inputs.
2. **Question decomposition:** Breaks complex questions into smaller sub-questions.
3. **Parallel retrieval:** Retrieves supporting documents for the sub-questions.
4. **Follow-up retrieval:** Performs additional retrieval when required.
5. **Final answer generation:** Generates an answer using the collected evidence.
6. **Supervisor finalization:** Handles the final response.

The current workflow uses a predefined graph structure. Dynamic planning, more advanced reasoning loops, and additional agentic capabilities are planned for subsequent phases.

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

Results are based on a small development evaluation of six questions and three additional warm executions. These measurements represent the agent workflow before the latest retrieval and reranking changes.

## Hybrid Retrieval

The retrieval layer combines semantic search with keyword-based search to improve evidence retrieval.

### Retrieval Pipeline

1. Dense retrieval using ChromaDB.
2. BM25 retrieval using SQLite FTS5.
3. Parallel execution of both retrieval methods.
4. Reciprocal Rank Fusion (RRF) to combine ranked results.
5. Deduplication using stable chunk IDs.
6. BGE reranking of fused candidates.
7. Return of the highest-ranked documents to the agent.

Both retrieval backends expose stable chunk IDs through `Document.metadata["id"]`.

### Configuration

Defaults can be overridden in `backend/.env`:

| Variable | Default |
|---|---:|
| `DENSE_RETRIEVAL_K` | 10 |
| `BM25_RETRIEVAL_K` | 10 |
| `HYBRID_RETRIEVAL_K` | 5 |
| `RRF_K` | 60 |

### Reranker

The project uses `BAAI/bge-reranker-base`, hosted as a standalone local service.

- Service directory: `reranker-service/`
- Health endpoint: `GET /health`
- Readiness endpoint: `GET /ready`
- Reranking endpoint: `POST /v1/rerank`
- Default service URL: `http://localhost:8010`
- Backend client: `backend/src/services/reranker_client.py`

The reranker scores the fused retrieval candidates and reorders them based on their relevance to the query.

The service is integrated into the existing retrieval pipeline without changing the agentic graph structure.

## Retrieval Evaluation

A 20-question HotpotQA evaluation was conducted to compare dense retrieval, BM25, Reciprocal Rank Fusion, and reranked hybrid retrieval.

The evaluation includes bridge and comparison questions across easy, medium, and hard difficulty levels.

### Results

| Metric | Dense | BM25 | RRF | Reranked |
|---|---:|---:|---:|---:|
| Recall@5 | 85.0% | 62.5% | 82.5% | **92.5%** |
| Recall@10 | 85.0% | 82.5% | 92.5% | **95.0%** |
| Recall@20 | N/A | N/A | 95.0% | 95.0% |
| MRR@10 | 0.913 | 0.788 | 0.854 | **1.000** |
| nDCG@5 | 0.826 | 0.616 | 0.757 | **0.936** |
| Evidence Coverage@5 | 85.8% | 64.2% | 82.5% | **93.3%** |

### Retrieval Latency

| Component | Mean Latency |
|---|---:|
| Dense retrieval | 23 ms |
| BM25 retrieval | 24 ms |
| Parallel retrieval | 27.5 ms |
| RRF fusion | 0.02 ms |
| Reranking | 396.9 ms |
| Total retrieval pipeline | 424.5 ms |

### Observations

- Dense retrieval remains a strong baseline, particularly for initial candidate retrieval.
- BM25 complements semantic retrieval by identifying keyword-relevant documents.
- RRF improves recall at larger retrieval depths, although its top-5 ranking does not consistently outperform dense retrieval alone.
- Reranking significantly improves the ordering of relevant documents within the final top-5 results.
- The reranker improves Recall@5, MRR@10, nDCG@5, and evidence coverage compared with the evaluated dense and hybrid baselines.

These results are preliminary and based on a 20-question evaluation sample. Larger evaluations are required to establish general performance.

## Current Phase 4 Progress

| Component | Status |
|---|---|
| Local ingestion and persistent indexes | Complete |
| Supervisor integration | Complete |
| Latency optimization and streaming | Complete |
| Persistent BM25 retrieval | Complete |
| Hybrid retrieval using RRF | Complete |
| Standalone reranker service | Complete |
| Reranker integration with hybrid retrieval | Complete |
| Retrieval evaluation | Complete (20-question sample) |
| End-to-end agent evaluation | Pending |
| Agentic architecture improvements | Planned |

## Next Steps

The next step is to evaluate the complete existing agent pipeline using the same evaluation questions.

This evaluation will measure answer correctness, multi-hop performance, evidence utilization, latency, and execution traces to identify the actual failure points in the current workflow.

These findings will guide subsequent improvements to the agentic architecture, including planning, evidence sufficiency, iterative retrieval, reasoning, and answer verification.

The agentic graph will remain unchanged until the end-to-end baseline has been established.
