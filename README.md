# Multi-Hop Agentic RAG

A locally running, agentic Retrieval-Augmented Generation (RAG) system designed to answer complex questions requiring information retrieval and reasoning across multiple documents.

The project explores multi-hop question answering through question decomposition, hybrid retrieval, reranking, and evidence-based answer generation, using locally hosted language models.

## Overview

Traditional RAG systems often struggle with questions that require connecting information across multiple documents.

This project addresses these limitations through a multi-hop workflow that decomposes complex questions, retrieves relevant evidence, and generates answers from the collected context.

The system runs locally using Ollama, without relying on hosted inference APIs.

## Problems Addressed

- **Multi-hop reasoning:** Connecting information spread across multiple documents.
- **Retrieval limitations:** Combining semantic and keyword-based search.
- **Inference latency:** Reducing unnecessary sequential LLM calls.
- **Local resource constraints:** Optimizing inference and retrieval for local hardware.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Agent orchestration | LangGraph, LangChain |
| Local inference | Ollama |
| Generation model | Llama 3 |
| Supervisor model | Qwen 2.5 1.5B |
| Embeddings | nomic-embed-text |
| Vector database | ChromaDB |
| Sparse retrieval | SQLite FTS5 BM25 |
| Reranking | BGE Reranker Base |
| Frontend | React |
| Dataset | HotpotQA |

## Data Ingestion

The project uses a 500-example sample of the HotpotQA distractor dataset for development and evaluation.

The ingestion pipeline processes the dataset, creates document chunks, generates local embeddings, and builds synchronized ChromaDB and BM25 indexes.

**Current index:**
- 500 dataset examples
- 5,348 indexed document chunks
- Consistent document IDs across both retrieval indexes

Dataset files, embeddings, and local index artifacts are excluded from Git.

## Agent Workflow

The current LangGraph workflow follows a predefined multi-hop execution path:

1. **Supervisor classification:** Classifies requests and handles invalid or out-of-scope inputs.
2. **Question decomposition:** Breaks complex questions into smaller sub-questions.
3. **Query rewriting:** Refines sub-questions for retrieval.
4. **Sequential retrieval and reasoning:** Retrieves evidence and generates intermediate answers for each sub-question.
5. **Final answer generation:** Combines intermediate answers and supporting evidence.
6. **Supervisor finalization:** Handles the final response.

The current workflow is structured and sequential. More advanced planning, evidence sufficiency checks, and adaptive reasoning are not yet implemented.

## Performance Optimization

The existing workflow was optimized to reduce unnecessary LLM calls and improve response latency.

Key improvements include:
- Reduced intermediate generation calls.
- Reuse of initialized LLM instances.
- Configurable retrieval limits.
- Token-by-token streaming of final answers.

**Development benchmark:**

| Metric | Before | After |
|---|---:|---:|
| Average latency | 17.67s | 7.00s |
| Median latency | 16.53s | 7.48s |
| P95 latency | 28.02s | 11.90s |
| LLM calls per query | 6-12 | 3 |

These measurements represent an earlier development benchmark and are not a benchmark of the complete current system.

## Hybrid Retrieval

The retrieval layer combines dense vector search and BM25 keyword search.

**Retrieval pipeline:**

1. Dense retrieval using ChromaDB.
2. BM25 retrieval using SQLite FTS5.
3. Reciprocal Rank Fusion (RRF) to combine results.
4. Deduplication using stable document IDs.
5. BGE reranking to prioritize relevant evidence.

The system retrieves up to 10 candidates from each retrieval method, combines them, and passes the top 5 reranked documents to the agent.

**Retrieval evaluation:**

A preliminary 20-question HotpotQA evaluation showed that reranking improved the quality of the top-ranked results:

| Metric | Dense | Reranked |
|---|---:|---:|
| Recall@5 | 85.0% | 92.5% |
| Recall@10 | 85.0% | 95.0% |
| MRR@10 | 0.913 | 1.000 |
| nDCG@5 | 0.826 | 0.936 |

These results are preliminary and require validation on a larger evaluation set.

## Development Progress

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Project foundation and local RAG setup | Complete |
| Phase 2 | Supervisor integration and request classification | Complete |
| Phase 3 | Initial performance optimization and streaming | Complete |
| Phase 4 | Hybrid retrieval, reranking, and retrieval evaluation | Complete |
| Phase 5 | Agentic architecture improvements | Planned |
| Phase 6 | Evidence grounding and answer verification | Planned |
| Phase 7 | Reliability, observability, and final optimization | Planned |

**Deferred work:** Query expansion and MMR were considered but skipped because the existing hybrid retrieval and reranking pipeline provides a strong foundation. Additional retrieval techniques can be introduced later if evaluation identifies a need.

A separate retrieval-evaluation phase was skipped because retrieval performance was already evaluated in Phase 4.

## Next Step

Phase 5 will focus on restructuring the existing sequential workflow into a more adaptive agentic architecture.

The planned improvements include:
- Structured planning and task decomposition.
- Parallel execution of independent retrieval tasks.
- Dynamic task routing and dependency management.
- Evidence assessment and adaptive follow-up retrieval.
- Controlled reasoning and execution loops.

The goal is to move towards an orchestrator-worker architecture that can plan, delegate, evaluate intermediate results, and determine when additional retrieval is required before generating a final answer.

## Project Goal

The long-term goal is to develop a reliable, fully local agentic RAG system that can autonomously plan multi-hop questions, retrieve sufficient evidence, reason across multiple sources, and produce accurate, well-grounded answers.

The architecture will evolve based on measured weaknesses in the current system, rather than introducing additional agentic components without evidence.