# Phase 4.1 retrieval validation

These scripts evaluate the existing Chroma and BM25 indexes without rebuilding
or adding records to either index. BM25 evaluation connections use SQLite
read-only mode; the Chroma client only reads IDs and queries the configured
collection. Regenerable results are written under `backend/evaluation/results/`,
which is ignored by Git.

From `backend/`, run up to all 500 processed questions (questions with an
unmapped supporting sentence are excluded) with equal 10-result candidate/final
limits and RRF K=60:

```bash
.venv/bin/python evaluation/phase4_1_retrieval_eval.py \
  --limit 500 --dense-k 10 --bm25-k 10 --result-k 10 --rrf-k 60
```

Question selection is deterministic: the first N questions in the processed
HotpotQA JSONL file whose supporting-fact sentences map to indexed chunks are
used. The output records each question ID, both ranked result lists, relevance
labels, aggregate quality metrics, and latency measurements.

Run one live LangGraph smoke question using the configured local Ollama models:

```bash
.venv/bin/python evaluation/phase4_1_graph_smoke.py
```

The smoke test defaults to the first processed question and records retrieved
chunk IDs and whether retrieved evidence appeared in an answer-generation
prompt. It exercises the agent's native Ollama structured-output calls and
requires the configured local models. Both evaluation scripts leave the
production indexes unchanged.
