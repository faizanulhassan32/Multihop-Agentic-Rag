# Standalone BGE Reranker Service

Independent FastAPI service that reranks up to 20 submitted documents with
`BAAI/bge-reranker-base`. It does not import or modify the backend and does not
persist request content.

## Requirements and local run

Use Python 3.11:

```bash
cd reranker-service
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
cp .env.example .env
```

Set the variables in `.env` in your shell (or export them individually), then
start the service:

```bash
set -a
source .env
set +a
uvicorn reranker_service.main:app --host 127.0.0.1 --port 8010
```

The model is loaded once during application startup. The first startup downloads
the configured revision if it is not already present in the Hugging Face cache;
subsequent requests reuse the in-memory model and cached weights.

Run the mocked service tests without downloading model weights:

```bash
python -m unittest discover -s tests -v
```

## Live HotpotQA smoke test

The standalone smoke script uses the existing backend environment only for
read-only Chroma/BM25 hybrid retrieval. It starts and stops the reranker service
from this service's own virtual environment, sends 20 candidates, and validates
the top five IDs, scores, and ranks. The original candidate text is joined back
to each result using its returned ID, matching the API's ID/score/rank response.
On Apple Silicon, the default device is MPS:

```bash
# From the repository root, after backend ingestion and service installation:
PYTHONPATH=reranker-service/scripts backend/.venv/bin/python -m smoke_test --device mps
```

The script prints model loading time, service startup-to-ready time, logged
inference latency, HTTP round-trip time, and peak service-process RSS. Model
weights are cached under `reranker-service/.cache/huggingface/` by default.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-base` | CrossEncoder model name or local path |
| `RERANKER_MODEL_REVISION` | empty | Optional Hub branch, tag, or commit |
| `RERANKER_CACHE_DIR` | Hugging Face default | Optional model cache directory |
| `RERANKER_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N`, or `mps` |
| `RERANKER_BATCH_SIZE` | `16` | CrossEncoder inference batch size |
| `RERANKER_MAX_INPUT_LENGTH` | `512` | Maximum tokenized pair length; longer inputs are truncated |
| `RERANKER_MAX_CONCURRENT_REQUESTS` | `1` | Maximum simultaneous model inference calls |

`auto` chooses CUDA, then MPS, then CPU. Explicitly requesting an unavailable
accelerator causes model loading to fail and `/ready` to return `503`. For CUDA,
install a PyTorch build compatible with the host's CUDA runtime.

## API

`GET /health` reports process health. `GET /ready` reports whether model loading
succeeded. Reranking is available at `POST /v1/rerank`:

```json
{
  "query": "What is the capital of France?",
  "documents": [
    {"id": "doc-1", "text": "Paris is the capital of France."},
    {"id": "doc-2", "text": "Berlin is the capital of Germany."}
  ],
  "top_n": 2
}
```

The response contains the original IDs, raw CrossEncoder relevance scores, and
1-based ranks in descending score order. Scores are not softmax-normalized.
`top_n` is optional and defaults to all submitted documents.
