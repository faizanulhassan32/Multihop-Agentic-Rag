#!/usr/bin/env python3
"""Run a real HotpotQA hybrid-retrieval-to-reranker smoke test."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import re
import resource
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SERVICE_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = SERVICE_DIR.parent
BACKEND_DIR = REPOSITORY_DIR / "backend"
BACKEND_SRC = REPOSITORY_DIR / "backend" / "src"
sys.path.insert(0, str(BACKEND_SRC))

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from agent.hybrid_retriever import hybrid_search
from ingestion.config import Config
from ingestion.preprocess import PROCESSED_SAMPLE_PATH


MODEL_NAME = "BAAI/bge-reranker-base"
MODEL_LOAD_LOG = re.compile(
    r"Loaded reranker model .* in ([0-9.]+) seconds\."
)
INFERENCE_LOG = re.compile(
    r"Reranker inference completed for \d+ documents in ([0-9.]+) ms\."
)


def _backend_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = BACKEND_DIR / path
    return path.resolve()


def _read_question(question_index: int) -> tuple[str | None, str]:
    if question_index < 0:
        raise ValueError("--question-index must be zero or greater.")
    if not PROCESSED_SAMPLE_PATH.is_file():
        raise FileNotFoundError(
            f"Processed HotpotQA data not found: {PROCESSED_SAMPLE_PATH}"
        )

    with PROCESSED_SAMPLE_PATH.open(encoding="utf-8") as questions_file:
        for index, line in enumerate(questions_file):
            if not line.strip():
                continue
            if index != question_index:
                continue
            record = json.loads(line)
            question = record.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Processed question at index {question_index} is invalid."
                )
            question_id = record.get("id")
            return (question_id if isinstance(question_id, str) else None, question)

    raise IndexError(
        f"Question index {question_index} is outside {PROCESSED_SAMPLE_PATH}."
    )


async def _retrieve_candidates(query: str):
    chroma_path = _backend_path(Config.CHROMA_PERSIST_DIR)
    bm25_path = _backend_path(Config.BM25_DB_PATH)
    if not chroma_path.is_dir():
        raise FileNotFoundError(f"Chroma index not found: {chroma_path}")
    if not bm25_path.is_file():
        raise FileNotFoundError(f"BM25 index not found: {bm25_path}")

    chroma_client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )
    chroma_client.get_collection(Config.CHROMA_COLLECTION_NAME)
    vectorstore = Chroma(
        client=chroma_client,
        embedding_function=OllamaEmbeddings(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_EMBEDDING_MODEL,
        ),
        collection_name=Config.CHROMA_COLLECTION_NAME,
        create_collection_if_not_exists=False,
    )
    return await hybrid_search(
        query=query,
        vectorstore=vectorstore,
        dense_limit=20,
        bm25_limit=20,
        result_limit=20,
        rrf_constant=Config.RRF_K,
        bm25_db_path=bm25_path,
    )


def _reserve_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _get_json(url: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def _wait_for_ready(process: subprocess.Popen, url: str, timeout: float) -> tuple[dict, float]:
    started_at = time.perf_counter()
    deadline = started_at + timeout
    last_error = "service did not respond"
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Reranker service exited with status {process.returncode}.")
        try:
            response = _get_json(url)
            return response, time.perf_counter() - started_at
        except HTTPError as exc:
            last_error = exc.read().decode("utf-8", errors="replace")
            if exc.code != 503:
                raise
            try:
                detail = json.loads(last_error).get("detail", {})
            except json.JSONDecodeError:
                detail = {}
            if isinstance(detail, dict) and detail.get("error"):
                raise RuntimeError(
                    f"Reranker model failed to load: {detail['error']}"
                ) from exc
        except URLError as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"Reranker service was not ready within {timeout:g}s: {last_error}")


def _stop_service(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _read_logs(log_file) -> str:
    log_file.flush()
    log_file.seek(0)
    return log_file.read()


def _log_duration(pattern: re.Pattern, logs: str, label: str) -> float:
    matches = pattern.findall(logs)
    if not matches:
        raise RuntimeError(f"Could not read {label} from reranker service logs.")
    return float(matches[-1])


def _cached_model_weights(cache_dir: Path) -> bool:
    model_cache = cache_dir / "models--BAAI--bge-reranker-base" / "snapshots"
    if not model_cache.is_dir():
        return False
    weight_names = ("model.safetensors", "pytorch_model.bin")
    return any(
        (snapshot / name).is_file()
        for snapshot in model_cache.iterdir()
        if snapshot.is_dir()
        for name in weight_names
    )


def _peak_child_rss_mib() -> float:
    peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
    return round(peak_bytes / (1024 * 1024), 1)


def _run(args: argparse.Namespace) -> dict:
    question_id, query = _read_question(args.question_index)
    candidates = asyncio.run(_retrieve_candidates(query))
    if len(candidates) != 20:
        raise RuntimeError(
            f"Hybrid retrieval returned {len(candidates)} candidates; expected 20."
        )

    candidate_ids = [document.metadata.get("id") for document in candidates]
    if any(not isinstance(document_id, str) or not document_id for document_id in candidate_ids):
        raise RuntimeError("Hybrid retrieval returned a candidate without a stable ID.")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("Hybrid retrieval returned duplicate candidate IDs.")
    candidates_by_id = dict(zip(candidate_ids, candidates))

    service_python = Path(args.service_python).expanduser()
    if not service_python.is_absolute():
        service_python = SERVICE_DIR / service_python
    if not service_python.is_file():
        raise FileNotFoundError(
            f"Service Python not found: {service_python}. "
            "Create reranker-service/.venv and install this service first."
        )

    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else Path(
        os.environ.get(
            "RERANKER_CACHE_DIR",
            str(SERVICE_DIR / ".cache" / "huggingface"),
        )
    ).expanduser()
    cache_dir = cache_dir.resolve()
    weights_cached_before = _cached_model_weights(cache_dir)

    port = args.port or _reserve_port()
    base_url = f"http://127.0.0.1:{port}"
    service_env = os.environ.copy()
    service_env["RERANKER_MODEL_NAME"] = MODEL_NAME
    service_env["RERANKER_DEVICE"] = args.device
    service_env["RERANKER_CACHE_DIR"] = str(cache_dir)
    service_command = [
        str(service_python),
        "-m",
        "uvicorn",
        "reranker_service.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "info",
    ]

    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as service_logs:
        process = subprocess.Popen(
            service_command,
            cwd=SERVICE_DIR,
            env=service_env,
            stdout=service_logs,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            ready, startup_to_ready_seconds = _wait_for_ready(
                process,
                f"{base_url}/ready",
                args.startup_timeout,
            )
            if ready.get("model") != MODEL_NAME:
                raise RuntimeError(
                    f"Service loaded unexpected model: {ready.get('model')!r}."
                )

            request_payload = {
                "query": query,
                "documents": [
                    {"id": document.metadata["id"], "text": document.page_content}
                    for document in candidates
                ],
                "top_n": 5,
            }
            submitted_text_by_id = {
                document["id"]: document["text"]
                for document in request_payload["documents"]
            }
            request_started = time.perf_counter()
            response = _get_json(f"{base_url}/v1/rerank", request_payload)
            http_round_trip_ms = (time.perf_counter() - request_started) * 1000
        except Exception as exc:
            _stop_service(process)
            logs = _read_logs(service_logs)
            raise RuntimeError(f"{exc}\nReranker service logs:\n{logs}") from exc
        finally:
            _stop_service(process)

        logs = _read_logs(service_logs)

    try:
        model_loading_seconds = _log_duration(
            MODEL_LOAD_LOG, logs, "model loading time"
        )
        inference_latency_ms = _log_duration(
            INFERENCE_LOG, logs, "inference latency"
        )
    except RuntimeError as exc:
        raise RuntimeError(f"{exc}\nReranker service logs:\n{logs}") from exc
    results = response.get("results")
    if not isinstance(results, list) or len(results) != 5:
        raise RuntimeError("Reranker did not return exactly five results.")

    result_ids = []
    scores = []
    top_five = []
    for expected_rank, result in enumerate(results, start=1):
        document_id = result.get("id")
        score = result.get("score")
        rank = result.get("rank")
        if document_id not in candidates_by_id:
            raise RuntimeError(f"Reranker returned an unknown document ID: {document_id!r}")
        if rank != expected_rank:
            raise RuntimeError(f"Reranker returned invalid rank {rank!r}.")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise RuntimeError(f"Reranker returned an invalid score for {document_id!r}.")

        original_document = candidates_by_id[document_id]
        original_text = submitted_text_by_id[document_id]
        if original_text != original_document.page_content:
            raise RuntimeError(f"Submitted text changed for document {document_id!r}.")
        result_ids.append(document_id)
        scores.append(float(score))
        top_five.append(
            {
                "id": document_id,
                "score": float(score),
                "rank": rank,
                "title": original_document.metadata.get("title"),
                "text": original_text,
            }
        )

    if len(set(result_ids)) != 5:
        raise RuntimeError("Reranker returned duplicate document IDs.")
    if scores != sorted(scores, reverse=True):
        raise RuntimeError("Reranker results are not sorted by descending score.")
    weights_cached_after = _cached_model_weights(cache_dir)
    if not weights_cached_after:
        raise RuntimeError(f"Model weights were not found in cache: {cache_dir}")

    return {
        "status": "passed",
        "platform": platform.platform(),
        "device_requested": args.device,
        "model": MODEL_NAME,
        "model_weights_downloaded_this_run": not weights_cached_before,
        "model_cache_dir": str(cache_dir),
        "question_id": question_id,
        "query": query,
        "hybrid_candidate_count": len(candidates),
        "model_loading_seconds": model_loading_seconds,
        "service_startup_to_ready_seconds": round(startup_to_ready_seconds, 3),
        "inference_latency_ms": inference_latency_ms,
        "http_round_trip_ms": round(http_round_trip_ms, 2),
        "peak_service_process_rss_mib": _peak_child_rss_mib(),
        "top_5": top_five,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-index", type=int, default=0)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu", "auto"), default="mps")
    parser.add_argument(
        "--service-python",
        default=str(SERVICE_DIR / ".venv" / "bin" / "python"),
        help="Python executable from the standalone reranker-service environment.",
    )
    parser.add_argument("--cache-dir", help="Override the model cache directory.")
    parser.add_argument("--port", type=int, help="Port for the temporary local service.")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    args = parser.parse_args()

    report = _run(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
