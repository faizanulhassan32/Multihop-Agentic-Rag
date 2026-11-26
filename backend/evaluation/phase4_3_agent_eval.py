#!/usr/bin/env python3
"""
Phase 4.3: End-to-end evaluation of the existing LangGraph RAG workflow.

Read-only with respect to the application and retrieval indexes:
- Does not modify graph.py or other application files.
- Does not rebuild ChromaDB or the BM25 index.
- Runs questions sequentially through the existing compiled LangGraph.
- Captures the graph's state trace, retrieval calls, retrieved documents,
  answer-generation contexts, timings, and final answers.
- Calculates Exact Match and token F1 when the dataset has an "answer" field.

Run from the backend directory:
    python evaluation/phase4_3_agent_eval.py --limit 1
    python evaluation/phase4_3_agent_eval.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
import statistics
import sys
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch


# ---------------------------------------------------------------------
# Paths and imports
# ---------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BACKEND_DIR / "src"))

import chromadb
import httpx
from chromadb.config import Settings
from langchain_core.prompts import ChatPromptTemplate

graph_module = __import__("agent.graph", fromlist=["graph"])

from config import Config

DEFAULT_QUESTIONS = (
    EVALUATION_DIR / "data" / "hotpotqa_retrieval_eval_20.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    EVALUATION_DIR / "results" / "phase4_3_agent_eval"
)


# ---------------------------------------------------------------------
# Console logging
# ---------------------------------------------------------------------

class Tee:
    """Write console output to both the terminal and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)()
                   for stream in self.streams)


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def json_default(value: Any):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def normalize_answer(value: Any) -> str:
    """HotpotQA-style answer normalization."""
    text = str(value or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(
        character for character in text
        if character.isalnum() or character.isspace()
    )
    return " ".join(text.split())


def exact_match(prediction: Any, expected: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(expected))


def token_f1(prediction: Any, expected: Any) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    expected_tokens = normalize_answer(expected).split()

    if not predicted_tokens or not expected_tokens:
        return float(predicted_tokens == expected_tokens)

    common = Counter(predicted_tokens) & Counter(expected_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)

    return 2 * precision * recall / (precision + recall)


def percentile_95(values: list[float]) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def document_record(document: Any) -> dict:
    return {
        "id": document.metadata.get("id"),
        "title": document.metadata.get("title"),
        "metadata": dict(document.metadata),
        "content": document.page_content,
    }


def load_questions(path: Path, limit: int | None) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    questions = []
    seen_ids = set()

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            question_id = str(record.get("id", "")).strip()
            question = record.get("question")

            if not question_id:
                raise ValueError(
                    f"Dataset line {line_number} has no question ID."
                )

            if question_id in seen_ids:
                raise ValueError(
                    f"Duplicate question ID in dataset: {question_id}"
                )

            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Question {question_id} has invalid question text."
                )

            record["id"] = question_id
            record["question"] = question.strip()
            seen_ids.add(question_id)
            questions.append(record)

            if limit is not None and len(questions) >= limit:
                break

    if not questions:
        raise ValueError("No evaluation questions were loaded.")

    return questions


# ---------------------------------------------------------------------
# Dependency and index checks
# ---------------------------------------------------------------------

async def verify_services() -> dict:
    """Check Ollama and reranker readiness before running expensive queries."""

    checks = {}

    async with httpx.AsyncClient(timeout=10.0) as client:
        ollama_url = Config.OLLAMA_BASE_URL.rstrip("/") + "/api/tags"
        response = await client.get(ollama_url)
        response.raise_for_status()
        checks["ollama"] = {
            "ready": True,
            "url": Config.OLLAMA_BASE_URL,
        }

        reranker_url = (
            Config.RERANKER_SERVICE_URL.rstrip("/") + "/ready"
        )
        response = await client.get(reranker_url)
        response.raise_for_status()
        checks["reranker"] = {
            "ready": True,
            "url": Config.RERANKER_SERVICE_URL,
            "status_code": response.status_code,
        }

    return checks


def verify_indexes() -> dict:
    chroma_path = Path(Config.CHROMA_PERSIST_DIR).expanduser().resolve()
    bm25_path = Path(Config.BM25_DB_PATH).expanduser().resolve()

    if not chroma_path.is_dir():
        raise FileNotFoundError(f"Chroma directory not found: {chroma_path}")

    if not bm25_path.is_file():
        raise FileNotFoundError(f"BM25 database not found: {bm25_path}")

    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_collection(Config.CHROMA_COLLECTION_NAME)

    return {
        "chroma_path": str(chroma_path),
        "chroma_collection": Config.CHROMA_COLLECTION_NAME,
        "chroma_document_count": collection.count(),
        "bm25_path": str(bm25_path),
        "read_only": True,
    }


# ---------------------------------------------------------------------
# Instrumentation for the existing graph
# ---------------------------------------------------------------------

async def run_one_question(question_record: dict) -> dict:
    """
    Invoke the existing graph once.

    Instrumentation is applied externally by temporarily wrapping:
    - rag_system.retrieve_documents
    - ChatPromptTemplate.ainvoke

    No production source files are changed.
    """

    question = question_record["question"]
    request_id = uuid.uuid4().hex[:12]

    retrieval_calls = []
    answer_contexts = []

    original_retrieve = graph_module.rag_system.retrieve_documents
    original_prompt_ainvoke = ChatPromptTemplate.ainvoke

    async def capture_retrieval(query: str, request_id: str = "unknown"):
        started = time.perf_counter()

        documents = await original_retrieve(
            query,
            request_id=request_id,
        )

        elapsed_ms = (time.perf_counter() - started) * 1000

        retrieval_calls.append({
            "query": query,
            "request_id": request_id,
            "elapsed_ms": elapsed_ms,
            "documents": [
                document_record(document)
                for document in documents
            ],
        })

        return documents

    async def capture_prompt_context(
        prompt,
        input,
        *args,
        **kwargs,
    ):
        if isinstance(input, dict):
            context = input.get("context")

            if isinstance(context, str):
                answer_contexts.append({
                    "question": input.get("question"),
                    "context": context,
                    "captured_at": timestamp(),
                })

        return await original_prompt_ainvoke(
            prompt,
            input,
            *args,
            **kwargs,
        )

    graph_config = {
        "configurable": {
            "model": Config.OLLAMA_GENERATION_MODEL,
            "temperature": 0.0,
            "max_tokens": 500,
        },
        "metadata": {
            "request_id": request_id,
            "evaluation": "phase4_3",
            "question_id": question_record["id"],
        },
    }

    started = time.perf_counter()

    with (
        patch.object(
            graph_module.rag_system,
            "retrieve_documents",
            new=capture_retrieval,
        ),
        patch.object(
            ChatPromptTemplate,
            "ainvoke",
            new=capture_prompt_context,
        ),
    ):
        result = await graph_module.graph.ainvoke(
            {"original_question": question},
            config=graph_config,
        )

    total_elapsed_ms = (time.perf_counter() - started) * 1000

    final_answer = result.get("final_answer", "")
    trace = result.get("trace", [])

    retrieved_document_count = sum(
        len(call["documents"])
        for call in retrieval_calls
    )

    return {
        "id": question_record["id"],
        "question": question,
        "type": question_record.get("type", "unknown"),
        "level": question_record.get("level", "unknown"),
        "expected_answer": question_record.get("answer"),
        "predicted_answer": final_answer,
        "classification": result.get("classification"),
        "classification_reason": result.get("classification_reason"),
        "total_elapsed_ms": total_elapsed_ms,
        "decomposed_questions": result.get("decomposed_questions", []),
        "intermediate_answers": result.get("intermediate_answers", []),
        "retrieval_call_count": len(retrieval_calls),
        "retrieved_document_count": retrieved_document_count,
        "retrieval_calls": retrieval_calls,
        "answer_contexts": answer_contexts,
        "trace": trace,
        "worker_answer": result.get("worker_answer"),
    }


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

async def run_evaluation(questions: list[dict]) -> list[dict]:
    results = []

    graph_module.rag_system.startup()

    try:
        for index, question_record in enumerate(questions, start=1):
            print("\n" + "=" * 90)
            print(
                f"QUESTION {index}/{len(questions)} "
                f"| ID: {question_record['id']}"
            )
            print("=" * 90)
            print(f"Question: {question_record['question']}")
            print()

            try:
                result = await run_one_question(question_record)

                expected = result.get("expected_answer")
                predicted = result.get("predicted_answer", "")

                if expected is not None:
                    result["exact_match"] = exact_match(predicted, expected)
                    result["token_f1"] = token_f1(predicted, expected)
                else:
                    result["exact_match"] = None
                    result["token_f1"] = None

                result["status"] = "completed"

                print("\nQUESTION COMPLETED")
                print(f"Classification: {result['classification']}")
                print(f"Sub-questions: {len(result['decomposed_questions'])}")
                print(f"Retrieval calls: {result['retrieval_call_count']}")
                print(
                    f"Retrieved documents: "
                    f"{result['retrieved_document_count']}"
                )
                print(
                    f"Total latency: "
                    f"{result['total_elapsed_ms']:.2f} ms"
                )

                if result["exact_match"] is not None:
                    print(f"Exact Match: {result['exact_match']:.2f}")
                    print(f"Token F1: {result['token_f1']:.4f}")

                print(f"Final answer: {predicted}")

            except Exception as exc:
                result = {
                    "id": question_record["id"],
                    "question": question_record["question"],
                    "type": question_record.get("type", "unknown"),
                    "level": question_record.get("level", "unknown"),
                    "expected_answer": question_record.get("answer"),
                    "predicted_answer": None,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }

                print(
                    f"\nQUESTION FAILED: "
                    f"{type(exc).__name__}: {exc}"
                )

            results.append(result)

    finally:
        await graph_module.rag_system.shutdown()

    return results


def write_artifacts(
    results: list[dict],
    output_dir: Path,
    checks: dict,
    index_info: dict,
    dataset_path: Path,
) -> dict:

    output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = output_dir / "execution_traces.jsonl"
    csv_path = output_dir / "per_question.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "evaluation_report.md"

    with trace_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    default=json_default,
                ) + "\n"
            )

    csv_fields = [
        "id",
        "question",
        "type",
        "level",
        "status",
        "classification",
        "total_elapsed_ms",
        "decomposed_question_count",
        "retrieval_call_count",
        "retrieved_document_count",
        "expected_answer",
        "predicted_answer",
        "exact_match",
        "token_f1",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()

        for result in results:
            row = dict(result)
            row["decomposed_question_count"] = len(
                result.get("decomposed_questions", [])
            )
            writer.writerow({
                field: row.get(field)
                for field in csv_fields
            })

    completed = [
        result for result in results
        if result.get("status") == "completed"
    ]

    latencies = [
        result["total_elapsed_ms"]
        for result in completed
        if isinstance(result.get("total_elapsed_ms"), (int, float))
    ]

    exact_matches = [
        result["exact_match"]
        for result in completed
        if result.get("exact_match") is not None
    ]

    f1_scores = [
        result["token_f1"]
        for result in completed
        if result.get("token_f1") is not None
    ]

    classifications = Counter(
        result.get("classification", "unknown")
        for result in completed
    )

    summary = {
        "phase": "4.3",
        "created_at": timestamp(),
        "dataset": str(dataset_path),
        "questions_requested": len(results),
        "questions_completed": len(completed),
        "questions_failed": len(results) - len(completed),
        "latency_ms": {
            "mean": mean_or_none(latencies),
            "median": statistics.median(latencies) if latencies else None,
            "p95": percentile_95(latencies),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "answer_quality": {
            "exact_match_mean": mean_or_none(exact_matches),
            "token_f1_mean": mean_or_none(f1_scores),
            "scored_question_count": len(exact_matches),
            "note": (
                "Answer metrics are omitted for records without an "
                "'answer' field in the evaluation dataset."
            ),
        },
        "workflow": {
            "classification_counts": dict(classifications),
            "mean_decomposed_questions": mean_or_none([
                len(result.get("decomposed_questions", []))
                for result in completed
            ]),
            "mean_retrieval_calls": mean_or_none([
                result.get("retrieval_call_count", 0)
                for result in completed
            ]),
            "mean_retrieved_documents": mean_or_none([
                result.get("retrieved_document_count", 0)
                for result in completed
            ]),
        },
        "services": checks,
        "indexes": index_info,
        "artifacts": {
            "summary": str(summary_path),
            "per_question": str(csv_path),
            "execution_traces": str(trace_path),
            "console_log": str(output_dir / "console.log"),
            "report": str(report_path),
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# Phase 4.3: End-to-End Agent Evaluation",
        "",
        f"- Run date: {summary['created_at']}",
        f"- Dataset: `{dataset_path}`",
        f"- Questions requested: {len(results)}",
        f"- Completed: {len(completed)}",
        f"- Failed: {len(results) - len(completed)}",
        "",
        "## Latency",
        "",
        "| Metric | Milliseconds |",
        "|---|---:|",
        f"| Mean | {summary['latency_ms']['mean']} |",
        f"| Median | {summary['latency_ms']['median']} |",
        f"| P95 | {summary['latency_ms']['p95']} |",
        f"| Min | {summary['latency_ms']['min']} |",
        f"| Max | {summary['latency_ms']['max']} |",
        "",
        "## Answer quality",
        "",
        f"- Mean Exact Match: {summary['answer_quality']['exact_match_mean']}",
        f"- Mean Token F1: {summary['answer_quality']['token_f1_mean']}",
        f"- Scored questions: {summary['answer_quality']['scored_question_count']}",
        "",
        "## Workflow",
        "",
        f"- Classifications: `{dict(classifications)}`",
        f"- Mean decomposed questions: {summary['workflow']['mean_decomposed_questions']}",
        f"- Mean retrieval calls: {summary['workflow']['mean_retrieval_calls']}",
        f"- Mean retrieved documents: {summary['workflow']['mean_retrieved_documents']}",
        "",
        "## Artifacts",
        "",
        f"- `{trace_path.name}`: full per-question execution traces",
        f"- `{csv_path.name}`: per-question metrics",
        f"- `{summary_path.name}`: machine-readable summary",
        f"- `console.log`: terminal output",
    ]

    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

async def async_main(args) -> int:
    dataset_path = args.questions.expanduser().resolve()

    questions = load_questions(dataset_path, args.limit)

    run_name = "smoke_test" if args.limit == 1 else "full_run"
    output_dir = args.output_dir.expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "console.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)

        try:
            print("=" * 90)
            print("PHASE 4.3: END-TO-END AGENT EVALUATION")
            print("=" * 90)
            print(f"Questions: {len(questions)}")
            print(f"Dataset: {dataset_path}")
            print(f"Output: {output_dir}")
            print(f"Generation model: {Config.OLLAMA_GENERATION_MODEL}")
            print(f"Classifier model: {Config.OLLAMA_CLASSIFIER_MODEL}")
            print(f"Embedding model: {Config.OLLAMA_EMBEDDING_MODEL}")
            print()

            index_info = verify_indexes()
            print("INDEX CHECK")
            print(json.dumps(index_info, indent=2))
            print()

            checks = await verify_services()
            print("SERVICE CHECK")
            print(json.dumps(checks, indent=2))
            print()

            print("Starting graph evaluation...")
            results = await run_evaluation(questions)

            summary = write_artifacts(
                results,
                output_dir,
                checks,
                index_info,
                dataset_path,
            )

            print("\n" + "=" * 90)
            print("EVALUATION SUMMARY")
            print("=" * 90)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            print("\nArtifacts saved:")
            print(f"  {output_dir}")
            print("\nEvaluation finished.")

            return 1 if summary["questions_failed"] else 0

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions. Use --limit 1 for the smoke test.",
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help="Path to the evaluation JSONL dataset.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory for evaluation artifacts.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")

    try:
        exit_code = asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")
        exit_code = 130
    except Exception as exc:
        print(f"\nEVALUATION ABORTED: {type(exc).__name__}: {exc}")
        exit_code = 1

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()