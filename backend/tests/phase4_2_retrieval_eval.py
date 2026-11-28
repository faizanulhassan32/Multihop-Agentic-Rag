#!/usr/bin/env python3
"""
Phase 4.2: Evaluate the existing Dense + BM25 + RRF + Reranker pipeline.

Read-only evaluation:
- Does not rebuild or modify ChromaDB.
- Does not rebuild or modify the BM25 index.
- Uses the existing retrieval functions from hybrid_retriever.py.
- Uses the existing RerankerClient.
- Does not execute the LangGraph question-answering workflow.

Expected test JSONL:
backend/tests/data/hotpotqa_retrieval_eval_20.jsonl

Each question must contain:
{
    "id": "...",
    "question": "...",
    "type": "bridge",
    "level": "medium",
    "supporting_evidence": [
        {
            "title": "Article title",
            "sentence_index": 0,
            "text": "Exact supporting sentence from the article."
        }
    ]
}

Supporting evidence is mapped to the actual indexed chunks by matching
the article title and exact sentence text.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# ---------------------------------------------------------------------
# Paths and imports
# ---------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BACKEND_DIR / "src"))

import chromadb
import httpx
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings

from agent.retrieval import (
    _retrieve_dense,
    _retrieve_bm25,
    reciprocal_rank_fusion,
)
from config import Config
from ingestion.preprocess import CHUNKS_PATH
from services.reranker_client import RerankerClient


DEFAULT_QUESTIONS = (
    TESTS_DIR / "data" / "hotpotqa_retrieval_eval_20.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    TESTS_DIR / "results" / "phase4_2_retrieval_eval"
)

MAX_RERANKER_DOCUMENTS = 20


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def normalize_text(value: str) -> str:
    """Normalize whitespace and case for evidence matching."""
    return " ".join(value.split()).casefold()


def percentile_95(values: list[float]) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    index = math.ceil(0.95 * len(ordered)) - 1
    return ordered[max(0, index)]


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def document_id(document: Any) -> str:
    value = document.metadata.get("id")

    if not isinstance(value, str) or not value:
        raise ValueError(
            "Retrieved document is missing its stable metadata ID."
        )

    return value


def document_title(document: Any) -> str:
    return str(document.metadata.get("title", "Unknown title"))


# ---------------------------------------------------------------------
# Load and validate evaluation questions
# ---------------------------------------------------------------------

def load_chunks() -> tuple[list[dict], dict[str, list[dict]]]:
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(
            f"Processed chunks file was not found: {CHUNKS_PATH}"
        )

    chunks = []
    chunks_by_title: dict[str, list[dict]] = defaultdict(list)
    seen_ids = set()

    with CHUNKS_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            chunk = json.loads(line)

            chunk_id = chunk.get("id")
            content = chunk.get("content")
            title = chunk.get("title")

            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(
                    f"Chunk at line {line_number} has no valid ID."
                )

            if chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk ID in processed file: {chunk_id}"
                )

            if not isinstance(content, str):
                raise ValueError(
                    f"Chunk {chunk_id} has invalid content."
                )

            if not isinstance(title, str) or not title:
                raise ValueError(
                    f"Chunk {chunk_id} has no valid title."
                )

            seen_ids.add(chunk_id)
            chunks.append(chunk)
            chunks_by_title[normalize_text(title)].append(chunk)

    return chunks, chunks_by_title


def map_supporting_evidence(
    evidence: list[dict],
    chunks_by_title: dict[str, list[dict]],
    question_id: str,
) -> tuple[set[str], list[dict]]:
    """
    Map each supporting sentence to the indexed chunk IDs containing it.

    A sentence may map to multiple chunks if chunk overlap causes it
    to appear in more than one chunk.
    """

    relevant_ids: set[str] = set()
    evidence_mappings = []

    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise ValueError(
                f"Question {question_id}: supporting_evidence[{index}] "
                "must be an object."
            )

        title = item.get("title")
        sentence = item.get("text") or item.get("sentence")

        if not isinstance(title, str) or not title.strip():
            raise ValueError(
                f"Question {question_id}: evidence item {index} "
                "has no valid title."
            )

        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError(
                f"Question {question_id}: evidence item {index} "
                "has no valid sentence text."
            )

        normalized_title = normalize_text(title)
        normalized_sentence = normalize_text(sentence)

        candidate_chunks = chunks_by_title.get(normalized_title, [])

        matching_ids = [
            chunk["id"]
            for chunk in candidate_chunks
            if normalized_sentence
            in normalize_text(chunk["content"])
        ]

        if not matching_ids:
            raise ValueError(
                f"\nCould not map supporting evidence to an indexed chunk."
                f"\nQuestion ID: {question_id}"
                f"\nArticle title: {title}"
                f"\nSentence: {sentence[:300]}"
                "\nCheck whether the evaluation sentence exactly matches "
                "the processed article text and chunk export."
            )

        relevant_ids.update(matching_ids)

        evidence_mappings.append(
            {
                "title": title,
                "sentence_index": item.get("sentence_index"),
                "text": sentence,
                "relevant_chunk_ids": matching_ids,
            }
        )

    if not relevant_ids:
        raise ValueError(
            f"Question {question_id} has no mapped relevant chunks."
        )

    return relevant_ids, evidence_mappings


def load_questions(
    path: Path,
    chunks_by_title: dict[str, list[dict]],
    limit: int | None,
) -> list[dict]:

    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation JSONL was not found: {path}\n"
            "Place hotpotqa_retrieval_eval_20.jsonl in "
            "backend/evaluation/data/ or pass --questions."
        )

    questions = []
    seen_question_ids = set()

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            question_id = str(record.get("id", "")).strip()
            question_text = record.get("question")

            if not question_id:
                raise ValueError(
                    f"Question at line {line_number} has no ID."
                )

            if question_id in seen_question_ids:
                raise ValueError(
                    f"Duplicate evaluation question ID: {question_id}"
                )

            if not isinstance(question_text, str) or not question_text.strip():
                raise ValueError(
                    f"Question {question_id} has no valid question text."
                )

            evidence = record.get("supporting_evidence")

            if not isinstance(evidence, list) or not evidence:
                raise ValueError(
                    f"Question {question_id} has no supporting_evidence list."
                )

            relevant_ids, evidence_mappings = map_supporting_evidence(
                evidence,
                chunks_by_title,
                question_id,
            )

            seen_question_ids.add(question_id)

            questions.append(
                {
                    "id": question_id,
                    "question": question_text.strip(),
                    "type": record.get("type", "unknown"),
                    "level": record.get("level", "unknown"),
                    "expected_source_titles": record.get(
                        "expected_source_titles", []
                    ),
                    "relevant_chunk_ids": sorted(relevant_ids),
                    "supporting_evidence": evidence_mappings,
                }
            )

            if limit is not None and len(questions) >= limit:
                break

    if not questions:
        raise ValueError("The evaluation JSONL contains no questions.")

    return questions


# ---------------------------------------------------------------------
# Index integrity checks
# ---------------------------------------------------------------------

def read_bm25_ids_read_only(db_path: Path) -> set[str]:
    """Read BM25 IDs without opening the database in write mode."""

    if not db_path.is_file():
        raise FileNotFoundError(
            f"BM25 database does not exist: {db_path}"
        )

    uri = db_path.resolve().as_uri() + "?mode=ro"

    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT id FROM bm25_documents"
        ).fetchall()

    return {row[0] for row in rows}


def verify_indexes(
    collection,
    chunks: list[dict],
    bm25_path: Path,
) -> dict:

    chroma_ids = set(collection.get(include=[])["ids"])
    bm25_ids = read_bm25_ids_read_only(bm25_path)
    chunk_ids = {chunk["id"] for chunk in chunks}

    chroma_only = chroma_ids - bm25_ids
    bm25_only = bm25_ids - chroma_ids

    chunk_only = chunk_ids - chroma_ids
    index_only = chroma_ids - chunk_ids

    result = {
        "chroma_count": len(chroma_ids),
        "bm25_count": len(bm25_ids),
        "processed_chunk_count": len(chunk_ids),
        "chroma_bm25_ids_match": chroma_ids == bm25_ids,
        "chroma_chunks_ids_match": chroma_ids == chunk_ids,
        "read_only": True,
        "mismatch_details": {
            "chroma_only_count": len(chroma_only),
            "bm25_only_count": len(bm25_only),
            "processed_chunks_only_count": len(chunk_only),
            "indexes_only_count": len(index_only),
            "chroma_only_examples": sorted(chroma_only)[:5],
            "bm25_only_examples": sorted(bm25_only)[:5],
            "processed_chunks_only_examples": sorted(chunk_only)[:5],
            "indexes_only_examples": sorted(index_only)[:5],
        },
    }

    if not result["chroma_bm25_ids_match"]:
        raise RuntimeError(
            "Chroma and BM25 contain different document IDs. "
            "Evaluation stopped to avoid misleading results.\n"
            + json.dumps(result["mismatch_details"], indent=2)
        )

    if not result["chroma_chunks_ids_match"]:
        raise RuntimeError(
            "The processed chunks file and production indexes contain "
            "different IDs. Evaluation stopped.\n"
            + json.dumps(result["mismatch_details"], indent=2)
        )

    return result


# ---------------------------------------------------------------------
# Retrieval and reranking
# ---------------------------------------------------------------------

async def timed_call(function, *args, **kwargs):
    started = time.perf_counter()

    result = await asyncio.to_thread(
        function,
        *args,
        **kwargs,
    )

    elapsed_ms = (time.perf_counter() - started) * 1000

    return result, elapsed_ms


def calculate_rrf_scores(
    dense_documents: list,
    bm25_documents: list,
    rrf_constant: int,
) -> dict[str, float]:

    scores: dict[str, float] = {}

    for results in (dense_documents, bm25_documents):
        seen = set()

        for rank, document in enumerate(results, start=1):
            doc_id = document_id(document)

            if doc_id in seen:
                continue

            seen.add(doc_id)

            scores[doc_id] = scores.get(doc_id, 0.0) + (
                1.0 / (rrf_constant + rank)
            )

    return scores


async def check_reranker_ready(service_url: str) -> None:
    url = service_url.rstrip("/") + "/ready"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()

    except Exception as exc:
        raise RuntimeError(
            f"Reranker service is not ready at {url}.\n"
            "Start the standalone reranker service before running "
            "the evaluation.\n"
            f"Details: {type(exc).__name__}: {exc}"
        ) from exc



async def retrieve_and_rerank(
    question: dict,
    vectorstore,
    reranker_client: RerankerClient,
    args,
    bm25_path: Path,
) -> dict:

    query = question["question"]
    request_id = f"eval-{question['id']}"

    total_started = time.perf_counter()

    # Run dense and BM25 concurrently.
    retrieval_started = time.perf_counter()

    dense_task = timed_call(
        _retrieve_dense,
        vectorstore,
        query,
        args.dense_k,
        request_id,
    )

    bm25_task = timed_call(
        _retrieve_bm25,
        query,
        args.bm25_k,
        bm25_path,
        request_id,
    )

    (dense_documents, dense_ms), (bm25_documents, bm25_ms) = (
        await asyncio.gather(dense_task, bm25_task)
    )

    parallel_retrieval_ms = (
        time.perf_counter() - retrieval_started
    ) * 1000

    # Reciprocal Rank Fusion.
    rrf_started = time.perf_counter()

    fused_documents = reciprocal_rank_fusion(
        dense_documents,
        bm25_documents,
        result_limit=args.rrf_candidate_k,
        rrf_constant=args.rrf_constant,
    )

    rrf_ms = (time.perf_counter() - rrf_started) * 1000

    rrf_scores = calculate_rrf_scores(
        dense_documents,
        bm25_documents,
        args.rrf_constant,
    )

    # All retrieval results are LangChain Document objects.
    documents_by_id = {
        document_id(document): document
        for document in fused_documents
    }

    # Rerank the fused candidate pool.
    reranker_started = time.perf_counter()

    reranked_results = []

    if fused_documents:
        rerank_count = min(
            args.rerank_eval_k,
            len(fused_documents),
        )

        reranked_results = await reranker_client.rerank(
            query=query,
            documents=[
                {
                    "id": doc_id,
                    "text": document.page_content,
                }
                for doc_id, document in documents_by_id.items()
            ],
            top_n=rerank_count,
            request_id=request_id,
        )

    reranker_ms = (
        time.perf_counter() - reranker_started
    ) * 1000

    total_ms = (time.perf_counter() - total_started) * 1000

    reranked_ids = [
        result.id for result in reranked_results
    ]

    reranker_scores = {
        result.id: result.score
        for result in reranked_results
    }

    return {
        "dense_documents": dense_documents,
        "bm25_documents": bm25_documents,
        "fused_documents": fused_documents,
        "reranked_results": reranked_results,
        "reranked_documents": [
            documents_by_id[doc_id]
            for doc_id in reranked_ids
        ],
        "rankings": {
            "dense": [
                document_id(document)
                for document in dense_documents
            ],
            "bm25": [
                document_id(document)
                for document in bm25_documents
            ],
            "rrf": [
                document_id(document)
                for document in fused_documents
            ],
            "reranked": reranked_ids,
        },
        "scores": {
            "rrf": rrf_scores,
            "reranker": reranker_scores,
        },
        "latencies_ms": {
            "dense": dense_ms,
            "bm25": bm25_ms,
            "parallel_retrieval_wall": parallel_retrieval_ms,
            "rrf": rrf_ms,
            "reranker": reranker_ms,
            "total": total_ms,
        },
    }


# ---------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------

def recall_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:

    if not relevant_ids:
        return 0.0

    retrieved = set(ranked_ids[:k])

    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:

    if k <= 0:
        return 0.0

    hits = len(set(ranked_ids[:k]) & relevant_ids)

    return hits / k


def reciprocal_rank_at_10(
    ranked_ids: list[str],
    relevant_ids: set[str],
) -> float:

    for rank, doc_id in enumerate(ranked_ids[:10], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_5(
    ranked_ids: list[str],
    relevant_ids: set[str],
) -> float:

    dcg = 0.0

    for rank, doc_id in enumerate(ranked_ids[:5], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_count = min(5, len(relevant_ids))

    if ideal_count == 0:
        return 0.0

    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    return dcg / ideal_dcg


def evidence_coverage_at_k(
    ranked_ids: list[str],
    evidence_mappings: list[dict],
    k: int,
) -> float:

    if not evidence_mappings:
        return 0.0

    retrieved = set(ranked_ids[:k])
    covered = 0

    for evidence in evidence_mappings:
        evidence_chunk_ids = set(
            evidence["relevant_chunk_ids"]
        )

        if retrieved & evidence_chunk_ids:
            covered += 1

    return covered / len(evidence_mappings)


def calculate_metrics(
    ranked_ids: list[str],
    relevant_ids: set[str],
    evidence_mappings: list[dict],
    maximum_available_k: int,
) -> dict:

    metrics = {}

    for k in (5, 10, 20):
        if k <= maximum_available_k:
            metrics[f"recall@{k}"] = recall_at_k(
                ranked_ids,
                relevant_ids,
                k,
            )

            metrics[f"evidence_coverage@{k}"] = (
                evidence_coverage_at_k(
                    ranked_ids,
                    evidence_mappings,
                    k,
                )
            )
        else:
            # Do not report Recall@20 for a top-10 retrieval result.
            metrics[f"recall@{k}"] = None
            metrics[f"evidence_coverage@{k}"] = None

    metrics["precision@5"] = (
        precision_at_k(ranked_ids, relevant_ids, 5)
        if maximum_available_k >= 5
        else None
    )

    metrics["mrr@10"] = (
        reciprocal_rank_at_10(ranked_ids, relevant_ids)
        if maximum_available_k >= 10
        else None
    )

    metrics["ndcg@5"] = (
        ndcg_at_5(ranked_ids, relevant_ids)
        if maximum_available_k >= 5
        else None
    )

    return metrics


def ranking_details(
    ranked_ids: list[str],
    documents_by_id: dict[str, Any],
    relevant_ids: set[str],
    scores: dict[str, float] | None = None,
) -> list[dict]:

    details = []

    for rank, doc_id in enumerate(ranked_ids, start=1):
        document = documents_by_id.get(doc_id)

        details.append(
            {
                "rank": rank,
                "id": doc_id,
                "title": (
                    document_title(document)
                    if document is not None
                    else "Unknown title"
                ),
                "relevant": doc_id in relevant_ids,
                "score": (
                    scores.get(doc_id)
                    if scores is not None
                    else None
                ),
            }
        )

    return details


# ---------------------------------------------------------------------
# Aggregate results and comparisons
# ---------------------------------------------------------------------

METRIC_NAMES = (
    "recall@5",
    "recall@10",
    "recall@20",
    "precision@5",
    "mrr@10",
    "ndcg@5",
    "evidence_coverage@5",
    "evidence_coverage@10",
    "evidence_coverage@20",
)


def aggregate_stage(rows: list[dict], stage: str) -> dict:

    summary = {}

    for metric in METRIC_NAMES:
        values = [
            row["metrics"][stage].get(metric)
            for row in rows
            if row["metrics"][stage].get(metric) is not None
        ]

        summary[metric] = mean_or_none(values)

    latency_values = [
        row["latencies_ms"]["total"]
        if stage == "reranked"
        else row["latencies_ms"].get(stage)
        for row in rows
    ]

    summary["average_latency_ms"] = mean_or_none(
        latency_values
    )

    summary["p95_latency_ms"] = percentile_95(
        latency_values
    )

    return summary


def paired_comparison(
    rows: list[dict],
    first_stage: str,
    second_stage: str,
    metric: str,
) -> dict:

    differences = []

    for row in rows:
        first = row["metrics"][first_stage].get(metric)
        second = row["metrics"][second_stage].get(metric)

        if first is None or second is None:
            continue

        differences.append(second - first)

    if not differences:
        return {
            "mean_delta": None,
            "improved": 0,
            "regressed": 0,
            "tied": 0,
        }

    return {
        "mean_delta": statistics.mean(differences),
        "improved": sum(value > 0 for value in differences),
        "regressed": sum(value < 0 for value in differences),
        "tied": sum(value == 0 for value in differences),
        "question_count": len(differences),
    }


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

async def run_evaluation(
    questions: list[dict],
    vectorstore,
    args,
    bm25_path: Path,
) -> list[dict]:

    await check_reranker_ready(Config.RERANKER_SERVICE_URL)

    rows = []

    async with RerankerClient() as reranker_client:

        print("\nRunning one warm-up query. It will not be included in metrics.")

        await retrieve_and_rerank(
            questions[0],
            vectorstore,
            reranker_client,
            args,
            bm25_path,
        )

        for index, question in enumerate(questions, start=1):

            print(
                f"\n[{index}/{len(questions)}] "
                f"{question['id']} | "
                f"{question['type']} | "
                f"{question['level']}"
            )

            print(f"Question: {question['question']}")

            result = await retrieve_and_rerank(
                question,
                vectorstore,
                reranker_client,
                args,
                bm25_path,
            )

            relevant_ids = set(
                question["relevant_chunk_ids"]
            )

            dense_ids = result["rankings"]["dense"]
            bm25_ids = result["rankings"]["bm25"]
            rrf_ids = result["rankings"]["rrf"]
            reranked_ids = result["rankings"]["reranked"]

            dense_documents_by_id = {
                document_id(doc): doc
                for doc in result["dense_documents"]
            }

            bm25_documents_by_id = {
                document_id(document): document
                for document in result["bm25_documents"]
            }

            fused_documents_by_id = {
                document_id(doc): doc
                for doc in result["fused_documents"]
            }

            reranked_documents_by_id = {
                document_id(doc): doc
                for doc in result["reranked_documents"]
            }

            metrics = {
                "dense": calculate_metrics(
                    dense_ids,
                    relevant_ids,
                    question["supporting_evidence"],
                    args.dense_k,
                ),
                "bm25": calculate_metrics(
                    bm25_ids,
                    relevant_ids,
                    question["supporting_evidence"],
                    args.bm25_k,
                ),
                "rrf": calculate_metrics(
                    rrf_ids,
                    relevant_ids,
                    question["supporting_evidence"],
                    args.rrf_candidate_k,
                ),
                "reranked": calculate_metrics(
                    reranked_ids,
                    relevant_ids,
                    question["supporting_evidence"],
                    args.rerank_eval_k,
                ),
            }

            row = {
                "id": question["id"],
                "question": question["question"],
                "type": question["type"],
                "level": question["level"],
                "expected_source_titles": question[
                    "expected_source_titles"
                ],
                "relevant_chunk_ids": sorted(relevant_ids),
                "supporting_evidence": question[
                    "supporting_evidence"
                ],
                "rankings": result["rankings"],
                "metrics": metrics,
                "latencies_ms": result["latencies_ms"],
                "ranking_details": {
                    "dense": ranking_details(
                        dense_ids,
                        dense_documents_by_id,
                        relevant_ids,
                    ),
                    "bm25": ranking_details(
                        bm25_ids,
                        bm25_documents_by_id,
                        relevant_ids,
                    ),
                    "rrf": ranking_details(
                        rrf_ids,
                        fused_documents_by_id,
                        relevant_ids,
                        result["scores"]["rrf"],
                    ),
                    "reranked": ranking_details(
                        reranked_ids,
                        reranked_documents_by_id,
                        relevant_ids,
                        result["scores"]["reranker"],
                    ),
                },
            }

            rows.append(row)

            print(
                "Candidates: "
                f"dense={len(dense_ids)}, "
                f"BM25={len(bm25_ids)}, "
                f"RRF={len(rrf_ids)}, "
                f"reranked={len(reranked_ids)}"
            )

            print(
                "Latency: "
                f"dense={result['latencies_ms']['dense']:.1f} ms, "
                f"BM25={result['latencies_ms']['bm25']:.1f} ms, "
                f"parallel retrieval="
                f"{result['latencies_ms']['parallel_retrieval_wall']:.1f} ms, "
                f"reranker={result['latencies_ms']['reranker']:.1f} ms, "
                f"total={result['latencies_ms']['total']:.1f} ms"
            )

            print(
                "Recall@5: "
                f"dense={metrics['dense']['recall@5']:.3f}, "
                f"BM25={metrics['bm25']['recall@5']:.3f}, "
                f"RRF={metrics['rrf']['recall@5']:.3f}, "
                f"reranked={metrics['reranked']['recall@5']:.3f}"
            )

            print("Reranked top 5:")

            for item in row["ranking_details"]["reranked"][:5]:
                marker = "✓" if item["relevant"] else " "
                print(
                    f"  {marker} {item['rank']}. "
                    f"{item['title']} [{item['id']}]"
                )

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:

    fields = [
        "id",
        "question",
        "type",
        "level",
    ]

    for stage in ("dense", "bm25", "rrf", "reranked"):
        for metric in METRIC_NAMES:
            fields.append(f"{stage}_{metric}")

    for latency in (
        "dense",
        "bm25",
        "parallel_retrieval_wall",
        "rrf",
        "reranker",
        "total",
    ):
        fields.append(f"latency_{latency}_ms")

    fields.extend(
        [
            "dense_top5_ids",
            "bm25_top5_ids",
            "rrf_top5_ids",
            "reranked_top5_ids",
        ]
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            output = {
                "id": row["id"],
                "question": row["question"],
                "type": row["type"],
                "level": row["level"],
            }

            for stage in ("dense", "bm25", "rrf", "reranked"):
                for metric in METRIC_NAMES:
                    output[f"{stage}_{metric}"] = (
                        row["metrics"][stage].get(metric)
                    )

            for latency, value in row["latencies_ms"].items():
                output[f"latency_{latency}_ms"] = value

            for stage in ("dense", "bm25", "rrf", "reranked"):
                output[f"{stage}_top5_ids"] = " | ".join(
                    row["rankings"][stage][:5]
                )

            writer.writerow(output)


def print_summary(report: dict) -> None:

    print("\n" + "=" * 78)
    print("PHASE 4.2 RETRIEVAL EVALUATION SUMMARY")
    print("=" * 78)

    print(f"Questions evaluated: {report['dataset']['count']}")

    print("\nRetrieval metrics")
    print("-" * 78)

    columns = [
        "Stage",
        "Recall@5",
        "Recall@10",
        "Recall@20",
        "MRR@10",
        "nDCG@5",
        "Evidence@5",
    ]

    print(
        f"{columns[0]:<12}"
        f"{columns[1]:>11}"
        f"{columns[2]:>12}"
        f"{columns[3]:>12}"
        f"{columns[4]:>11}"
        f"{columns[5]:>11}"
        f"{columns[6]:>13}"
    )

    for stage, values in report["metrics"].items():

        def fmt(value):
            return f"{value:.4f}" if value is not None else "N/A"

        print(
            f"{stage:<12}"
            f"{fmt(values['recall@5']):>11}"
            f"{fmt(values['recall@10']):>12}"
            f"{fmt(values['recall@20']):>12}"
            f"{fmt(values['mrr@10']):>11}"
            f"{fmt(values['ndcg@5']):>11}"
            f"{fmt(values['evidence_coverage@5']):>13}"
        )

    print("\nLatency")
    print("-" * 78)

    for stage, values in report["metrics"].items():
        print(
            f"{stage:<12} "
            f"mean={values['average_latency_ms']:.2f} ms, "
            f"P95={values['p95_latency_ms']:.2f} ms"
        )

    print("\nPaired comparisons")
    print("-" * 78)

    for comparison, metrics in report["comparisons"].items():
        print(f"\n{comparison}")

        for metric, result in metrics.items():
            print(
                f"  {metric}: "
                f"mean delta={result['mean_delta']}, "
                f"improved={result['improved']}, "
                f"regressed={result['regressed']}, "
                f"tied={result['tied']}"
            )

    print("\n" + "=" * 78)


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help="Path to the evaluation JSONL file.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of questions to evaluate.",
    )

    parser.add_argument(
        "--dense-k",
        type=int,
        default=Config.DENSE_RETRIEVAL_K,
    )

    parser.add_argument(
        "--bm25-k",
        type=int,
        default=Config.BM25_RETRIEVAL_K,
    )

    parser.add_argument(
        "--rrf-candidate-k",
        type=int,
        default=20,
        help="Production RRF candidate pool size.",
    )

    parser.add_argument(
        "--rrf-constant",
        type=int,
        default=Config.RRF_K,
    )

    parser.add_argument(
        "--rerank-eval-k",
        type=int,
        default=20,
        help=(
            "Number of reranked candidates to retain for evaluation. "
            "The first 5 results represent "
            "the production output."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")

    if min(
        args.dense_k,
        args.bm25_k,
        args.rrf_candidate_k,
        args.rrf_constant,
        args.rerank_eval_k,
    ) <= 0:
        parser.error("Retrieval limits and RRF constant must be positive.")

    if args.rrf_candidate_k > MAX_RERANKER_DOCUMENTS:
        parser.error(
            f"--rrf-candidate-k cannot exceed "
            f"{MAX_RERANKER_DOCUMENTS}."
        )

    if args.rerank_eval_k > args.rrf_candidate_k:
        parser.error(
            "--rerank-eval-k cannot exceed --rrf-candidate-k."
        )

    if args.rerank_eval_k > MAX_RERANKER_DOCUMENTS:
        parser.error(
            f"--rerank-eval-k cannot exceed "
            f"{MAX_RERANKER_DOCUMENTS}."
        )

    return args


def main():

    args = parse_args()

    chroma_path = Path(
        Config.CHROMA_PERSIST_DIR
    ).expanduser().resolve()

    bm25_path = Path(
        Config.BM25_DB_PATH
    ).expanduser().resolve()

    if not chroma_path.is_dir():
        raise FileNotFoundError(
            f"Chroma persistence directory does not exist: {chroma_path}"
        )

    if not bm25_path.is_file():
        raise FileNotFoundError(
            f"BM25 database does not exist: {bm25_path}"
        )

    print("=" * 78)
    print("PHASE 4.2: DENSE + BM25 + RRF + RERANKER EVALUATION")
    print("=" * 78)

    print(f"Questions: {args.questions}")
    print(f"Chroma: {chroma_path}")
    print(f"Collection: {Config.CHROMA_COLLECTION_NAME}")
    print(f"BM25: {bm25_path}")
    print(f"Embedding model: {Config.OLLAMA_EMBEDDING_MODEL}")
    print(f"Reranker: {Config.RERANKER_SERVICE_URL}")

    print("\nLoading processed chunks...")
    chunks, chunks_by_title = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    print("\nLoading evaluation questions and mapping evidence...")
    questions = load_questions(
        args.questions,
        chunks_by_title,
        args.limit,
    )

    print(f"Loaded {len(questions)} evaluation questions.")
    print(
        "All supporting evidence successfully mapped to indexed chunks."
    )

    print("\nOpening existing Chroma collection...")

    chroma_client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(
            anonymized_telemetry=False
        ),
    )

    collection = chroma_client.get_collection(
        Config.CHROMA_COLLECTION_NAME
    )

    print("\nVerifying Chroma, BM25, and processed chunk IDs...")

    index_report = verify_indexes(
        collection,
        chunks,
        bm25_path,
    )

    print(
        json.dumps(index_report, indent=2)
    )

    vectorstore = SimpleNamespace(
        _collection=collection,
        _embedding_function=OllamaEmbeddings(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_EMBEDDING_MODEL,
        ),
    )

    print("\nStarting evaluation...")

    rows = asyncio.run(
        run_evaluation(
            questions,
            vectorstore,
            args,
            bm25_path,
        )
    )

    stages = ("dense", "bm25", "rrf", "reranked")

    metrics = {
        stage: aggregate_stage(rows, stage)
        for stage in stages
    }

    comparisons = {
        "RRF_vs_Dense": {
            metric: paired_comparison(
                rows,
                "dense",
                "rrf",
                metric,
            )
            for metric in (
                "recall@5",
                "recall@10",
                "mrr@10",
                "ndcg@5",
            )
        },
        "Reranked_vs_RRF": {
            metric: paired_comparison(
                rows,
                "rrf",
                "reranked",
                metric,
            )
            for metric in (
                "recall@5",
                "recall@10",
                "mrr@10",
                "ndcg@5",
            )
        },
    }

    report = {
        "metadata": {
            "evaluation": "Phase 4.2 retrieval evaluation",
            "generated_at": timestamp(),
            "branch_reference": "phase-4",
            "script": str(Path(__file__).resolve()),
            "read_only": True,
        },
        "dataset": {
            "source": str(args.questions.resolve()),
            "count": len(rows),
            "question_types": {
                key: sum(
                    row["type"] == key for row in rows
                )
                for key in sorted(
                    {row["type"] for row in rows}
                )
            },
            "difficulty_levels": {
                key: sum(
                    row["level"] == key for row in rows
                )
                for key in sorted(
                    {row["level"] for row in rows}
                )
            },
            "support_mapping": (
                "Supporting evidence sentences were matched against "
                "processed chunk text using normalized article titles "
                "and sentence text."
            ),
        },
        "indexes": index_report,
        "configuration": {
            "ollama_base_url": Config.OLLAMA_BASE_URL,
            "embedding_model": Config.OLLAMA_EMBEDDING_MODEL,
            "chroma_collection": Config.CHROMA_COLLECTION_NAME,
            "dense_k": args.dense_k,
            "bm25_k": args.bm25_k,
            "rrf_candidate_k": args.rrf_candidate_k,
            "rrf_constant": args.rrf_constant,
            "rerank_evaluation_k": args.rerank_eval_k,
            "production_rerank_k": 5,
            "reranker_url": Config.RERANKER_SERVICE_URL,
            "warmup_queries_excluded": 1,
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "per_question": rows,
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        args.output_dir / "phase4_2_retrieval_eval_report.json"
    )

    csv_path = (
        args.output_dir / "phase4_2_retrieval_eval_per_question.csv"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    write_csv(
        rows,
        csv_path,
    )

    print_summary(report)

    print("\nOutput files:")
    print(f"JSON report: {json_path}")
    print(f"CSV report:  {csv_path}")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
