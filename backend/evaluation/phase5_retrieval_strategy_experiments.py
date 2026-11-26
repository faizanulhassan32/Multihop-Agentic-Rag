
#!/usr/bin/env python3
"""
Controlled retrieval strategy experiment for the Phase 5 RAG system.

Strategies:
A. Dense top 10
B. Dense top 10 -> rerank
C. Dense top 20 -> rerank
D. Dense top 10 + BM25 top 10 -> RRF top 20 -> rerank

Read-only:
- Does not rebuild Chroma.
- Does not rebuild BM25.
- Does not modify either index.
- Does not invoke the LangGraph answering workflow.

Run from the backend directory:
    python evaluation/retrieval_strategy_experiments.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR / "src"))

import chromadb
import httpx
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings

from agent.hybrid_retriever import (
    _retrieve_dense,
    _retrieve_bm25,
    reciprocal_rank_fusion,
)
from config import Config
from ingestion.preprocess import CHUNKS_PATH
from services.reranker_client import RerankerClient


DEFAULT_QUESTIONS = (
    EVALUATION_DIR / "data" / "hotpotqa_retrieval_eval_20.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    EVALUATION_DIR / "results" / "retrieval_strategy_experiments"
)

RERANKER_MAX_DOCUMENTS = 20


def normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def document_id(document: Any) -> str:
    value = document.metadata.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError("Retrieved document has no stable metadata ID.")
    return value


def document_title(document: Any) -> str:
    return str(document.metadata.get("title", "Unknown title"))


def percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(0.95 * len(ordered)) - 1
    return ordered[max(0, index)]


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def load_chunks() -> tuple[list[dict], dict[str, list[dict]]]:
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(
            f"Processed chunks file not found: {CHUNKS_PATH}"
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
                    f"Invalid chunk ID at line {line_number}."
                )

            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk ID: {chunk_id}")

            if not isinstance(content, str):
                raise ValueError(f"Invalid content for chunk {chunk_id}.")

            if not isinstance(title, str) or not title:
                raise ValueError(f"Invalid title for chunk {chunk_id}.")

            seen_ids.add(chunk_id)
            chunks.append(chunk)
            chunks_by_title[normalize_text(title)].append(chunk)

    return chunks, chunks_by_title


def map_supporting_evidence(
    evidence: list[dict],
    chunks_by_title: dict[str, list[dict]],
    question_id: str,
) -> tuple[set[str], list[dict]]:
    relevant_ids: set[str] = set()
    mappings = []

    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise ValueError(
                f"Question {question_id}: evidence item {index} is invalid."
            )

        title = item.get("title")
        sentence = item.get("text") or item.get("sentence")

        if not isinstance(title, str) or not title.strip():
            raise ValueError(
                f"Question {question_id}: evidence item {index} has no title."
            )

        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError(
                f"Question {question_id}: evidence item {index} has no text."
            )

        normalized_title = normalize_text(title)
        normalized_sentence = normalize_text(sentence)

        matching_chunks = [
            chunk
            for chunk in chunks_by_title.get(normalized_title, [])
            if normalized_sentence in normalize_text(chunk["content"])
        ]

        if not matching_chunks:
            raise ValueError(
                "Could not map supporting evidence to an indexed chunk.\n"
                f"Question: {question_id}\n"
                f"Title: {title}\n"
                f"Sentence: {sentence[:250]}"
            )

        matching_ids = [chunk["id"] for chunk in matching_chunks]
        relevant_ids.update(matching_ids)

        mappings.append(
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

    return relevant_ids, mappings


def load_questions(
    path: Path,
    chunks_by_title: dict[str, list[dict]],
    limit: int | None,
) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation file not found: {path}"
        )

    questions = []
    seen_ids = set()

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            question_id = str(record.get("id", "")).strip()
            question_text = record.get("question")
            evidence = record.get("supporting_evidence")

            if not question_id:
                raise ValueError(
                    f"Missing question ID at line {line_number}."
                )

            if question_id in seen_ids:
                raise ValueError(
                    f"Duplicate question ID: {question_id}"
                )

            if not isinstance(question_text, str) or not question_text.strip():
                raise ValueError(
                    f"Question {question_id} has invalid question text."
                )

            if not isinstance(evidence, list) or not evidence:
                raise ValueError(
                    f"Question {question_id} has no supporting evidence."
                )

            relevant_ids, mappings = map_supporting_evidence(
                evidence,
                chunks_by_title,
                question_id,
            )

            seen_ids.add(question_id)

            questions.append(
                {
                    "id": question_id,
                    "question": question_text.strip(),
                    "type": record.get("type", "unknown"),
                    "level": record.get("level", "unknown"),
                    "relevant_chunk_ids": sorted(relevant_ids),
                    "supporting_evidence": mappings,
                }
            )

            if limit is not None and len(questions) >= limit:
                break

    if not questions:
        raise ValueError("No evaluation questions were loaded.")

    return questions


def verify_indexes(collection, chunks: list[dict], bm25_path: Path) -> dict:
    if not bm25_path.is_file():
        raise FileNotFoundError(
            f"BM25 database not found: {bm25_path}"
        )

    chroma_ids = set(collection.get(include=[])["ids"])
    chunk_ids = {chunk["id"] for chunk in chunks}

    uri = bm25_path.resolve().as_uri() + "?mode=ro"
    import sqlite3

    with sqlite3.connect(uri, uri=True) as connection:
        bm25_ids = {
            row[0]
            for row in connection.execute(
                "SELECT id FROM bm25_documents"
            )
        }

    if chroma_ids != bm25_ids:
        raise RuntimeError(
            "Chroma and BM25 IDs do not match. Evaluation stopped."
        )

    if chroma_ids != chunk_ids:
        raise RuntimeError(
            "Processed chunk IDs and index IDs do not match. "
            "Evaluation stopped."
        )

    return {
        "chroma_count": len(chroma_ids),
        "bm25_count": len(bm25_ids),
        "processed_chunk_count": len(chunk_ids),
        "ids_match": True,
        "read_only": True,
    }


async def timed_dense(vectorstore, query: str, limit: int, request_id: str):
    started = time.perf_counter()
    documents = await asyncio.to_thread(
        _retrieve_dense,
        vectorstore,
        query,
        limit,
        request_id,
    )
    return documents, (time.perf_counter() - started) * 1000


async def timed_bm25(query: str, limit: int, bm25_path: Path, request_id: str):
    started = time.perf_counter()
    documents = await asyncio.to_thread(
        _retrieve_bm25,
        query,
        limit,
        bm25_path,
        request_id,
    )
    return documents, (time.perf_counter() - started) * 1000


async def rerank_documents(
    query: str,
    documents: list,
    reranker_client: RerankerClient,
    top_n: int,
    request_id: str,
):
    if not documents:
        return [], 0.0

    if len(documents) > RERANKER_MAX_DOCUMENTS:
        raise ValueError(
            f"Reranker accepts at most {RERANKER_MAX_DOCUMENTS} candidates."
        )

    documents_by_id = {
        document_id(document): document
        for document in documents
    }

    started = time.perf_counter()

    results = await reranker_client.rerank(
        query=query,
        documents=[
            {
                "id": doc_id,
                "text": document.page_content,
            }
            for doc_id, document in documents_by_id.items()
        ],
        top_n=min(top_n, len(documents)),
        request_id=request_id,
    )

    elapsed_ms = (time.perf_counter() - started) * 1000

    ranked_documents = []
    for result in results:
        document = documents_by_id.get(result.id)
        if document is None:
            raise RuntimeError(
                f"Reranker returned unknown document ID: {result.id}"
            )
        ranked_documents.append(document)

    return ranked_documents, elapsed_ms


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def precision_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / k


def mrr_at_10(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(ranked_ids[:10], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_5(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:5], start=1)
        if doc_id in relevant_ids
    )

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
    covered = sum(
        bool(retrieved & set(item["relevant_chunk_ids"]))
        for item in evidence_mappings
    )
    return covered / len(evidence_mappings)


def evaluate_ranking(
    ranked_documents: list,
    relevant_ids: set[str],
    evidence_mappings: list[dict],
) -> dict:
    ranked_ids = [document_id(doc) for doc in ranked_documents]

    return {
        "retrieved_count": len(ranked_ids),
        "relevant_count": len(set(ranked_ids) & relevant_ids),
        "relevant_ids": sorted(set(ranked_ids) & relevant_ids),
        "recall@5": recall_at_k(ranked_ids, relevant_ids, 5),
        "recall@10": recall_at_k(ranked_ids, relevant_ids, 10),
        "precision@5": precision_at_k(ranked_ids, relevant_ids, 5),
        "mrr@10": mrr_at_10(ranked_ids, relevant_ids),
        "ndcg@5": ndcg_at_5(ranked_ids, relevant_ids),
        "evidence_coverage@5": evidence_coverage_at_k(
            ranked_ids, evidence_mappings, 5
        ),
        "top5": [
            {
                "rank": rank,
                "id": document_id(doc),
                "title": document_title(doc),
                "relevant": document_id(doc) in relevant_ids,
            }
            for rank, doc in enumerate(ranked_documents[:5], start=1)
        ],
        "ranked_ids": ranked_ids,
    }


async def run_question(
    question: dict,
    vectorstore,
    reranker_client: RerankerClient,
    bm25_path: Path,
    args,
) -> dict:
    query = question["question"]
    request_id = f"strategy-{question['id']}"
    relevant_ids = set(question["relevant_chunk_ids"])
    timings = {}

    # Dense top 10 and top 20 share the same top-20 retrieval result.
    dense_documents, timings["dense_top20_ms"] = await timed_dense(
        vectorstore,
        query,
        args.dense_k_max,
        request_id,
    )

    dense10 = dense_documents[:args.dense_k]
    dense20 = dense_documents[:args.dense_k_max]

    # A: Dense top 10, without reranking.
    strategy_a = dense10

    # B: Dense top 10, reranked.
    strategy_b, timings["dense10_rerank_ms"] = await rerank_documents(
        query,
        dense10,
        reranker_client,
        args.final_k,
        request_id + "-B",
    )

    # C: Dense top 20, reranked.
    strategy_c, timings["dense20_rerank_ms"] = await rerank_documents(
        query,
        dense20,
        reranker_client,
        args.final_k,
        request_id + "-C",
    )

    # D: Dense 10 + BM25 10, RRF top 20, reranked.
    bm25_documents, timings["bm25_top10_ms"] = await timed_bm25(
        query,
        args.bm25_k,
        bm25_path,
        request_id,
    )

    rrf_started = time.perf_counter()
    fused_documents = reciprocal_rank_fusion(
        dense10,
        bm25_documents,
        result_limit=args.rrf_candidate_k,
        rrf_constant=args.rrf_constant,
    )
    timings["rrf_ms"] = (time.perf_counter() - rrf_started) * 1000

    strategy_d, timings["hybrid_rerank_ms"] = await rerank_documents(
        query,
        fused_documents,
        reranker_client,
        args.final_k,
        request_id + "-D",
    )

    candidate_metrics = {
        "A_dense10": evaluate_ranking(
            dense10, relevant_ids, question["supporting_evidence"]
        ),
        "B_dense10_rerank": evaluate_ranking(
            dense10, relevant_ids, question["supporting_evidence"]
        ),
        "C_dense20_rerank": evaluate_ranking(
            dense20, relevant_ids, question["supporting_evidence"]
        ),
        "D_dense10_bm2510_rrf20": evaluate_ranking(
            fused_documents, relevant_ids, question["supporting_evidence"]
        ),
    }

    final_metrics = {
        "A_dense10": evaluate_ranking(
            strategy_a, relevant_ids, question["supporting_evidence"]
        ),
        "B_dense10_rerank": evaluate_ranking(
            strategy_b, relevant_ids, question["supporting_evidence"]
        ),
        "C_dense20_rerank": evaluate_ranking(
            strategy_c, relevant_ids, question["supporting_evidence"]
        ),
        "D_dense10_bm2510_rrf20": evaluate_ranking(
            strategy_d, relevant_ids, question["supporting_evidence"]
        ),
    }

    return {
        "id": question["id"],
        "question": query,
        "type": question["type"],
        "level": question["level"],
        "relevant_chunk_ids": sorted(relevant_ids),
        "supporting_evidence": question["supporting_evidence"],
        "candidate_metrics": candidate_metrics,
        "final_metrics": final_metrics,
        "latencies_ms": timings,
    }


def aggregate(rows: list[dict], section: str) -> dict:
    strategies = (
        "A_dense10",
        "B_dense10_rerank",
        "C_dense20_rerank",
        "D_dense10_bm2510_rrf20",
    )

    metric_names = (
        "recall@5",
        "recall@10",
        "precision@5",
        "mrr@10",
        "ndcg@5",
        "evidence_coverage@5",
    )

    summary = {}

    for strategy in strategies:
        summary[strategy] = {}

        for metric in metric_names:
            values = [
                row[section][strategy][metric]
                for row in rows
            ]
            summary[strategy][metric] = mean_or_none(values)

        summary[strategy]["hit@5"] = mean_or_none([
            float(row[section][strategy]["relevant_count"] > 0)
            for row in rows
        ])

        summary[strategy]["source_recall@5"] = summary[strategy][
            "evidence_coverage@5"
        ]

    return summary


def aggregate_latencies(rows: list[dict]) -> dict:
    latency_keys = sorted({
        key
        for row in rows
        for key in row["latencies_ms"]
    })

    result = {}
    for key in latency_keys:
        values = [
            row["latencies_ms"][key]
            for row in rows
            if key in row["latencies_ms"]
        ]
        result[key] = {
            "average_ms": mean_or_none(values),
            "p95_ms": percentile_95(values),
        }

    return result


async def check_reranker_ready() -> None:
    url = Config.RERANKER_SERVICE_URL.rstrip("/") + "/ready"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"Reranker is not ready at {url}. Start the reranker service first. "
            f"Details: {type(exc).__name__}: {exc}"
        ) from exc


async def run_all(questions, vectorstore, bm25_path: Path, args):
    await check_reranker_ready()

    rows = []

    async with RerankerClient() as reranker_client:
        print("Running one warm-up query. It will not be included in results.")
        await run_question(
            questions[0],
            vectorstore,
            reranker_client,
            bm25_path,
            args,
        )

        for index, question in enumerate(questions, start=1):
            print(
                f"\n[{index}/{len(questions)}] "
                f"{question['id']}: {question['question']}"
            )

            row = await run_question(
                question,
                vectorstore,
                reranker_client,
                bm25_path,
                args,
            )
            rows.append(row)

            for strategy, metrics in row["final_metrics"].items():
                print(
                    f"  {strategy}: "
                    f"recall@5={metrics['recall@5']:.3f}, "
                    f"MRR@10={metrics['mrr@10']:.3f}, "
                    f"coverage@5={metrics['evidence_coverage@5']:.3f}"
                )

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    fields = [
        "id",
        "question",
        "type",
        "level",
    ]

    strategies = (
        "A_dense10",
        "B_dense10_rerank",
        "C_dense20_rerank",
        "D_dense10_bm2510_rrf20",
    )

    metrics = (
        "recall@5",
        "recall@10",
        "precision@5",
        "mrr@10",
        "ndcg@5",
        "evidence_coverage@5",
        "hit@5",
    )

    for strategy in strategies:
        for metric in metrics:
            fields.append(f"{strategy}_{metric}")

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            csv_row = {
                "id": row["id"],
                "question": row["question"],
                "type": row["type"],
                "level": row["level"],
            }

            for strategy in strategies:
                result = row["final_metrics"][strategy]
                for metric in metrics:
                    if metric == "hit@5":
                        csv_row[f"{strategy}_{metric}"] = int(
                            result["relevant_count"] > 0
                        )
                    else:
                        csv_row[f"{strategy}_{metric}"] = result[metric]

            writer.writerow(csv_row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dense-k", type=int, default=10)
    parser.add_argument("--dense-k-max", type=int, default=20)
    parser.add_argument("--bm25-k", type=int, default=10)
    parser.add_argument("--rrf-candidate-k", type=int, default=20)
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.dense_k <= 0 or args.dense_k_max < args.dense_k:
        parser.error("dense-k-max must be >= dense-k, and both must be positive.")

    if args.bm25_k <= 0 or args.rrf_candidate_k <= 0:
        parser.error("BM25 and RRF candidate limits must be positive.")

    if args.rrf_candidate_k > RERANKER_MAX_DOCUMENTS:
        parser.error("RRF candidate pool cannot exceed 20 reranker documents.")

    if args.final_k <= 0 or args.final_k > RERANKER_MAX_DOCUMENTS:
        parser.error("final-k must be between 1 and 20.")

    chunks, chunks_by_title = load_chunks()
    questions = load_questions(
        args.questions,
        chunks_by_title,
        args.limit,
    )

    chroma_path = Path(Config.CHROMA_PERSIST_DIR).expanduser().resolve()
    bm25_path = Path(Config.BM25_DB_PATH).expanduser().resolve()

    if not chroma_path.is_dir():
        raise FileNotFoundError(
            f"Chroma persistence directory not found: {chroma_path}"
        )

    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_collection(Config.CHROMA_COLLECTION_NAME)

    index_status = verify_indexes(
        collection,
        chunks,
        bm25_path,
    )

    vectorstore = SimpleNamespace(
        _collection=collection,
        _embedding_function=OllamaEmbeddings(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_EMBEDDING_MODEL,
        ),
    )

    rows = asyncio.run(
        run_all(
            questions,
            vectorstore,
            bm25_path,
            args,
        )
    )

    result = {
        "experiment": "retrieval_strategy_experiments",
        "created_at": timestamp(),
        "dataset": {
            "questions_file": str(args.questions),
            "question_count": len(questions),
            "supporting_evidence_mapping": (
                "Supporting evidence title and exact sentence text mapped "
                "to processed chunks, matching Phase 4.2 evaluation."
            ),
        },
        "indexes": index_status,
        "settings": {
            "dense_k": args.dense_k,
            "dense_k_max": args.dense_k_max,
            "bm25_k": args.bm25_k,
            "rrf_candidate_k": args.rrf_candidate_k,
            "rrf_constant": args.rrf_constant,
            "final_k": args.final_k,
            "reranker_max_documents": RERANKER_MAX_DOCUMENTS,
            "embedding_model": Config.OLLAMA_EMBEDDING_MODEL,
            "reranker_service_url": Config.RERANKER_SERVICE_URL,
        },
        "candidate_stage_summary": aggregate(rows, "candidate_metrics"),
        "final_stage_summary": aggregate(rows, "final_metrics"),
        "latency_summary": aggregate_latencies(rows),
        "per_question": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"retrieval_strategy_{stamp}.json"
    csv_path = args.output_dir / f"retrieval_strategy_{stamp}.csv"

    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_csv(rows, csv_path)

    print("\n========== FINAL STRATEGY SUMMARY ==========")
    print(json.dumps(result["final_stage_summary"], indent=2))
    print("\nLatency:")
    print(json.dumps(result["latency_summary"], indent=2))
    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
