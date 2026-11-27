#!/usr/bin/env python3
"""Read-only Dense-vs-Hybrid retrieval evaluation over the processed HotpotQA sample."""

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Set


BACKEND_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR / "src"))

import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings

from agent import hybrid_retriever
from agent.hybrid_retriever import _retrieve_dense, hybrid_search
from config import Config
from ingestion.preprocess import CHUNKS_PATH, PROCESSED_SAMPLE_PATH


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
DEFAULT_OUTPUT = TESTS_DIR / "results" / "phase4_1_dense_vs_hybrid.json"


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _article_id(title: str, sentences: List[str]) -> str:
    normalized_title = _normalize(title).lower()
    article_text = _normalize(" ".join(_normalize(sentence) for sentence in sentences))
    digest = hashlib.sha256(
        f"{normalized_title}:{article_text}".encode()
    ).hexdigest()[:8]
    return f"{normalized_title}_{digest}"


def _load_chunks_by_article() -> Dict[str, List[Dict]]:
    chunks_by_article: Dict[str, List[Dict]] = {}
    with CHUNKS_PATH.open(encoding="utf-8") as chunks_file:
        for line in chunks_file:
            if line.strip():
                chunk = json.loads(line)
                article_id = chunk["metadata"]["article_id"]
                chunks_by_article.setdefault(article_id, []).append(chunk)
    return chunks_by_article


def _supporting_chunk_ids(
    question: Dict, chunks_by_article: Dict[str, List[Dict]]
) -> tuple[Set[str], List[str], int, int]:
    support_facts = question.get("supporting_facts") or {}
    support_titles = support_facts.get("title", [])
    sentence_ids = support_facts.get("sent_id", [])
    facts = list(zip(support_titles, sentence_ids))
    context = question.get("context") or {}
    context_titles = context.get("title", [])
    sentence_lists = context.get("sentences", [])

    relevant_ids: Set[str] = set()
    mapped_facts = 0
    for support_title, sentence_id in facts:
        matching_chunks = []
        for title, sentences in zip(context_titles, sentence_lists):
            if _normalize(title).casefold() != _normalize(support_title).casefold():
                continue
            if not isinstance(sentence_id, int) or not 0 <= sentence_id < len(sentences):
                continue

            article_chunks = chunks_by_article.get(_article_id(title, sentences), [])
            support_sentence = _normalize(sentences[sentence_id])
            matching_chunks.extend(
                chunk
                for chunk in article_chunks
                if support_sentence and support_sentence in chunk["content"]
            )

        if matching_chunks:
            mapped_facts += 1
            relevant_ids.update(chunk["id"] for chunk in matching_chunks)
    return relevant_ids, list(support_titles), mapped_facts, len(facts)


def _load_evaluation_questions(limit: int, chunks_by_article: Dict[str, List[Dict]]):
    eligible = []
    total_records = 0
    mapped_records = 0
    with PROCESSED_SAMPLE_PATH.open(encoding="utf-8") as questions_file:
        for line in questions_file:
            if not line.strip():
                continue
            total_records += 1
            question = json.loads(line)
            relevant_ids, titles, mapped_facts, fact_count = _supporting_chunk_ids(
                question, chunks_by_article
            )
            if fact_count and mapped_facts == fact_count and relevant_ids:
                mapped_records += 1
                eligible.append(
                    {
                        "id": question.get("id"),
                        "question": question["question"].strip(),
                        "supporting_titles": titles,
                        "relevant_chunk_ids": sorted(relevant_ids),
                    }
                )
    return eligible[:limit], total_records, mapped_records


def _read_only_bm25_ids(db_path: Path) -> Set[str]:
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return {row[0] for row in connection.execute("SELECT id FROM bm25_documents")}


def _read_only_bm25_search(query: str, limit: int = 10, db_path=None):
    path = Path(db_path if db_path is not None else Config.BM25_DB_PATH)
    tokens = TOKEN_PATTERN.findall(query)
    if not tokens or limit <= 0:
        return []
    match_query = " OR ".join(f'"{token}"' for token in tokens)
    uri = path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            """
            SELECT d.id, d.content, d.metadata_json, bm25(bm25_fts) AS rank
            FROM bm25_fts
            JOIN bm25_documents AS d ON d.id = bm25_fts.id
            WHERE bm25_fts MATCH ?
            ORDER BY rank ASC, d.id ASC
            LIMIT ?
            """,
            (match_query, limit),
        ).fetchall()
    return [
        {
            "id": row[0],
            "content": row[1],
            "metadata": json.loads(row[2]),
            "score": row[3],
        }
        for row in rows
    ]


def _recall(retrieved_ids: List[str], relevant_ids: Set[str], cutoff: int) -> float:
    return len(set(retrieved_ids[:cutoff]) & relevant_ids) / len(relevant_ids)


def _mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    for rank, document_id in enumerate(retrieved_ids[:10], start=1):
        if document_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _ndcg_at_5(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, document_id in enumerate(retrieved_ids[:5], start=1)
        if document_id in relevant_ids
    )
    ideal_count = min(5, len(relevant_ids))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _metrics(rows: List[Dict], method: str, latencies_ms: List[float]) -> Dict:
    return {
        "recall@5": sum(row[method]["recall@5"] for row in rows) / len(rows),
        "recall@10": sum(row[method]["recall@10"] for row in rows) / len(rows),
        "mrr@10": sum(row[method]["mrr@10"] for row in rows) / len(rows),
        "ndcg@5": sum(row[method]["ndcg@5"] for row in rows) / len(rows),
        "average_latency_ms": sum(latencies_ms) / len(latencies_ms),
        "p95_latency_ms": sorted(latencies_ms)[math.ceil(0.95 * len(latencies_ms)) - 1],
    }


async def _timed_dense(vectorstore, query: str, limit: int):
    start = time.perf_counter()
    results = await asyncio.to_thread(_retrieve_dense, vectorstore, query, limit)
    return results, (time.perf_counter() - start) * 1000


async def _timed_hybrid(query: str, vectorstore, args, bm25_path: Path):
    start = time.perf_counter()
    results = await hybrid_search(
        query=query,
        vectorstore=vectorstore,
        dense_limit=args.dense_k,
        bm25_limit=args.bm25_k,
        result_limit=args.result_k,
        rrf_constant=args.rrf_k,
        bm25_db_path=bm25_path,
    )
    return results, (time.perf_counter() - start) * 1000


async def _evaluate(questions, vectorstore, args, bm25_path: Path):
    first_query = questions[0]["question"]
    await _timed_dense(vectorstore, first_query, args.dense_k)
    await _timed_hybrid(first_query, vectorstore, args, bm25_path)

    rows = []
    dense_latencies = []
    hybrid_latencies = []
    for index, question in enumerate(questions):
        query = question["question"]
        if index % 2 == 0:
            dense_results, dense_ms = await _timed_dense(
                vectorstore, query, args.dense_k
            )
            hybrid_results, hybrid_ms = await _timed_hybrid(
                query, vectorstore, args, bm25_path
            )
        else:
            hybrid_results, hybrid_ms = await _timed_hybrid(
                query, vectorstore, args, bm25_path
            )
            dense_results, dense_ms = await _timed_dense(
                vectorstore, query, args.dense_k
            )

        bm25_candidates = await asyncio.to_thread(
            _read_only_bm25_search, query, args.bm25_k, bm25_path
        )
        relevant_ids = set(question["relevant_chunk_ids"])
        dense_ids = [document.metadata["id"] for document in dense_results]
        hybrid_ids = [document.metadata["id"] for document in hybrid_results]
        bm25_ids = [document["id"] for document in bm25_candidates]
        row = {
            **question,
            "bm25_candidate_ids": bm25_ids,
            "bm25_relevant_ids": sorted(set(bm25_ids) & relevant_ids),
            "dense": {
                "retrieved_ids": dense_ids,
                "relevant_ids": sorted(set(dense_ids) & relevant_ids),
                "recall@5": _recall(dense_ids, relevant_ids, 5),
                "recall@10": _recall(dense_ids, relevant_ids, 10),
                "mrr@10": _mrr(dense_ids, relevant_ids),
                "ndcg@5": _ndcg_at_5(dense_ids, relevant_ids),
                "latency_ms": dense_ms,
            },
            "hybrid": {
                "retrieved_ids": hybrid_ids,
                "relevant_ids": sorted(set(hybrid_ids) & relevant_ids),
                "recall@5": _recall(hybrid_ids, relevant_ids, 5),
                "recall@10": _recall(hybrid_ids, relevant_ids, 10),
                "mrr@10": _mrr(hybrid_ids, relevant_ids),
                "ndcg@5": _ndcg_at_5(hybrid_ids, relevant_ids),
                "latency_ms": hybrid_ms,
            },
        }
        rows.append(row)
        dense_latencies.append(dense_ms)
        hybrid_latencies.append(hybrid_ms)
    return rows, dense_latencies, hybrid_latencies


def _per_question_analysis(rows: List[Dict]) -> Dict[str, List[Dict]]:
    improved, dense_better, both_miss = [], [], []
    for row in rows:
        dense_recall = row["dense"]["recall@10"]
        hybrid_recall = row["hybrid"]["recall@10"]
        example = {
            "id": row["id"],
            "question": row["question"],
            "supporting_titles": row["supporting_titles"],
            "bm25_relevant_ids": row["bm25_relevant_ids"],
            "dense_relevant_ids": row["dense"]["relevant_ids"],
            "hybrid_relevant_ids": row["hybrid"]["relevant_ids"],
            "dense_recall@10": dense_recall,
            "hybrid_recall@10": hybrid_recall,
        }
        if hybrid_recall > dense_recall:
            improved.append(example)
        elif dense_recall > hybrid_recall:
            dense_better.append(example)
        elif dense_recall == 0:
            both_miss.append(example)
    return {
        "hybrid_improves_recall@10": improved,
        "dense_better_recall@10": dense_better,
        "both_miss_all_supporting_chunks@10": both_miss,
    }


def _paired_delta(rows: List[Dict], metric: str) -> Dict:
    differences = [
        row["hybrid"][metric] - row["dense"][metric]
        for row in rows
    ]
    generator = random.Random(42)
    bootstrap_means = sorted(
        sum(generator.choices(differences, k=len(differences))) / len(differences)
        for _ in range(5000)
    )
    return {
        "hybrid_minus_dense": sum(differences) / len(differences),
        "bootstrap_95_ci": [
            bootstrap_means[int(0.025 * len(bootstrap_means))],
            bootstrap_means[int(0.975 * len(bootstrap_means))],
        ],
        "questions_improved": sum(value > 0 for value in differences),
        "questions_regressed": sum(value < 0 for value in differences),
        "questions_tied": sum(value == 0 for value in differences),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--dense-k", type=int, default=10)
    parser.add_argument("--bm25-k", type=int, default=10)
    parser.add_argument("--result-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if min(args.limit, args.dense_k, args.bm25_k, args.result_k, args.rrf_k) <= 0:
        parser.error("All question and retrieval limits and RRF K must be positive.")

    chunks_by_article = _load_chunks_by_article()
    questions, total_records, fully_mapped = _load_evaluation_questions(
        args.limit, chunks_by_article
    )
    if len(questions) < min(50, args.limit):
        raise RuntimeError(
            f"Only {len(questions)} questions have complete supporting-chunk labels; "
            "at least 50 are required when available."
        )

    chroma_path = Path(Config.CHROMA_PERSIST_DIR).expanduser().resolve()
    bm25_path = Path(Config.BM25_DB_PATH).expanduser().resolve()
    if not chroma_path.is_dir() or not bm25_path.is_file():
        raise FileNotFoundError("Configured production Chroma or BM25 index is missing.")

    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(Config.CHROMA_COLLECTION_NAME)
    chroma_ids = set(collection.get(include=[])["ids"])
    bm25_ids = _read_only_bm25_ids(bm25_path)
    if chroma_ids != bm25_ids:
        raise RuntimeError(
            "Production Chroma and BM25 indexes do not have matching IDs; "
            "evaluation aborted."
        )

    vectorstore = SimpleNamespace(
        _collection=collection,
        _embedding_function=OllamaEmbeddings(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_EMBEDDING_MODEL,
        ),
    )
    original_bm25_search = hybrid_retriever.search_bm25
    hybrid_retriever.search_bm25 = _read_only_bm25_search
    try:
        rows, dense_latencies, hybrid_latencies = asyncio.run(
            _evaluate(questions, vectorstore, args, bm25_path)
        )
    finally:
        hybrid_retriever.search_bm25 = original_bm25_search

    analysis = _per_question_analysis(rows)
    result = {
        "dataset": {
            "name": Config.HOTPOT_DATASET,
            "config": Config.HOTPOT_CONFIG,
            "split": Config.HOTPOT_SPLIT,
            "source": str(PROCESSED_SAMPLE_PATH),
            "processed_question_count": total_records,
            "questions_with_complete_support_mapping": fully_mapped,
            "evaluated_question_count": len(rows),
            "question_selection": (
                "First N fully mapped questions in processed JSONL file order; "
                "question IDs are recorded per row."
            ),
            "support_mapping": (
                "Each HotpotQA supporting_facts title/sent_id pair maps to the "
                "indexed chunk(s) containing that normalized support sentence."
            ),
        },
        "indexes": {
            "chroma_collection": Config.CHROMA_COLLECTION_NAME,
            "chroma_document_count": len(chroma_ids),
            "bm25_document_count": len(bm25_ids),
            "ids_match": True,
            "read_only": True,
        },
        "settings": {
            "dense_candidate_count": args.dense_k,
            "bm25_candidate_count": args.bm25_k,
            "final_result_count": args.result_k,
            "rrf_constant": args.rrf_k,
            "dense_and_hybrid_use_same_questions": True,
            "warmup_queries_excluded_from_metrics": 2,
            "execution_order": "Alternating Dense-first and Hybrid-first by question.",
            "embedding_model": Config.OLLAMA_EMBEDDING_MODEL,
        },
        "metrics": {
            "dense_only": _metrics(rows, "dense", dense_latencies),
            "hybrid": _metrics(rows, "hybrid", hybrid_latencies),
            "latency_unit": "milliseconds",
            "mrr_cutoff": 10,
        },
        "paired_deltas": {
            metric: _paired_delta(rows, metric)
            for metric in ("recall@5", "recall@10", "mrr@10", "ndcg@5")
        },
        "per_question_analysis": analysis,
        "per_question": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "evaluated_questions": len(rows),
        "indexes": result["indexes"],
        "settings": result["settings"],
        "metrics": result["metrics"],
        "analysis_counts": {key: len(value) for key, value in analysis.items()},
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
