#!/usr/bin/env python3
"""Run one real LangGraph question and verify retrieval evidence reaches generation."""

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR / "src"))

import chromadb
from chromadb.config import Settings
from langchain_core.prompts import ChatPromptTemplate

graph_module = importlib.import_module("agent.graph")
from config import Config


TEST_QUESTIONS_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "hotpotqa_retrieval_eval_20.jsonl"
)

DEFAULT_OUTPUT = TESTS_DIR / "results" / "phase4_1_graph_smoke.json"


def _first_question():
    with TEST_QUESTIONS_PATH.open(encoding="utf-8") as questions_file:
        for line in questions_file:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"No processed questions found at {TEST_QUESTIONS_PATH}.")


def _verify_indexes_exist():
    chroma_path = Path(Config.CHROMA_PERSIST_DIR).expanduser().resolve()
    bm25_path = Path(Config.BM25_DB_PATH).expanduser().resolve()
    if not chroma_path.is_dir() or not bm25_path.is_file():
        raise FileNotFoundError("Configured production Chroma or BM25 index is missing.")
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )
    client.get_collection(Config.CHROMA_COLLECTION_NAME)


async def _run(question: str):
    retrieval_calls = []
    answer_contexts = []
    original_retrieve = graph_module.rag_system.retrieve_documents

    async def capture_retrieval(query):
        documents = await original_retrieve(query)
        retrieval_calls.append(
            {
                "query": query,
                "documents": documents,
            }
        )
        return documents

    original_prompt_ainvoke = ChatPromptTemplate.ainvoke

    async def capture_prompt_context(prompt, input, *args, **kwargs):
        if isinstance(input, dict) and isinstance(input.get("context"), str):
            answer_contexts.append(input["context"])
        return await original_prompt_ainvoke(prompt, input, *args, **kwargs)

    config = {
        "configurable": {
            "model": Config.OLLAMA_GENERATION_MODEL,
            "temperature": 0.0,
            "max_tokens": 200,
        }
    }

    with patch.object(
        graph_module.rag_system,
        "retrieve_documents",
        new=capture_retrieval,
    ), patch.object(
        ChatPromptTemplate,
        "ainvoke",
        new=capture_prompt_context,
    ):
        result = await graph_module.graph.ainvoke(
            {"original_question": question},
            config=config,
        )
    return result, retrieval_calls, answer_contexts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="Question text; defaults to the first processed HotpotQA question.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    _verify_indexes_exist()
    question_record = _first_question()
    question = args.question or question_record["question"]
    result, retrieval_calls, answer_contexts = asyncio.run(_run(question))

    retrieved_documents = [
        document
        for call in retrieval_calls
        for document in call["documents"]
    ]
    evidence_reached_generation = any(
        document.page_content in context
        for document in retrieved_documents
        for context in answer_contexts
    )
    final_answer = result.get("final_answer", "")
    report = {
        "question_id": question_record.get("id") if not args.question else None,
        "question": question,
        "classifier_structured_output": "Ollama native JSON Schema",
        "classification": result.get("classification"),
        "hybrid_retrieval_calls": len(retrieval_calls),
        "retrieved_document_count": len(retrieved_documents),
        "retrieved_documents": [
            {
                "id": document.metadata.get("id"),
                "title": document.metadata.get("title"),
                "content_preview": document.page_content[:300],
            }
            for document in retrieved_documents
        ],
        "evidence_reached_answer_generation": evidence_reached_generation,
        "answer_context_count": len(answer_contexts),
        "final_answer": final_answer,
        "passed": bool(retrieval_calls and retrieved_documents and evidence_reached_generation and final_answer.strip()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({**report, "output": str(args.output)}, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise RuntimeError("LangGraph smoke test did not complete retrieval and answer generation.")


if __name__ == "__main__":
    main()
