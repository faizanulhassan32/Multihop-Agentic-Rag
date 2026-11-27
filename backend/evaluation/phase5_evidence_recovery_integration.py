#!/usr/bin/env python3
"""Deterministically exercise Phase 5 follow-up retrieval through the live graph.

Run from backend:
    PYTHONPATH=src python evaluation/phase5_evidence_recovery_integration.py

The classifier, task planner, retrieval stack (Chroma + BM25 + reranker),
follow-up planner, final answer model, and LangGraph routing remain live.
Only the first evidence assessment is controlled: it is forced to report a
specific gap so the test must execute the recovery path. The second assessment
uses the real lexical evidence assessor.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "src"))

from config import Config  # noqa: E402
from ingestion.preprocess import PROCESSED_SAMPLE_PATH  # noqa: E402


def _load_default_question() -> dict[str, Any]:
    with PROCESSED_SAMPLE_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"No questions found in {PROCESSED_SAMPLE_PATH}")


def _verify_indexes() -> dict[str, str]:
    chroma_path = Path(Config.CHROMA_PERSIST_DIR).expanduser().resolve()
    bm25_path = Path(Config.BM25_DB_PATH).expanduser().resolve()
    if not chroma_path.is_dir():
        raise FileNotFoundError(f"Configured Chroma index not found: {chroma_path}")
    if not bm25_path.is_file():
        raise FileNotFoundError(f"Configured BM25 index not found: {bm25_path}")
    return {"chroma_path": str(chroma_path), "bm25_path": str(bm25_path)}


def _forced_gap_assessment(real_assessment: Any, task_id: str) -> Any:
    """Clone a valid assessment but force one actionable gap for round one."""
    payload = real_assessment.model_dump()
    payload["status"] = "insufficient"
    payload["coverage_score"] = 0.0
    payload["gap_task_ids"] = [task_id]
    payload["reason"] = "Integration test: force one recoverable evidence gap."

    task_assessments = payload.get("task_assessments", [])
    matching = next((item for item in task_assessments if item.get("task_id") == task_id), None)
    if matching is None:
        matching = {
            "task_id": task_id,
            "status": "insufficient",
            "coverage_score": 0.0,
            "matched_terms": [],
            "missing_terms": ["founding date", "publication year"],
            "evidence_ids": [],
            "reason": "Integration test: force a recoverable gap.",
        }
        task_assessments.append(matching)
    else:
        matching["status"] = "insufficient"
        matching["coverage_score"] = 0.0
        # These terms are deliberately relevant to the magazine comparison.
        matching["missing_terms"] = ["founding date", "publication year"]
        matching["reason"] = "Integration test: force a recoverable gap."
    payload["task_assessments"] = task_assessments

    models = importlib.import_module("agent.evidence_models")
    assessment_model = getattr(models, "EvidenceAssessment")
    return assessment_model.model_validate(payload)


async def _run(question: str, request_id: str) -> dict[str, Any]:
    graph_module = importlib.import_module("agent.graph")
    orchestrator_module = importlib.import_module("agent.orchestrator")
    executor_module = importlib.import_module("agent.task_executor")

    retrieval_log: list[dict[str, Any]] = []
    generation_prompts: list[str] = []
    assessment_log: list[dict[str, Any]] = []
    original_retrieve = executor_module.rag_system.retrieve_documents
    original_chat = graph_module.AsyncClient.chat
    original_assess = orchestrator_module.assess_evidence

    async def capture_retrieval(*args: Any, **kwargs: Any):
        query = kwargs.get("query")
        if query is None and args:
            query = args[0]
        docs = await original_retrieve(*args, **kwargs)
        retrieval_log.append({
            "query": str(query or ""),
            "request_id": kwargs.get("request_id"),
            "documents": [
                {
                    "doc_id": doc.metadata.get("id"),
                    "title": doc.metadata.get("title"),
                    "content": doc.page_content,
                }
                for doc in docs
            ],
        })
        return docs

    async def capture_chat(client: Any, *args: Any, **kwargs: Any):
        messages = kwargs.get("messages")
        if messages is None and args:
            messages = args[0]
        if messages:
            generation_prompts.append("\n".join(
                str(message.get("content", "")) for message in messages
            ))
        return await original_chat(client, *args, **kwargs)

    assessment_calls = 0

    def force_first_gap(*, tasks: Any, evidence: Any):
        nonlocal assessment_calls
        real_result = original_assess(tasks=tasks, evidence=evidence)
        assessment_calls += 1
        task_ids = [
            task.get("task_id") if isinstance(task, dict) else task.task_id
            for task in tasks
        ]
        if assessment_calls == 1:
            if not task_ids:
                raise AssertionError("No task available to attach the forced evidence gap.")
            result = _forced_gap_assessment(real_result, task_ids[0])
            assessment_log.append({
                "call": assessment_calls,
                "forced_gap": True,
                "status": result.status,
                "gap_task_ids": result.gap_task_ids,
                "evidence_count": len(evidence),
            })
            return result

        assessment_log.append({
            "call": assessment_calls,
            "forced_gap": False,
            "status": real_result.status,
            "gap_task_ids": real_result.gap_task_ids,
            "evidence_count": len(evidence),
        })
        return real_result

    run_config = {
        "configurable": {
            "model": Config.OLLAMA_GENERATION_MODEL,
            "temperature": 0.0,
            "max_tokens": 250,
        },
        "metadata": {"request_id": request_id},
    }

    executor_module.rag_system.startup()
    try:
        with (
            patch.object(executor_module.rag_system, "retrieve_documents", new=capture_retrieval),
            patch.object(graph_module.AsyncClient, "chat", new=capture_chat),
            patch.object(orchestrator_module, "assess_evidence", new=force_first_gap),
        ):
            state = await graph_module.graph.ainvoke(
                {"original_question": question},
                config=run_config,
            )
    finally:
        await executor_module.rag_system.shutdown()

    tasks = state.get("tasks", [])
    evidence = state.get("evidence", [])
    followup_tasks = [
        task for task in tasks
        if int(task.get("followup_round", 0)) > 0
        or "_followup_" in task.get("task_id", "")
    ]
    followup_ids = {task.get("task_id") for task in followup_tasks}
    followup_evidence = [
        item for item in evidence
        if followup_ids.intersection(item.get("source_task_ids", []))
    ]
    final_prompt = generation_prompts[-1] if generation_prompts else ""
    evidence_in_final_prompt = bool(evidence) and all(
        item.get("content", "") in final_prompt
        for item in evidence if item.get("content")
    )
    followup_evidence_in_final_prompt = bool(followup_evidence) and all(
        item.get("content", "") in final_prompt
        for item in followup_evidence if item.get("content")
    )

    return {
        "state": state,
        "retrieval_log": retrieval_log,
        "generation_prompt_count": len(generation_prompts),
        "final_prompt": final_prompt,
        "assessment_log": assessment_log,
        "followup_tasks": followup_tasks,
        "followup_evidence": followup_evidence,
        "evidence_in_final_prompt": evidence_in_final_prompt,
        "followup_evidence_in_final_prompt": followup_evidence_in_final_prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="Exact question to use instead of the first processed sample.")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "results" / "phase5_evidence_recovery_integration.json",
    )
    args = parser.parse_args()

    indexes = _verify_indexes()
    sample = _load_default_question()
    question = args.question or sample["question"]
    request_id = f"phase5-recovery-{uuid.uuid4().hex[:8]}"
    result = asyncio.run(_run(question, request_id))
    state = result["state"]

    followup_tasks = result["followup_tasks"]
    followup_evidence = result["followup_evidence"]
    retrieval_log = result["retrieval_log"]
    initial_doc_ids = {
        doc.get("doc_id") for doc in retrieval_log[0]["documents"]
        if doc.get("doc_id")
    } if retrieval_log else set()
    followup_doc_ids = {
        doc.get("doc_id") for doc in retrieval_log[1]["documents"]
        if doc.get("doc_id")
    } if len(retrieval_log) >= 2 else set()
    newly_retrieved_doc_ids = sorted(followup_doc_ids - initial_doc_ids)
    shared_doc_ids = sorted(followup_doc_ids & initial_doc_ids)
    followup_overlap_ratio = (
        len(shared_doc_ids) / len(followup_doc_ids)
        if followup_doc_ids else None
    )
    trace_steps = [
        item.get("step") for item in state.get("trace", [])
        if isinstance(item, dict)
    ]
    checks = {
        "classified_in_scope": state.get("classification") == "in_scope",
        "initial_retrieval_ran": len(retrieval_log) >= 1,
        "first_assessment_forced_gap": bool(result["assessment_log"])
            and result["assessment_log"][0]["forced_gap"],
        "followup_task_created": len(followup_tasks) >= 1,
        "followup_round_incremented": int(state.get("followup_round", 0)) >= 1,
        "followup_retrieval_ran": len(retrieval_log) >= 2,
        "followup_evidence_attributed": len(followup_evidence) >= 1,
        "reassessment_ran": len(result["assessment_log"]) >= 2,
        "final_answer_generated": bool(state.get("final_answer", "").strip()),
        "followup_evidence_in_final_prompt": result["followup_evidence_in_final_prompt"],
    }
    passed = all(checks.values())

    report = {
        "test": "phase5_evidence_recovery_integration",
        "request_id": request_id,
        "question_id": sample.get("id") if not args.question else None,
        "question": question,
        "indexes": indexes,
        "classification": state.get("classification"),
        "followup_round": state.get("followup_round", 0),
        "tasks": [
            {
                "task_id": task.get("task_id"),
                "followup_round": task.get("followup_round", 0),
                "retrieval_query": task.get("retrieval_query"),
                "status": task.get("status"),
            }
            for task in state.get("tasks", [])
        ],
        "retrieval_calls": [
            {
                "query": call["query"],
                "document_count": len(call["documents"]),
                "doc_ids": [doc["doc_id"] for doc in call["documents"]],
            }
            for call in retrieval_log
        ],
        "retrieval_quality_comparison": {
            "initial_unique_documents": len(initial_doc_ids),
            "followup_unique_documents": len(followup_doc_ids),
            "new_documents_from_followup": len(newly_retrieved_doc_ids),
            "new_document_ids": newly_retrieved_doc_ids,
            "shared_document_count": len(shared_doc_ids),
            "followup_overlap_ratio": followup_overlap_ratio,
            "expanded_document_set": bool(newly_retrieved_doc_ids),
            "interpretation": (
                "Follow-up retrieval expanded the document set."
                if newly_retrieved_doc_ids
                else "Follow-up retrieval ran successfully but returned no documents beyond the initial result set."
            ),
        },
        "assessment_log": result["assessment_log"],
        "final_assessment": state.get("evidence_assessment", {}),
        "evidence_count_final": len(state.get("evidence", [])),
        "followup_evidence_count": len(followup_evidence),
        "followup_evidence_doc_ids": [item.get("doc_id") for item in followup_evidence],
        "trace_steps": trace_steps,
        "generation_prompt_count": result["generation_prompt_count"],
        "followup_evidence_in_final_prompt": result["followup_evidence_in_final_prompt"],
        "evidence_in_final_prompt": result["evidence_in_final_prompt"],
        "final_answer": state.get("final_answer", ""),
        "checks": checks,
        "passed": passed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_path": str(args.output)}, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit("Phase 5 evidence recovery integration test failed. Review the report.")


if __name__ == "__main__":
    main()
