from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict


def _record_to_dict(record: Any) -> dict[str, Any]:
    """Normalize Pydantic models and dictionaries into plain dictionaries."""
    if hasattr(record, "model_dump"):
        return record.model_dump(mode="json")
    if isinstance(record, dict):
        return record
    raise TypeError(f"Unsupported state record type: {type(record).__name__}")


def _merge_records_by_key(
    existing: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    """Insert or update records while preserving their original order."""
    merged = [_record_to_dict(item) for item in existing]

    positions = {
        item[key]: index
        for index, item in enumerate(merged)
        if key in item
    }

    for update in updates:
        item = _record_to_dict(update)
        record_id = item.get(key)

        if not record_id:
            raise ValueError(f"State record is missing required key: {key}")

        if record_id in positions:
            index = positions[record_id]
            merged[index] = {**merged[index], **item}
        else:
            positions[record_id] = len(merged)
            merged.append(item)

    return merged


def merge_task_results(
    existing: list[dict[str, Any]],
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge worker results by task ID."""
    return _merge_records_by_key(existing, updates, "task_id")


def merge_evidence(
    existing: list[dict[str, Any]],
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate documents while preserving all task provenance."""
    merged = _merge_records_by_key(existing, updates, "doc_id")

    existing_by_id = {
        item["doc_id"]: _record_to_dict(item)
        for item in existing
    }

    for record in merged:
        record["source_task_ids"] = list(
            dict.fromkeys(record.get("source_task_ids", []))
        )

    for update in updates:
        item = _record_to_dict(update)
        record = next(
            record for record in merged
            if record["doc_id"] == item["doc_id"]
        )

        previous = existing_by_id.get(item["doc_id"], {})
        provenance = (
            previous.get("source_task_ids", [])
            + item.get("source_task_ids", [])
        )
        record["source_task_ids"] = list(dict.fromkeys(provenance))

    return merged


def merge_trace(
    existing: list[dict[str, Any]],
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge trace entries without duplicating cumulative trace updates."""
    merged = [_record_to_dict(item) for item in existing]

    seen = {
        item.get("event_id") or json.dumps(
            item, sort_keys=True, default=str
        )
        for item in merged
    }

    for update in updates:
        item = _record_to_dict(update)
        identity = item.get("event_id") or json.dumps(
            item, sort_keys=True, default=str
        )

        if identity not in seen:
            merged.append(item)
            seen.add(identity)

    return merged


class MultiHopState(TypedDict, total=False):
    """Shared state for the dynamic multi-hop RAG graph."""

    # Original request and supervisor routing
    original_question: str
    request_id: str
    classification: str
    classification_reason: str
    worker_answer: str

    # Legacy workflow fields retained for compatibility
    decomposed_questions: list[str]
    intermediate_answers: list[str]
    retrieved_docs: list[Any]
    rewritten_query: str
    current_question_index: int
    final_answer: str

    # Planner output
    tasks: list[dict[str, Any]]

    # Parallel worker outputs
    task_results: Annotated[
        list[dict[str, Any]],
        merge_task_results,
    ]
    evidence: Annotated[
        list[dict[str, Any]],
        merge_evidence,
    ]

    # Execution control
    execution_status: str
    execution_limits: dict[str, int]
    followup_round: int
    followup_tasks_created: bool

    # Evidence evaluation and response validation
    evidence_assessment: dict[str, Any]
    grounding_result: dict[str, Any]

    # Observability
    trace: Annotated[
        list[dict[str, Any]],
        merge_trace,
    ]