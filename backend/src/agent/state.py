from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict


def _record_to_dict(record: Any) -> dict[str, Any]:
    """Normalize Pydantic models and dictionaries into plain dictionaries."""
    if hasattr(record, "model_dump"):
        return record.model_dump(mode="json")

    if isinstance(record, dict):
        return dict(record)

    raise TypeError(
        f"Unsupported state record type: {type(record).__name__}"
    )


def _merge_records_by_key(
    existing: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    """Insert or update records while preserving their original order."""
    merged = [
        _record_to_dict(item)
        for item in existing
    ]

    positions = {
        item[key]: index
        for index, item in enumerate(merged)
        if key in item
    }

    for update in updates:
        item = _record_to_dict(update)
        record_id = item.get(key)

        if not record_id:
            raise ValueError(
                f"State record is missing required key: {key}"
            )

        if record_id in positions:
            index = positions[record_id]
            merged[index] = {
                **merged[index],
                **item,
            }
        else:
            positions[record_id] = len(merged)
            merged.append(item)

    return merged


def merge_task_results(
    existing: list[dict[str, Any]],
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge worker results by task ID."""
    return _merge_records_by_key(
        existing,
        updates,
        "task_id",
    )


def _ordered_unique(values: Any) -> list[Any]:
    """Return values in first-seen order without duplicates."""
    result = []
    seen = set()

    for value in values or []:
        marker = json.dumps(
            value,
            sort_keys=True,
            default=str,
        )

        if marker not in seen:
            result.append(value)
            seen.add(marker)

    return result


def merge_evidence(
    existing: list[dict[str, Any]],
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Deduplicate evidence by stable document ID while preserving provenance.

    Parallel workers may return the same document in the same reducer
    invocation. Every task attribution and retrieval round must survive.
    """
    merged = []
    positions = {}

    for raw_record in [
        *(existing or []),
        *(updates or []),
    ]:
        item = _record_to_dict(raw_record)

        doc_id = item.get("doc_id")

        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(
                "Evidence record is missing a valid doc_id."
            )

        doc_id = doc_id.strip()

        incoming_task_ids = [
            str(value).strip()
            for value in item.get("source_task_ids", [])
            if str(value).strip()
        ]

        incoming_rounds = [
            int(value)
            for value in item.get("retrieval_rounds", [])
            if value is not None
        ]

        if doc_id not in positions:
            record = dict(item)

            record["doc_id"] = doc_id
            record["source_task_ids"] = _ordered_unique(
                incoming_task_ids
            )
            record["retrieval_rounds"] = _ordered_unique(
                incoming_rounds
            )
            record["metadata"] = dict(
                record.get("metadata") or {}
            )

            positions[doc_id] = len(merged)
            merged.append(record)

            continue

        current = merged[positions[doc_id]]

        # Preserve provenance from every worker and retrieval round.
        current["source_task_ids"] = _ordered_unique(
            [
                *current.get("source_task_ids", []),
                *incoming_task_ids,
            ]
        )

        current["retrieval_rounds"] = _ordered_unique(
            [
                *current.get("retrieval_rounds", []),
                *incoming_rounds,
            ]
        )

        # Preserve the first available canonical document content.
        if not current.get("content") and item.get("content"):
            current["content"] = item["content"]

        # Merge metadata without discarding earlier metadata.
        metadata = dict(
            current.get("metadata") or {}
        )

        for metadata_key, metadata_value in (
            item.get("metadata") or {}
        ).items():

            if (
                metadata_key not in metadata
                or metadata[metadata_key] in (None, "", [], {})
            ):
                metadata[metadata_key] = metadata_value

        current["metadata"] = metadata

        # Preserve other useful fields without replacing established values.
        protected_fields = {
            "doc_id",
            "content",
            "metadata",
            "source_task_ids",
            "retrieval_rounds",
        }

        for key, value in item.items():
            if key in protected_fields:
                continue

            if (
                key not in current
                or current[key] in (None, "", [], {})
            ):
                current[key] = value

    return merged


def merge_trace(
    existing: list[dict[str, Any]],
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge trace entries without duplicating cumulative trace updates."""
    merged = [
        _record_to_dict(item)
        for item in existing
    ]

    seen = {
        item.get("event_id") or json.dumps(
            item,
            sort_keys=True,
            default=str,
        )
        for item in merged
    }

    for update in updates:
        item = _record_to_dict(update)

        identity = item.get("event_id") or json.dumps(
            item,
            sort_keys=True,
            default=str,
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

    # Stage 6 structured synthesis output
    answer_synthesis: dict[str, Any]

    # Execution control
    execution_status: str
    execution_limits: dict[str, int]
    followup_round: int
    followup_tasks_created: bool

    # Evidence evaluation and grounding validation
    evidence_assessment: dict[str, Any]
    grounding_result: dict[str, Any]

    # Observability
    trace: Annotated[
        list[dict[str, Any]],
        merge_trace,
    ]