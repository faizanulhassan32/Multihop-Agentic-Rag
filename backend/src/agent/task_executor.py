import logging
import time

from langchain_core.documents import Document

from agent.task_models import (
    EvidenceRecord,
    TaskResult,
    TaskSpec,
)

from agent.rag_system import rag_system


logger = logging.getLogger(__name__)


def _document_to_evidence(
    document: Document,
    task_id: str,
    retrieval_round: int = 0,
) -> EvidenceRecord:
    """
    Convert a retrieved document into structured evidence.

    Preserve stable document identity, metadata, originating task,
    and retrieval round.
    """

    document_id = document.metadata.get("id")

    if (
        not isinstance(document_id, str)
        or not document_id.strip()
    ):
        raise ValueError(
            "Retrieved document is missing a stable string ID."
        )

    return EvidenceRecord(
        doc_id=document_id,
        content=document.page_content,
        metadata=dict(document.metadata),
        source_task_ids=[task_id],
        retrieval_rounds=[retrieval_round],
    )


async def execute_retrieval_task(
    task: TaskSpec,
    request_id: str = "unknown",
) -> tuple[TaskResult, list[EvidenceRecord]]:
    """
    Execute one retrieval task.

    A task with no retrieved evidence is considered failed so that
    the orchestrator can distinguish an unsuccessful retrieval from
    a successful retrieval that produced supporting evidence.
    """

    started_at = time.perf_counter()

    try:
        documents = await rag_system.retrieve_documents(
            query=task.retrieval_query,
            request_id=request_id,
        )

        evidence = [
            _document_to_evidence(
                document=document,
                task_id=task.task_id,
                retrieval_round=task.followup_round,
            )
            for document in documents
        ]

        elapsed_ms = (
            time.perf_counter() - started_at
        ) * 1000

        if not evidence:

            result = TaskResult(
                task_id=task.task_id,
                status="failed",
                evidence_ids=[],
                attempts=task.attempts,
                error="Retrieval returned no documents.",
                latency_ms=elapsed_ms,
            )

            logger.warning(
                "Retrieval task returned no evidence: task_id=%s",
                task.task_id,
            )

            return result, []

        result = TaskResult(
            task_id=task.task_id,
            status="completed",
            evidence_ids=[
                item.doc_id
                for item in evidence
            ],
            attempts=task.attempts,
            latency_ms=elapsed_ms,
        )

        logger.info(
            "Retrieval task completed: task_id=%s "
            "documents=%d elapsed_ms=%.2f",
            task.task_id,
            len(evidence),
            elapsed_ms,
        )

        return result, evidence

    except Exception as exc:

        elapsed_ms = (
            time.perf_counter() - started_at
        ) * 1000

        logger.exception(
            "Retrieval task failed: task_id=%s",
            task.task_id,
        )

        result = TaskResult(
            task_id=task.task_id,
            status="failed",
            evidence_ids=[],
            attempts=task.attempts,
            error=f"{type(exc).__name__}: {exc}",
            latency_ms=elapsed_ms,
        )

        return result, []