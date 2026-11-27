import re
from typing import Any, Sequence

from agent.evidence_models import EvidenceAssessment, TaskEvidenceAssessment
from agent.task_models import EvidenceRecord, TaskSpec


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "in",
    "into", "is", "it", "of", "on", "or", "that", "the", "their",
    "them", "there", "these", "this", "those", "to", "was", "were",
    "what", "when", "where", "which", "who", "whom", "why", "with",
}


def _tokens(value: str) -> set[str]:
    """Return normalized, meaningful terms."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 1 and token not in _STOP_WORDS
    }


def _root_task_id(task_id: str) -> str:
    """Map follow-up task IDs back to their original task."""
    return re.sub(r"(?:_followup_\d+)+$", "", task_id)


def _normalize_tasks(
    tasks: Sequence[TaskSpec | dict[str, Any]],
) -> list[TaskSpec]:
    return [
        task if isinstance(task, TaskSpec)
        else TaskSpec.model_validate(task)
        for task in tasks
    ]


def _normalize_evidence(
    evidence: Sequence[EvidenceRecord | dict[str, Any]],
) -> list[EvidenceRecord]:
    return [
        item if isinstance(item, EvidenceRecord)
        else EvidenceRecord.model_validate(item)
        for item in evidence
    ]


def assess_evidence(
    tasks: Sequence[TaskSpec | dict[str, Any]],
    evidence: Sequence[EvidenceRecord | dict[str, Any]],
    minimum_coverage: float = 0.35,
) -> EvidenceAssessment:
    """
    Estimate task-level retrieval coverage.

    This is a lightweight retrieval heuristic, not semantic entailment.
    Named-entity detection is deliberately not used as a mandatory gate:
    capitalization-based entity extraction is unreliable across domains.

    Evidence from follow-up tasks is attributed to the corresponding
    original task using the root task ID.
    """
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be between 0 and 1.")

    normalized_tasks = _normalize_tasks(tasks)
    normalized_evidence = _normalize_evidence(evidence)

    original_tasks = [
        task for task in normalized_tasks
        if task.followup_round == 0
    ]

    if not original_tasks:
        return EvidenceAssessment(
            status="insufficient",
            coverage_score=0.0,
            task_assessments=[],
            gap_task_ids=[],
            evidence_count=len(normalized_evidence),
            assessment_method="lexical_coverage",
            reason="No original planned tasks were available for assessment.",
        )

    assessments: list[TaskEvidenceAssessment] = []
    gap_task_ids: list[str] = []

    for task in original_tasks:
        root_id = _root_task_id(task.task_id)

        task_evidence = [
            item
            for item in normalized_evidence
            if any(
                _root_task_id(source_id) == root_id
                for source_id in item.source_task_ids
            )
        ]

        query_terms = _tokens(
            f"{task.question} {task.retrieval_query}"
        )
        evidence_terms = _tokens(
            " ".join(item.content for item in task_evidence)
        )

        matched_terms = sorted(query_terms & evidence_terms)
        missing_terms = sorted(query_terms - evidence_terms)

        coverage = (
            len(matched_terms) / len(query_terms)
            if query_terms
            else (1.0 if task_evidence else 0.0)
        )

        if not task_evidence:
            status = "insufficient"
            reason = "No evidence was retrieved for this task."
        elif coverage == 0.0:
            status = "insufficient"
            reason = "Retrieved evidence has no lexical overlap with the task."
        elif coverage < minimum_coverage:
            status = "partial"
            reason = (
                "Some query terms appear in the evidence, but coverage "
                "is below the configured threshold."
            )
        else:
            status = "sufficient"
            reason = (
                "Retrieved evidence meets the configured lexical "
                "coverage threshold."
            )

        if status != "sufficient":
            gap_task_ids.append(task.task_id)

        assessments.append(
            TaskEvidenceAssessment(
                task_id=task.task_id,
                status=status,
                coverage_score=coverage,
                matched_terms=matched_terms,
                missing_terms=missing_terms,
                evidence_ids=[item.doc_id for item in task_evidence],
                reason=reason,
            )
        )

    overall_coverage = sum(
        item.coverage_score for item in assessments
    ) / len(assessments)

    sufficient_count = sum(
        item.status == "sufficient" for item in assessments
    )

    if sufficient_count == len(assessments):
        overall_status = "sufficient"
        reason = "All original tasks meet the lexical coverage threshold."
    elif sufficient_count > 0 or overall_coverage > 0:
        overall_status = "partial"
        reason = "Some task evidence exists, but one or more gaps remain."
    else:
        overall_status = "insufficient"
        reason = "Retrieved evidence does not cover the original tasks."

    return EvidenceAssessment(
        status=overall_status,
        coverage_score=overall_coverage,
        task_assessments=assessments,
        gap_task_ids=gap_task_ids,
        evidence_count=len(normalized_evidence),
        assessment_method="lexical_coverage",
        reason=reason,
    )