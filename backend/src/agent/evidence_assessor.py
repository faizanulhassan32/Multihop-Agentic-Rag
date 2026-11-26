import re
from typing import Any, Sequence

from agent.evidence_models import (
    EvidenceAssessment,
    TaskEvidenceAssessment,
)
from agent.task_models import EvidenceRecord, TaskSpec


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "in",
    "into", "is", "it", "of", "on", "or", "that", "the", "their",
    "them", "there", "these", "this", "those", "to", "was", "were",
    "what", "when", "where", "which", "who", "whom", "why", "with",
}

_ENTITY_EXCLUSIONS = {
    "who", "what", "when", "where", "why", "how",
    "which", "whose", "whom",
}


def _tokens(value: str) -> set[str]:
    """Extract normalized meaningful terms from text."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 1 and token not in _STOP_WORDS
    }


def _extract_required_entities(question: str) -> set[str]:
    """
    Extract likely named entities from a task question.

    This intentionally uses a lightweight heuristic rather than an NER model.
    It recognizes:
    - Internal capitalization: SpaceX, YouTube, iPhone
    - Acronyms: NASA, IBM
    - Capitalized words appearing after the first question word

    The first word is excluded to avoid treating question starters such as
    'Who' or 'What' as entities.
    """
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9]*\b", question)

    entities = set()

    for index, word in enumerate(words):
        if index == 0:
            continue

        if word.lower() in _ENTITY_EXCLUSIONS:
            continue

        is_internal_capitalization = bool(
            re.search(r"[a-z][A-Z]", word)
        )

        is_acronym = (
            len(word) >= 2
            and word.isupper()
        )

        is_capitalized = (
            word[0].isupper()
            and word[1:].islower()
        )

        if (
            is_internal_capitalization
            or is_acronym
            or is_capitalized
        ):
            entities.add(word.lower())

    return entities


def _root_task_id(task_id: str) -> str:
    """Resolve follow-up task IDs to their original task IDs."""
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
    Assess accumulated evidence against the original tasks.

    Sufficiency requires:
    1. Evidence exists.
    2. Required named entities are present.
    3. Lexical coverage meets the configured threshold.

    This is a retrieval-quality heuristic, not semantic entailment.
    """
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError(
            "minimum_coverage must be between 0 and 1."
        )

    normalized_tasks = _normalize_tasks(tasks)
    normalized_evidence = _normalize_evidence(evidence)

    original_tasks = [
        task
        for task in normalized_tasks
        if task.followup_round == 0
    ]

    if not original_tasks:
        return EvidenceAssessment(
            status="insufficient",
            coverage_score=0.0,
            task_assessments=[],
            gap_task_ids=[],
            evidence_count=len(normalized_evidence),
            reason="No original planned tasks were available for assessment.",
        )

    assessments = []
    gap_task_ids = []

    for task in original_tasks:
        root_id = _root_task_id(task.task_id)

        task_evidence = [
            item
            for item in normalized_evidence
            if any(
                _root_task_id(source_task_id) == root_id
                for source_task_id in item.source_task_ids
            )
        ]

        query_terms = _tokens(
            f"{task.question} {task.retrieval_query}"
        )

        required_entities = _extract_required_entities(
            task.question
        )

        evidence_text = " ".join(
            item.content for item in task_evidence
        )

        evidence_terms = _tokens(evidence_text)

        matched_terms = sorted(
            query_terms & evidence_terms
        )

        missing_terms = sorted(
            query_terms - evidence_terms
        )

        matched_entities = sorted(
            required_entities & evidence_terms
        )

        missing_entities = sorted(
            required_entities - evidence_terms
        )

        lexical_coverage = (
            len(matched_terms) / len(query_terms)
            if query_terms
            else 0.0
        )

        entity_coverage = (
            len(matched_entities) / len(required_entities)
            if required_entities
            else 1.0
        )

        # Entity coverage is mandatory when the task contains
        # recognizable named entities.
        entities_satisfied = not missing_entities

        if not task_evidence:
            status = "insufficient"
            reason = "No evidence was retrieved for this task."

        elif not entities_satisfied:
            status = "insufficient"
            reason = (
                "Required named entities are missing from retrieved evidence: "
                + ", ".join(missing_entities)
            )

        elif lexical_coverage < minimum_coverage:
            status = "partial"
            reason = (
                "Evidence contains the required entities but does not meet "
                "the lexical coverage threshold."
            )

        else:
            status = "sufficient"
            reason = (
                "Required entities are present and lexical coverage "
                "meets the configured threshold."
            )

        if status != "sufficient":
            gap_task_ids.append(task.task_id)

        # Entity coverage is incorporated into the score so that
        # missing key entities reduce the score significantly.
        coverage = lexical_coverage * entity_coverage

        assessments.append(
            TaskEvidenceAssessment(
                task_id=task.task_id,
                status=status,
                coverage_score=coverage,
                matched_terms=matched_terms,
                missing_terms=missing_terms,
                evidence_ids=[
                    item.doc_id for item in task_evidence
                ],
                reason=reason,
            )
        )

    overall_coverage = sum(
        item.coverage_score for item in assessments
    ) / len(assessments)

    sufficient_count = sum(
        item.status == "sufficient"
        for item in assessments
    )

    if sufficient_count == len(assessments) and normalized_evidence:
        overall_status = "sufficient"
        reason = (
            "All original tasks meet entity and lexical coverage requirements."
        )

    elif sufficient_count > 0 or any(
        item.coverage_score > 0
        for item in assessments
    ):
        overall_status = "partial"
        reason = (
            "Some evidence coverage exists, but one or more task gaps remain."
        )

    else:
        overall_status = "insufficient"
        reason = (
            "Retrieved evidence does not adequately cover the original tasks."
        )

    return EvidenceAssessment(
        status=overall_status,
        coverage_score=overall_coverage,
        task_assessments=assessments,
        gap_task_ids=gap_task_ids,
        evidence_count=len(normalized_evidence),
        assessment_method="entity_aware_lexical_coverage",
        reason=reason,
    )