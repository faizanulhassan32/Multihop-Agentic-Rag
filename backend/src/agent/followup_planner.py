from typing import Any, Sequence

from agent.evidence_models import EvidenceAssessment
from agent.task_manager import validate_task_plan
from agent.task_models import TaskSpec


def _normalize_tasks(
    tasks: Sequence[TaskSpec | dict[str, Any]],
) -> list[TaskSpec]:
    return [
        task if isinstance(task, TaskSpec)
        else TaskSpec.model_validate(task)
        for task in tasks
    ]


def _build_targeted_query(
    original: TaskSpec,
    missing_terms: list[str],
) -> str:
    """
    Build a retrieval query focused on the assessed evidence gap.

    Missing terms are diagnostic hints, not a semantic interpretation.
    Keep the original question in the query so retrieval retains context.
    """
    terms = [
        term.strip()
        for term in missing_terms
        if term.strip()
    ][:8]

    if terms:
        return f"{original.question} {' '.join(terms)}".strip()

    # If the assessor cannot identify useful missing terms, retry the
    # original focused retrieval query rather than adding generic filler.
    return original.retrieval_query.strip()


def create_followup_tasks(
    tasks: Sequence[TaskSpec | dict[str, Any]],
    assessment: EvidenceAssessment | dict[str, Any],
    followup_round: int,
    max_followup_rounds: int,
    max_planned_tasks: int,
) -> list[TaskSpec]:
    """Create bounded follow-up retrieval tasks for assessed evidence gaps."""

    if followup_round < 0:
        raise ValueError("followup_round cannot be negative.")
    if max_followup_rounds < 0:
        raise ValueError("max_followup_rounds cannot be negative.")
    if max_planned_tasks < 1:
        raise ValueError("max_planned_tasks must be at least 1.")

    normalized_tasks = _normalize_tasks(tasks)

    normalized_assessment = (
        assessment
        if isinstance(assessment, EvidenceAssessment)
        else EvidenceAssessment.model_validate(assessment)
    )

    if followup_round >= max_followup_rounds:
        return []

    if len(normalized_tasks) >= max_planned_tasks:
        return []

    tasks_by_id = {
        task.task_id: task for task in normalized_tasks
    }
    existing_ids = set(tasks_by_id)

    assessments_by_task = {
        item.task_id: item
        for item in normalized_assessment.task_assessments
    }

    followups: list[TaskSpec] = []

    for gap_task_id in normalized_assessment.gap_task_ids:
        if len(normalized_tasks) + len(followups) >= max_planned_tasks:
            break

        original = tasks_by_id.get(gap_task_id)
        if original is None:
            continue

        next_round = followup_round + 1
        followup_id = f"{original.task_id}_followup_{next_round}"

        if followup_id in existing_ids:
            continue

        task_assessment = assessments_by_task.get(gap_task_id)
        missing_terms = (
            task_assessment.missing_terms
            if task_assessment
            else []
        )

        completed_dependencies = [
            dependency_id
            for dependency_id in original.depends_on
            if dependency_id in tasks_by_id
            and tasks_by_id[dependency_id].status == "completed"
        ]

        followups.append(
            TaskSpec(
                task_id=followup_id,
                question=original.question,
                retrieval_query=_build_targeted_query(
                    original,
                    missing_terms,
                ),
                depends_on=completed_dependencies,
                priority=original.priority,
                followup_round=next_round,
            )
        )
        existing_ids.add(followup_id)

    if not followups:
        return []

    combined_plan = validate_task_plan(
        [*normalized_tasks, *followups],
        max_tasks=max_planned_tasks,
    )

    followup_ids = {task.task_id for task in followups}

    return [
        task for task in combined_plan
        if task.task_id in followup_ids
    ]