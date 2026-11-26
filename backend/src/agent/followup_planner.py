from typing import Any, Sequence

from agent.evidence_models import EvidenceAssessment
from agent.task_manager import validate_task_plan
from agent.task_models import TaskSpec


def create_followup_tasks(
    tasks: Sequence[TaskSpec | dict[str, Any]],
    assessment: EvidenceAssessment | dict[str, Any],
    followup_round: int,
    max_followup_rounds: int,
    max_planned_tasks: int,
) -> list[TaskSpec]:
    """
    Create targeted retrieval tasks for identified evidence gaps.

    Follow-up tasks preserve completed prerequisites but do not depend on
    failed or skipped tasks. This allows retrieval to retry the missing
    information without being blocked by the original task's failure.
    """

    if followup_round < 0:
        raise ValueError("followup_round cannot be negative.")

    if max_followup_rounds < 0:
        raise ValueError("max_followup_rounds cannot be negative.")

    if max_planned_tasks < 1:
        raise ValueError("max_planned_tasks must be at least 1.")

    normalized_tasks = [
        task if isinstance(task, TaskSpec)
        else TaskSpec.model_validate(task)
        for task in tasks
    ]

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
        task.task_id: task
        for task in normalized_tasks
    }

    existing_ids = set(tasks_by_id)
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

        # Keep only dependencies that completed successfully.
        completed_dependencies = [
            dependency_id
            for dependency_id in original.depends_on
            if dependency_id in tasks_by_id
            and tasks_by_id[dependency_id].status == "completed"
        ]

        targeted_query = (
            f"{original.retrieval_query} "
            f"{original.question} "
            "additional factual evidence"
        ).strip()

        followup = TaskSpec(
            task_id=followup_id,
            question=original.question,
            retrieval_query=targeted_query,
            depends_on=completed_dependencies,
            priority=original.priority,
            followup_round=next_round,
        )

        followups.append(followup)
        existing_ids.add(followup_id)

    if not followups:
        return []

    combined_plan = validate_task_plan(
        [*normalized_tasks, *followups],
        max_tasks=max_planned_tasks,
    )

    return [
        task
        for task in combined_plan
        if task.task_id in {item.task_id for item in followups}
    ]