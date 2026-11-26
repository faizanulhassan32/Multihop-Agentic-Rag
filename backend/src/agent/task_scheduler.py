
from typing import Sequence

from config import Config
from agent.task_manager import (
    get_blocked_tasks,
    get_ready_tasks,
    validate_task_plan,
)
from agent.task_models import TaskSpec


TERMINAL_STATUSES = {"completed", "failed", "skipped"}


def _validated_tasks(
    tasks: Sequence[TaskSpec],
) -> list[TaskSpec]:
    """Normalize and validate a task collection."""

    return validate_task_plan(tasks)


def _replace_task(
    task: TaskSpec,
    **updates,
) -> TaskSpec:
    """Return a validated copy of a task with updated fields."""

    task_data = task.model_dump()
    task_data.update(updates)

    return TaskSpec.model_validate(task_data)


def get_next_execution_batch(
    tasks: Sequence[TaskSpec],
    max_concurrency: int | None = None,
) -> list[TaskSpec]:
    """
    Select the next batch of tasks that can execute.

    Tasks are ordered by priority, with lower values representing
    higher priority. Existing running tasks consume concurrency slots.

    This function does not change task statuses.
    """

    validated = _validated_tasks(tasks)

    concurrency_limit = (
        Config.MAX_RETRIEVAL_CONCURRENCY
        if max_concurrency is None
        else max_concurrency
    )

    if concurrency_limit < 1:
        raise ValueError("max_concurrency must be at least 1.")

    running_count = sum(
        task.status == "running"
        for task in validated
    )

    available_slots = concurrency_limit - running_count

    if available_slots <= 0:
        return []

    ready_tasks = get_ready_tasks(validated)

    original_order = {
        task.task_id: index
        for index, task in enumerate(validated)
    }

    ready_tasks.sort(
        key=lambda task: (
            task.priority,
            original_order[task.task_id],
        )
    )

    return ready_tasks[:available_slots]


def mark_tasks_running(
    tasks: Sequence[TaskSpec],
    task_ids: list[str],
    max_concurrency: int | None = None,
) -> list[TaskSpec]:
    """
    Mark selected ready tasks as running.

    Only tasks returned by get_next_execution_batch can be started.
    The attempt counter increments when a task starts execution.
    """

    validated = _validated_tasks(tasks)

    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate task IDs cannot be started.")

    if not task_ids:
        return validated

    allowed_batch = get_next_execution_batch(
        validated,
        max_concurrency=max_concurrency,
    )

    allowed_ids = {
        task.task_id
        for task in allowed_batch
    }

    invalid_ids = [
        task_id
        for task_id in task_ids
        if task_id not in allowed_ids
    ]

    if invalid_ids:
        raise ValueError(
            "Tasks are not currently eligible to start: "
            f"{invalid_ids}"
        )

    selected_ids = set(task_ids)

    return [
        _replace_task(
            task,
            status="running",
            attempts=task.attempts + 1,
            error=None,
        )
        if task.task_id in selected_ids
        else task
        for task in validated
    ]


def complete_task(
    tasks: Sequence[TaskSpec],
    task_id: str,
    evidence_ids: list[str] | None = None,
) -> list[TaskSpec]:
    """Mark a running task as completed."""

    validated = _validated_tasks(tasks)

    matching_task = next(
        (task for task in validated if task.task_id == task_id),
        None,
    )

    if matching_task is None:
        raise ValueError(f"Unknown task ID: {task_id}")

    if matching_task.status != "running":
        raise ValueError(
            f"Task '{task_id}' must be running before it can complete."
        )

    updated_evidence_ids = list(
        dict.fromkeys(evidence_ids or [])
    )

    return [
        _replace_task(
            task,
            status="completed",
            evidence_ids=updated_evidence_ids,
            error=None,
        )
        if task.task_id == task_id
        else task
        for task in validated
    ]


def fail_task(
    tasks: Sequence[TaskSpec],
    task_id: str,
    error: str,
) -> list[TaskSpec]:
    """Mark a running task as failed."""

    validated = _validated_tasks(tasks)

    matching_task = next(
        (task for task in validated if task.task_id == task_id),
        None,
    )

    if matching_task is None:
        raise ValueError(f"Unknown task ID: {task_id}")

    if matching_task.status != "running":
        raise ValueError(
            f"Task '{task_id}' must be running before it can fail."
        )

    if not isinstance(error, str) or not error.strip():
        raise ValueError("A non-empty error message is required.")

    return [
        _replace_task(
            task,
            status="failed",
            error=error.strip(),
        )
        if task.task_id == task_id
        else task
        for task in validated
    ]


def skip_blocked_tasks(
    tasks: Sequence[TaskSpec],
) -> list[TaskSpec]:
    """
    Mark tasks as skipped when one or more dependencies failed or
    were skipped.

    Repeats until no additional tasks become blocked, so skipped
    dependencies are propagated through the dependency graph.
    """

    updated = _validated_tasks(tasks)

    while True:
        blocked_tasks = get_blocked_tasks(updated)

        if not blocked_tasks:
            break

        blocked_ids = {
            task.task_id
            for task in blocked_tasks
        }

        updated = [
            _replace_task(
                task,
                status="skipped",
                error="A required dependency failed or was skipped.",
            )
            if task.task_id in blocked_ids
            else task
            for task in updated
        ]

    return updated


def get_execution_status(
    tasks: Sequence[TaskSpec],
) -> str:
    """
    Return an overall execution status.

    Possible results:
    - completed: every task completed successfully
    - failed: at least one task failed
    - blocked: pending tasks remain but none can execute
    - running: at least one task is currently running
    - ready: at least one task is ready to execute
    """

    validated = _validated_tasks(tasks)

    if all(task.status == "completed" for task in validated):
        return "completed"

    if any(task.status == "failed" for task in validated):
        return "failed"

    if any(task.status == "running" for task in validated):
        return "running"

    if get_next_execution_batch(validated):
        return "ready"

    if any(task.status == "pending" for task in validated):
        return "blocked"

    if all(task.status in TERMINAL_STATUSES for task in validated):
        return "completed"

    return "blocked"
