from collections import deque
from typing import Any, Mapping, Sequence
from config import Config
from agent.models import TaskSpec

TERMINAL_STATUSES = {"completed", "failed", "skipped"}


# ============================================================
# Validation and dependency-aware queries
# ============================================================

def _normalize_tasks(
    tasks: Sequence[TaskSpec | Mapping[str, Any]],
) -> list[TaskSpec]:
    """Convert task dictionaries into validated TaskSpec instances."""

    normalized: list[TaskSpec] = []

    for task in tasks:
        if isinstance(task, TaskSpec):
            normalized.append(task)
        elif isinstance(task, Mapping):
            normalized.append(TaskSpec.model_validate(task))
        else:
            raise TypeError(
                "Every task must be a TaskSpec or a mapping. "
                f"Received: {type(task).__name__}"
            )

    return normalized


def validate_task_plan(
    tasks: Sequence[TaskSpec | Mapping[str, Any]],
    max_tasks: int | None = None,
) -> list[TaskSpec]:
    """
    Validate a task plan and return normalized TaskSpec objects.

    Checks:
    - Plan is not empty.
    - Maximum task count is respected.
    - Task IDs are unique.
    - Dependencies exist.
    - Tasks do not depend on themselves.
    - Dependencies are not duplicated.
    - Dependency graph contains no cycles.
    """

    normalized = _normalize_tasks(tasks)

    if not normalized:
        raise ValueError("A task plan cannot be empty.")

    task_limit = (
        Config.MAX_PLANNED_TASKS if max_tasks is None else max_tasks
    )

    if task_limit < 1:
        raise ValueError("max_tasks must be at least 1.")

    if len(normalized) > task_limit:
        raise ValueError(
            f"Task plan contains {len(normalized)} tasks, "
            f"exceeding the maximum allowed limit of {task_limit}."
        )

    task_ids = [task.task_id for task in normalized]
    task_id_set = set(task_ids)

    if len(task_ids) != len(task_id_set):
        duplicates = sorted(
            task_id
            for task_id in task_id_set
            if task_ids.count(task_id) > 1
        )
        raise ValueError(
            f"Duplicate task IDs found: {duplicates}"
        )

    for task in normalized:
        if task.task_id in task.depends_on:
            raise ValueError(
                f"Task '{task.task_id}' cannot depend on itself."
            )

        if len(task.depends_on) != len(set(task.depends_on)):
            raise ValueError(
                f"Task '{task.task_id}' contains duplicate dependencies."
            )

        missing_dependencies = [
            dependency
            for dependency in task.depends_on
            if dependency not in task_id_set
        ]

        if missing_dependencies:
            raise ValueError(
                f"Task '{task.task_id}' references unknown dependencies: "
                f"{missing_dependencies}"
            )

    _topological_sort(normalized)

    return normalized


def _topological_sort(tasks: Sequence[TaskSpec]) -> list[str]:
    """
    Return task IDs in dependency-safe order.

    Uses Kahn's algorithm.
    Raises ValueError if the dependency graph contains a cycle.
    """

    task_by_id = {task.task_id: task for task in tasks}

    in_degree = {
        task.task_id: len(task.depends_on)
        for task in tasks
    }

    dependents: dict[str, list[str]] = {
        task.task_id: []
        for task in tasks
    }

    for task in tasks:
        for dependency_id in task.depends_on:
            dependents[dependency_id].append(task.task_id)

    ready_queue = deque(
        task_id
        for task_id, degree in in_degree.items()
        if degree == 0
    )

    ordered_ids: list[str] = []

    while ready_queue:
        current_id = ready_queue.popleft()
        ordered_ids.append(current_id)

        for dependent_id in dependents[current_id]:
            in_degree[dependent_id] -= 1

            if in_degree[dependent_id] == 0:
                ready_queue.append(dependent_id)

    if len(ordered_ids) != len(task_by_id):
        cyclic_tasks = sorted(
            task_id
            for task_id, degree in in_degree.items()
            if degree > 0
        )

        raise ValueError(
            "Circular dependencies detected involving tasks: "
            f"{cyclic_tasks}"
        )

    return ordered_ids


def get_ready_tasks(
    tasks: Sequence[TaskSpec | Mapping[str, Any]],
) -> list[TaskSpec]:
    """
    Return pending tasks whose dependencies are all completed.

    Independent tasks can be returned together for parallel retrieval.
    This function does not execute tasks or change their status.
    """

    validated_tasks = validate_task_plan(tasks)

    task_statuses = {
        task.task_id: task.status
        for task in validated_tasks
    }

    return [
        task
        for task in validated_tasks
        if task.status == "pending"
        and all(
            task_statuses[dependency_id] == "completed"
            for dependency_id in task.depends_on
        )
    ]


def get_blocked_tasks(
    tasks: Sequence[TaskSpec | Mapping[str, Any]],
) -> list[TaskSpec]:
    """
    Return pending tasks with at least one failed or skipped dependency.

    These tasks cannot proceed until their dependency situation is resolved.
    """

    validated_tasks = validate_task_plan(tasks)

    task_statuses = {
        task.task_id: task.status
        for task in validated_tasks
    }

    terminal_blocking_statuses = {"failed", "skipped"}

    return [
        task
        for task in validated_tasks
        if task.status == "pending"
        and any(
            task_statuses[dependency_id] in terminal_blocking_statuses
            for dependency_id in task.depends_on
        )
    ]


# ============================================================
# Scheduler
# ============================================================

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