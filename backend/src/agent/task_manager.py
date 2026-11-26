
from collections import deque
from typing import Any, Mapping, Sequence

from config import Config

from agent.task_models import TaskSpec


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


def get_execution_order(
    tasks: Sequence[TaskSpec | Mapping[str, Any]],
) -> list[str]:
    """Return task IDs in a valid dependency execution order."""

    validated_tasks = validate_task_plan(tasks)
    return _topological_sort(validated_tasks)


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
