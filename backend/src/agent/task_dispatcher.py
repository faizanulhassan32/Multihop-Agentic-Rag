import logging
from typing import Any, TypedDict
from langgraph.types import Send
from config import Config
from agent.models import TaskResult, TaskSpec
from agent.retrieval import execute_retrieval_task
from agent.task_manager import (
    complete_task,
    fail_task,
    get_execution_status,
    get_next_execution_batch,
    mark_tasks_running,
    skip_blocked_tasks,
)

logger = logging.getLogger(__name__)


class RetrievalWorkerState(TypedDict, total=False):
    task: dict[str, Any]
    request_id: str


def _load_tasks(raw_tasks: list[Any]) -> list[TaskSpec]:
    """Convert serialized task dictionaries into task models."""
    return [
        task if isinstance(task, TaskSpec)
        else TaskSpec.model_validate(task)
        for task in raw_tasks
    ]


def _serialize_tasks(tasks: list[TaskSpec]) -> list[dict[str, Any]]:
    """Convert task models into graph-compatible dictionaries."""
    return [task.model_dump() for task in tasks]


def prepare_retrieval_batch(state: dict[str, Any]) -> dict[str, Any]:
    """Select and mark the next dependency-ready retrieval batch."""

    tasks = _load_tasks(state.get("tasks", []))
    tasks = skip_blocked_tasks(tasks)

    batch = get_next_execution_batch(
        tasks,
        max_concurrency=Config.MAX_RETRIEVAL_CONCURRENCY,
    )

    if not batch:
        return {
            "tasks": _serialize_tasks(tasks),
            "execution_status": get_execution_status(tasks),
        }

    batch_ids = [task.task_id for task in batch]

    updated_tasks = mark_tasks_running(
        tasks,
        task_ids=batch_ids,
        max_concurrency=Config.MAX_RETRIEVAL_CONCURRENCY,
    )

    logger.info(
        "Prepared retrieval batch: task_ids=%s concurrency=%d",
        batch_ids,
        Config.MAX_RETRIEVAL_CONCURRENCY,
    )

    return {
        "tasks": _serialize_tasks(updated_tasks),
        "execution_status": get_execution_status(updated_tasks),
    }


def dispatch_retrieval_tasks(
    state: dict[str, Any],
) -> list[Send] | str:
    """Dispatch all currently running tasks as parallel LangGraph workers."""

    tasks = _load_tasks(state.get("tasks", []))

    running_tasks = [
        task for task in tasks
        if task.status == "running"
    ]

    if not running_tasks:
        return "no_tasks_to_dispatch"

    request_id = state.get("request_id", "unknown")

    return [
        Send(
            "execute_retrieval_task_node",
            {
                "task": task.model_dump(),
                "request_id": request_id,
            },
        )
        for task in running_tasks
    ]


async def execute_retrieval_task_node(
    state: RetrievalWorkerState,
) -> dict[str, Any]:
    """Execute one retrieval task and return reducer-compatible updates."""

    task = TaskSpec.model_validate(state["task"])
    request_id = state.get("request_id", "unknown")

    result, evidence = await execute_retrieval_task(
        task=task,
        request_id=request_id,
    )

    return {
        "task_results": [result.model_dump()],
        "evidence": [
            item.model_dump()
            for item in evidence
        ],
    }


def collect_retrieval_results(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Apply worker results to the task plan and propagate failures."""

    tasks = _load_tasks(state.get("tasks", []))
    raw_results = state.get("task_results", [])

    results_by_id = {
        result["task_id"]: TaskResult.model_validate(result)
        for result in raw_results
    }

    updated_tasks = list(tasks)

    for task in tasks:
        if task.status != "running":
            continue

        result = results_by_id.get(task.task_id)

        if result is None or result.attempts != task.attempts:
            updated_tasks = fail_task(
                updated_tasks,
                task_id=task.task_id,
                error="Retrieval worker did not return a result for this attempt.",
            )
            continue

        if result.status == "completed":
            updated_tasks = complete_task(
                updated_tasks,
                task_id=task.task_id,
                evidence_ids=result.evidence_ids,
            )
        else:
            updated_tasks = fail_task(
                updated_tasks,
                task_id=task.task_id,
                error=result.error or "Retrieval task failed.",
            )

    updated_tasks = skip_blocked_tasks(updated_tasks)

    status = get_execution_status(updated_tasks)

    logger.info(
        "Collected retrieval results: status=%s completed=%d failed=%d",
        status,
        sum(task.status == "completed" for task in updated_tasks),
        sum(task.status == "failed" for task in updated_tasks),
    )

    return {
        "tasks": _serialize_tasks(updated_tasks),
        "execution_status": status,
    }