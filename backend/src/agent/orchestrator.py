import logging
import time
from datetime import datetime
from typing import Any

from langchain_core.runnables import RunnableConfig

from config import Config
from agent.planner import create_task_plan
from agent.task_models import TaskSpec
from agent.task_scheduler import get_execution_status
from agent.evidence_assessor import assess_evidence
from agent.followup_planner import create_followup_tasks

logger = logging.getLogger(__name__)


def _request_id(config: RunnableConfig) -> str:
    metadata = config.get("metadata") or {}
    return str(metadata.get("request_id", "unknown"))


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


async def initialize_task_execution(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    """Generate a structured task plan for the original question."""

    question = state.get("original_question", "").strip()

    if not question:
        raise ValueError("Cannot plan tasks for an empty question.")

    request_id = _request_id(config)
    started_at = time.perf_counter()

    logger.info(
        "Task planning started: request_id=%s",
        request_id,
    )

    tasks = create_task_plan(question)

    validated_tasks = [
        task if isinstance(task, TaskSpec)
        else TaskSpec.model_validate(task)
        for task in tasks
    ]

    if not validated_tasks:
        raise ValueError("Planner returned an empty task plan.")

    if len(validated_tasks) > Config.MAX_PLANNED_TASKS:
        raise ValueError(
            f"Planner returned {len(validated_tasks)} tasks, "
            f"exceeding the configured limit of {Config.MAX_PLANNED_TASKS}."
        )

    elapsed_ms = (time.perf_counter() - started_at) * 1000

    logger.info(
        "Task planning completed: request_id=%s tasks=%d elapsed_ms=%.2f",
        request_id,
        len(validated_tasks),
        elapsed_ms,
    )

    return {
        "request_id": request_id,
        "tasks": [
            task.model_dump()
            for task in validated_tasks
        ],
        "task_results": [],
        "evidence": [],
        "execution_status": get_execution_status(validated_tasks),
        "execution_limits": {
            "max_planned_tasks": Config.MAX_PLANNED_TASKS,
            "max_followup_rounds": Config.MAX_FOLLOWUP_ROUNDS,
            "max_retrieval_concurrency": Config.MAX_RETRIEVAL_CONCURRENCY,
            "max_retrieval_retries": Config.MAX_RETRIEVAL_RETRIES,
        },
        "followup_round": 0,
        "trace": [
            {
                "step": "initialize_task_execution",
                "timestamp": _timestamp(),
                "input": {
                    "original_question": question,
                },
                "output": {
                    "task_count": len(validated_tasks),
                    "task_ids": [
                        task.task_id for task in validated_tasks
                    ],
                },
            }
        ],
    }


def assess_current_evidence(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Assess accumulated evidence against the current task plan."""

    assessment = assess_evidence(
        tasks=state.get("tasks", []),
        evidence=state.get("evidence", []),
    )

    logger.info(
        "Evidence assessment completed: status=%s coverage=%.3f "
        "evidence_count=%d gaps=%d",
        assessment.status,
        assessment.coverage_score,
        assessment.evidence_count,
        len(assessment.gap_task_ids),
    )

    return {
        "evidence_assessment": assessment.model_dump(),
        "trace": [
            {
                "step": "assess_current_evidence",
                "timestamp": _timestamp(),
                "output": assessment.model_dump(),
            }
        ],
    }


def plan_followup_execution(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Create targeted follow-up tasks if evidence gaps remain.

    The total task count and follow-up rounds are strictly bounded.
    """

    raw_tasks = state.get("tasks", [])
    raw_assessment = state.get("evidence_assessment")

    if not raw_assessment:
        return {
            "tasks": raw_tasks,
            "execution_status": state.get("execution_status", "not_started"),
        }

    current_round = int(state.get("followup_round", 0))

    followup_tasks = create_followup_tasks(
        tasks=raw_tasks,
        assessment=raw_assessment,
        followup_round=current_round,
        max_followup_rounds=Config.MAX_FOLLOWUP_ROUNDS,
        max_planned_tasks=Config.MAX_PLANNED_TASKS,
    )

    if not followup_tasks:
        logger.info(
            "No follow-up tasks created: round=%d max_rounds=%d",
            current_round,
            Config.MAX_FOLLOWUP_ROUNDS,
        )

        return {
            "tasks": raw_tasks,
            "execution_status": get_execution_status([
                TaskSpec.model_validate(task)
                for task in raw_tasks
            ]),
        }

    updated_tasks = [
        TaskSpec.model_validate(task)
        for task in raw_tasks
    ] + followup_tasks

    logger.info(
        "Follow-up tasks created: round=%d count=%d total_tasks=%d",
        current_round + 1,
        len(followup_tasks),
        len(updated_tasks),
    )

    return {
        "tasks": [
            task.model_dump()
            for task in updated_tasks
        ],
        "followup_round": current_round + 1,
        "execution_status": get_execution_status(updated_tasks),
        "trace": [
            {
                "step": "plan_followup_execution",
                "timestamp": _timestamp(),
                "output": {
                    "followup_round": current_round + 1,
                    "created_task_ids": [
                        task.task_id for task in followup_tasks
                    ],
                },
            }
        ],
    }


def route_after_retrieval(
    state: dict[str, Any],
) -> str:
    """Route completed retrieval batches to evidence assessment."""

    raw_tasks = state.get("tasks", [])

    if not raw_tasks:
        return "assess_current_evidence"

    tasks = [
        task if isinstance(task, TaskSpec)
        else TaskSpec.model_validate(task)
        for task in raw_tasks
    ]

    terminal_statuses = {"completed", "failed", "skipped"}

    if all(task.status in terminal_statuses for task in tasks):
        return "assess_current_evidence"

    return "prepare_retrieval_batch"


def route_after_assessment(
    state: dict[str, Any],
) -> str:
    """Choose follow-up retrieval or final answer synthesis."""

    assessment = state.get("evidence_assessment") or {}

    if assessment.get("status") == "sufficient":
        return "generate_final_answer"

    current_round = int(state.get("followup_round", 0))
    task_count = len(state.get("tasks", []))
    gap_task_ids = assessment.get("gap_task_ids", [])

    if (
        gap_task_ids
        and current_round < Config.MAX_FOLLOWUP_ROUNDS
        and task_count < Config.MAX_PLANNED_TASKS
    ):
        return "plan_followup_execution"

    return "generate_final_answer"