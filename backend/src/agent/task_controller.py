from typing import Sequence

from config import Config
from agent.planner import TaskPlanner
from agent.task_manager import validate_task_plan
from agent.task_models import TaskSpec
from agent.task_scheduler import (
    complete_task,
    fail_task,
    get_execution_status,
    get_next_execution_batch,
    mark_tasks_running,
    skip_blocked_tasks,
)


class TaskController:
    """
    Coordinates planning, dependency-aware scheduling, and bounded
    follow-up task insertion.

    This controller manages task state transitions but does not execute
    retrieval operations itself.
    """

    def __init__(
        self,
        planner: TaskPlanner | None = None,
        max_concurrency: int | None = None,
        max_tasks: int | None = None,
        max_followup_rounds: int | None = None,
    ) -> None:
        self.planner = planner or TaskPlanner()
        self.max_concurrency = max_concurrency
        self.max_tasks = max_tasks or Config.MAX_PLANNED_TASKS
        self.max_followup_rounds = (
            Config.MAX_FOLLOWUP_ROUNDS
            if max_followup_rounds is None
            else max_followup_rounds
        )
        self.tasks: list[TaskSpec] = []
        self.followup_round = 0

    def create_plan(self, question: str) -> list[TaskSpec]:
        """Generate and validate a task plan for a user question."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        planned_tasks = self.planner.plan(question.strip())

        self.tasks = validate_task_plan(
            planned_tasks,
            max_tasks=self.max_tasks,
        )

        self.followup_round = 0
        return self.get_tasks()

    def load_plan(
        self,
        tasks: Sequence[TaskSpec],
    ) -> list[TaskSpec]:
        """Load an existing task plan into the controller."""

        self.tasks = validate_task_plan(
            tasks,
            max_tasks=self.max_tasks,
        )

        self.followup_round = max(
            (task.followup_round for task in self.tasks),
            default=0,
        )

        return self.get_tasks()

    def add_followup_tasks(
        self,
        followup_tasks: Sequence[TaskSpec],
        followup_round: int,
    ) -> list[TaskSpec]:
        """Append validated follow-up tasks within configured limits."""

        if not self.tasks:
            raise ValueError("No task plan has been loaded.")

        if followup_round > self.max_followup_rounds:
            raise ValueError("Maximum follow-up rounds exceeded.")

        if followup_round < self.followup_round:
            raise ValueError("Follow-up round cannot move backwards.")

        if not followup_tasks:
            return self.get_tasks()

        combined = [
            *self.tasks,
            *followup_tasks,
        ]

        self.tasks = validate_task_plan(
            combined,
            max_tasks=self.max_tasks,
        )

        self.followup_round = followup_round

        return self.get_tasks()

    def get_tasks(self) -> list[TaskSpec]:
        """Return a copy of the current task collection."""

        return [
            TaskSpec.model_validate(task.model_dump())
            for task in self.tasks
        ]

    def get_execution_batch(self) -> list[TaskSpec]:
        """Get the next dependency-safe batch of tasks."""

        if not self.tasks:
            raise ValueError("No task plan has been loaded.")

        return get_next_execution_batch(
            self.tasks,
            max_concurrency=self.max_concurrency,
        )

    def start_tasks(self, task_ids: list[str]) -> list[TaskSpec]:
        """Mark an eligible batch of tasks as running."""

        if not self.tasks:
            raise ValueError("No task plan has been loaded.")

        self.tasks = mark_tasks_running(
            self.tasks,
            task_ids,
            max_concurrency=self.max_concurrency,
        )

        return self.get_tasks()

    def record_success(
        self,
        task_id: str,
        evidence_ids: list[str] | None = None,
    ) -> list[TaskSpec]:
        """Record successful completion of a running task."""

        if not self.tasks:
            raise ValueError("No task plan has been loaded.")

        self.tasks = complete_task(
            self.tasks,
            task_id,
            evidence_ids=evidence_ids,
        )

        return self.get_tasks()

    def record_failure(
        self,
        task_id: str,
        error: str,
    ) -> list[TaskSpec]:
        """Record failure and propagate blocked dependencies."""

        if not self.tasks:
            raise ValueError("No task plan has been loaded.")

        self.tasks = fail_task(
            self.tasks,
            task_id,
            error,
        )

        self.tasks = skip_blocked_tasks(self.tasks)

        return self.get_tasks()

    def get_status(self) -> str:
        """Return the overall task execution status."""

        if not self.tasks:
            return "not_started"

        return get_execution_status(self.tasks)

    def is_finished(self) -> bool:
        """Return whether no more task execution is possible."""

        return self.get_status() in {"completed", "failed", "blocked"}