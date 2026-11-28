import json
import logging
from typing import Any, Sequence

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError

from config import Config
from agent.task_manager import validate_task_plan
from agent.models import (
    EvidenceAssessment,
    TaskSpec,
)

logger = logging.getLogger(__name__)


class PlannedTask(BaseModel):
    """Structured task proposed by the planning model."""

    task_id: str = Field(description="Unique task identifier")
    question: str = Field(description="Specific question this task must answer")
    retrieval_query: str = Field(
        description="Focused search query for retrieving supporting evidence"
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Task IDs that must complete before this task"
    )
    priority: int = Field(
        default=1,
        ge=1,
        description="Execution priority, where 1 is highest"
    )


class PlannerOutput(BaseModel):
    """Structured output returned by the planning model."""

    tasks: list[PlannedTask] = Field(
        description="Ordered list of tasks needed to answer the question"
    )


class TaskPlanner:
    """
    Creates validated task plans from user questions.

    Planning is intentionally separate from retrieval and answer generation.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model or Config.OLLAMA_CLASSIFIER_MODEL
        self.temperature = temperature

        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            format="json",
            base_url=Config.OLLAMA_BASE_URL,
        )

    def _build_prompt(self, question: str) -> str:
        return f"""
You are a task planning component for a multi-hop retrieval-augmented
generation system.

Your responsibility is to decompose the user's question into focused,
evidence-seeking retrieval tasks.

USER QUESTION:
{question}

PLANNING RULES:

1. Create the smallest number of tasks needed to answer the question.
2. A simple factual question should normally produce one task.
3. Complex questions may require multiple tasks.
4. Every task must be independently understandable.
5. Use task IDs in the format task_001, task_002, and so on.
6. Use depends_on only when a task genuinely requires the result of
   another task.
7. Tasks without dependencies should have an empty depends_on list.
8. A task must never depend on itself.
9. Do not create circular dependencies.
10. Retrieval queries must be concise and suitable for document search.
11. Do not answer the user's question.
12. Do not invent facts or assume answers.
13. Stay within the maximum task limit of {Config.MAX_PLANNED_TASKS}.
14. Do not create unnecessary tasks for information that can be retrieved
    with one focused query.

Return valid JSON using this exact structure:

{{
  "tasks": [
    {{
      "task_id": "task_001",
      "question": "A focused question",
      "retrieval_query": "A concise retrieval query",
      "depends_on": [],
      "priority": 1
    }}
  ]
}}

Return only JSON. Do not include Markdown fences or additional commentary.
""".strip()

    def _normalize_output(self, output: Any) -> list[TaskSpec]:
        if isinstance(output, PlannerOutput):
            planner_output = output
        elif isinstance(output, dict):
            planner_output = PlannerOutput.model_validate(output)
        else:
            raise TypeError(
                f"Unexpected planner output type: {type(output).__name__}"
            )

        tasks = [
            TaskSpec(
                task_id=item.task_id,
                question=item.question,
                retrieval_query=item.retrieval_query,
                depends_on=item.depends_on,
                priority=item.priority,
            )
            for item in planner_output.tasks
        ]

        return validate_task_plan(tasks)

    async def plan(self, question: str) -> list[TaskSpec]:
        """
        Generate and validate a task plan.

        Raises ValueError for invalid input or invalid generated plans.
        Raises other model/runtime exceptions to the caller.
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        prompt = self._build_prompt(question.strip())

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content

            if isinstance(content, str):
                output = json.loads(content)
            elif isinstance(content, dict):
                output = content
            else:
                raise TypeError(
                    "Planner returned an unsupported response content type."
                )

            tasks = self._normalize_output(output)

            logger.info(
                "Generated task plan with %d tasks for the user question.",
                len(tasks),
            )

            return tasks

        except (ValidationError, ValueError, TypeError) as exc:
            logger.exception("Planner generated an invalid task plan.")
            raise ValueError(
                f"Failed to generate a valid task plan: {exc}"
            ) from exc


async def create_task_plan(question: str) -> list[TaskSpec]:
    """Convenience function for generating a task plan."""

    planner = TaskPlanner()
    return await planner.plan(question)


# ============================================================
# Follow-up planning
# ============================================================

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