from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# Task lifecycle models
# ============================================================

TaskStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "skipped",
]


class TaskSpec(BaseModel):
    """A planned unit of work within a multi-hop RAG execution."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    retrieval_query: str = Field(min_length=1)

    depends_on: list[str] = Field(default_factory=list)

    status: TaskStatus = "pending"

    priority: int = Field(default=1, ge=1)

    followup_round: int = Field(default=0, ge=0)

    evidence_ids: list[str] = Field(default_factory=list)

    attempts: int = Field(default=0, ge=0)

    error: str | None = None

    @field_validator(
        "task_id",
        "question",
        "retrieval_query",
    )
    @classmethod
    def require_nonempty_strings(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "This field must not be empty."
            )

        return value

    @field_validator(
        "depends_on",
        "evidence_ids",
    )
    @classmethod
    def remove_duplicate_ids(
        cls,
        values: list[str],
    ) -> list[str]:
        return list(
            dict.fromkeys(
                value.strip()
                for value in values
                if value.strip()
            )
        )

    @model_validator(mode="after")
    def validate_dependencies(self) -> "TaskSpec":

        if self.task_id in self.depends_on:
            raise ValueError(
                "A task cannot depend on itself."
            )

        return self


class TaskResult(BaseModel):
    """Execution outcome returned by a retrieval worker."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)

    status: TaskStatus

    evidence_ids: list[str] = Field(
        default_factory=list
    )

    attempts: int = Field(
        default=0,
        ge=0,
    )

    error: str | None = None

    latency_ms: float | None = Field(
        default=None,
        ge=0,
    )


# ============================================================
# Evidence models
# ============================================================

class EvidenceRecord(BaseModel):
    """A retrieved document with stable identity and provenance."""

    model_config = ConfigDict(extra="allow")

    doc_id: str = Field(min_length=1)

    content: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    source_task_ids: list[str] = Field(
        default_factory=list
    )

    retrieval_rounds: list[int] = Field(
        default_factory=list
    )

    @field_validator("doc_id")
    @classmethod
    def require_nonempty_document_id(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "doc_id must not be empty."
            )

        return value

    @field_validator("source_task_ids")
    @classmethod
    def remove_duplicate_task_ids(
        cls,
        values: list[str],
    ) -> list[str]:

        return list(
            dict.fromkeys(
                value.strip()
                for value in values
                if value.strip()
            )
        )

    @field_validator("retrieval_rounds")
    @classmethod
    def remove_duplicate_rounds(
        cls,
        values: list[int],
    ) -> list[int]:

        if any(value < 0 for value in values):
            raise ValueError(
                "retrieval_rounds cannot contain negative values."
            )

        return list(dict.fromkeys(values))


# ============================================================
# Evidence assessment models
# ============================================================

class TaskEvidenceAssessment(BaseModel):
    """Evidence coverage assessment for one planned task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: Literal["sufficient", "partial", "insufficient"]
    coverage_score: float = Field(ge=0.0, le=1.0)
    matched_terms: list[str] = Field(default_factory=list)
    missing_terms: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str


class EvidenceAssessment(BaseModel):
    """Aggregate evidence assessment for the original question."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["sufficient", "partial", "insufficient"]
    coverage_score: float = Field(ge=0.0, le=1.0)
    task_assessments: list[TaskEvidenceAssessment]
    gap_task_ids: list[str] = Field(default_factory=list)
    evidence_count: int = Field(ge=0)
    assessment_method: str = "lexical_coverage"
    reason: str