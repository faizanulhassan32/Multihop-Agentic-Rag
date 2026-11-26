
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TaskStatus = Literal["pending", "running", "completed", "failed", "skipped"]


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

    @field_validator("task_id", "question", "retrieval_query")
    @classmethod
    def require_nonempty_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field must not be empty.")
        return value

    @field_validator("depends_on", "evidence_ids")
    @classmethod
    def remove_duplicate_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @model_validator(mode="after")
    def validate_dependencies(self) -> "TaskSpec":
        if self.task_id in self.depends_on:
            raise ValueError("A task cannot depend on itself.")
        return self


class TaskResult(BaseModel):
    """Execution outcome returned by a retrieval worker."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    status: TaskStatus
    evidence_ids: list[str] = Field(default_factory=list)
    attempts: int = Field(default=0, ge=0)
    error: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)


class EvidenceRecord(BaseModel):
    """A retrieved document with its provenance preserved."""

    model_config = ConfigDict(extra="allow")

    doc_id: str = Field(min_length=1)
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_task_ids: list[str] = Field(default_factory=list)

    @field_validator("source_task_ids")
    @classmethod
    def remove_duplicate_task_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))
