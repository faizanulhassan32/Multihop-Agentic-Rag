from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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