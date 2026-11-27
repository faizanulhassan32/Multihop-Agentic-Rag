from __future__ import annotations

import json
import re
from typing import Any, Sequence, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from agent.task_models import EvidenceRecord


class AnswerSynthesis(BaseModel):
    """Structured answer-generation result retained in graph state."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)

    used_evidence_ids: list[str] = Field(
        default_factory=list
    )

    status: Literal[
        "structured",
        "unstructured_fallback",
        "no_evidence",
    ] = "structured"

    warnings: list[str] = Field(
        default_factory=list
    )

    @field_validator("used_evidence_ids")
    @classmethod
    def deduplicate_evidence_ids(
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


def normalize_evidence(
    evidence: Sequence[
        EvidenceRecord | dict[str, Any]
    ],
) -> list[EvidenceRecord]:
    """Normalize evidence records into validated Pydantic models."""

    return [
        item
        if isinstance(item, EvidenceRecord)
        else EvidenceRecord.model_validate(item)
        for item in evidence
    ]


def format_evidence_context(
    evidence: Sequence[
        EvidenceRecord | dict[str, Any]
    ],
) -> str:
    """
    Format accumulated evidence with stable IDs and provenance.

    The prompt includes useful source metadata, originating task IDs,
    and retrieval rounds without dumping arbitrary metadata fields.
    """

    normalized = normalize_evidence(evidence)

    sections = []

    for item in normalized:

        metadata = item.metadata or {}

        source_fields = []

        for key in (
            "title",
            "source",
            "url",
            "page",
            "section",
        ):

            value = metadata.get(key)

            if value not in (None, ""):
                source_fields.append(
                    f"{key}={value}"
                )

        source_text = (
            "; ".join(source_fields)
            if source_fields
            else "source metadata unavailable"
        )

        task_ids = (
            ", ".join(item.source_task_ids)
            if item.source_task_ids
            else "unknown"
        )

        rounds = (
            ", ".join(
                str(value)
                for value in item.retrieval_rounds
            )
            if item.retrieval_rounds
            else "unknown"
        )

        sections.append(
            f"Evidence ID: {item.doc_id}\n"
            f"Source metadata: {source_text}\n"
            f"Source task IDs: {task_ids}\n"
            f"Retrieval rounds: {rounds}\n"
            f"Evidence content:\n{item.content}"
        )

    return "\n\n".join(sections)


def parse_synthesis_response(
    raw_response: str,
    available_evidence_ids: set[str],
) -> AnswerSynthesis:
    """
    Parse structured model output.

    Only evidence IDs actually supplied to the model are accepted.
    If structured output parsing fails, preserve the generated text
    as an unstructured fallback rather than losing the answer.
    """

    raw = (raw_response or "").strip()

    cleaned = raw

    if cleaned.startswith("```"):

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

    try:

        payload = json.loads(cleaned)

        parsed = AnswerSynthesis.model_validate(
            payload
        )

    except Exception:

        fallback = (
            raw
            or
            "I couldn't generate a reliable answer "
            "from the retrieved evidence."
        )

        return AnswerSynthesis(
            answer=fallback,
            status="unstructured_fallback",
            warnings=[
                "The model response was not valid structured synthesis JSON."
            ],
        )

    warnings = list(parsed.warnings)

    valid_ids = []

    for evidence_id in parsed.used_evidence_ids:

        if evidence_id in available_evidence_ids:

            valid_ids.append(evidence_id)

        else:

            warnings.append(
                "Model referenced an evidence ID that was not supplied: "
                f"{evidence_id}"
            )

    # Also inspect inline source markers.
    cited_ids = re.findall(
        r"\[source:([^\]]+)\]",
        parsed.answer,
    )

    for evidence_id in cited_ids:

        evidence_id = evidence_id.strip()

        if (
            evidence_id in available_evidence_ids
            and evidence_id not in valid_ids
        ):

            valid_ids.append(evidence_id)

        elif evidence_id not in available_evidence_ids:

            warnings.append(
                "Answer citation referenced an unknown evidence ID: "
                f"{evidence_id}"
            )

    return AnswerSynthesis(
        answer=parsed.answer.strip(),
        used_evidence_ids=valid_ids,
        status="structured",
        warnings=list(dict.fromkeys(warnings)),
    )