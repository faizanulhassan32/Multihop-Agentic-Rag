from __future__ import annotations

import json
import re
from typing import (
    Any,
    Literal,
    Sequence,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from config import Config

from agent.models import EvidenceRecord


# ============================================================
# Answer synthesis
# ============================================================

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


# ============================================================
# Grounding validation
# ============================================================

class GroundingClaimDraft(BaseModel):
    """Intermediate claim returned by the local grounding model."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1)

    supported: bool

    supporting_evidence_ids: list[str] = Field(
        default_factory=list
    )

    reason: str


class GroundingDraft(BaseModel):
    """Intermediate model response containing claim assessments."""

    model_config = ConfigDict(extra="forbid")

    claims: list[GroundingClaimDraft] = Field(
        default_factory=list
    )

    overall_reason: str


class GroundingClaim(BaseModel):
    """One factual answer claim and its evidence attribution."""

    model_config = ConfigDict(extra="forbid")

    claim: str

    supporting_evidence_ids: list[str] = Field(
        default_factory=list
    )

    reason: str


class GroundingResult(BaseModel):
    """Structured claim-to-evidence validation result."""

    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "supported",
        "partially_supported",
        "unsupported",
        "insufficient_evidence",
        "validation_error",
    ]

    supported_claims: list[GroundingClaim] = Field(
        default_factory=list
    )

    unsupported_claims: list[GroundingClaim] = Field(
        default_factory=list
    )

    supporting_evidence_ids: list[str] = Field(
        default_factory=list
    )

    reason: str

    validation_method: str = (
        "local_llm_claim_to_evidence"
    )


def _response_content(response: Any) -> str:
    """Extract text from Ollama SDK responses and test doubles."""

    if isinstance(response, dict):

        message = response.get("message") or {}

        return str(
            message.get("content") or ""
        )

    message = getattr(
        response,
        "message",
        None,
    )

    return str(
        getattr(message, "content", "") or ""
    )


def _normalize_claim_text(value: str) -> str:
    """Normalize whitespace and case for answer-span comparison."""
    return " ".join((value or "").casefold().split())


def _result_from_draft(
    draft: GroundingDraft,
    available_ids: set[str],
    answer: str,
) -> GroundingResult:
    """
    Convert the model assessment into a validated result.

    A claim is supported only if:
    1. It appears verbatim in the generated answer.
    2. The model marks it as supported.
    3. It references at least one supplied evidence ID.
    """
    supported = []
    unsupported = []
    all_supporting_ids = []

    normalized_answer = _normalize_claim_text(answer)

    for item in draft.claims:
        claim_text = item.claim.strip()
        normalized_claim = _normalize_claim_text(claim_text)

        # The validator must not invent claims from the evidence.
        claim_appears_in_answer = (
            bool(normalized_claim)
            and normalized_claim in normalized_answer
        )

        known_ids = list(
            dict.fromkeys(
                evidence_id
                for evidence_id in item.supporting_evidence_ids
                if evidence_id in available_ids
            )
        )

        unknown_ids = [
            evidence_id
            for evidence_id in item.supporting_evidence_ids
            if evidence_id not in available_ids
        ]

        accepted_ids = (
            known_ids if claim_appears_in_answer else []
        )

        is_supported = (
            item.supported
            and claim_appears_in_answer
            and bool(accepted_ids)
        )

        reason = (
            item.reason.strip()
            or "No validation reason was provided."
        )

        if not claim_appears_in_answer:
            reason += (
                " The validator returned a claim that does not appear "
                "verbatim in the generated answer. It was rejected."
            )

        if unknown_ids:
            reason += (
                " Unknown evidence references were discarded: "
                + ", ".join(unknown_ids)
                + "."
            )

        claim = GroundingClaim(
            claim=claim_text,
            supporting_evidence_ids=accepted_ids,
            reason=reason,
        )

        if is_supported:
            supported.append(claim)
            all_supporting_ids.extend(accepted_ids)
        else:
            if item.supported and not accepted_ids and claim_appears_in_answer:
                claim.reason += (
                    " The model did not identify a valid "
                    "supporting evidence ID."
                )

            unsupported.append(claim)

    if not draft.claims:
        status = "insufficient_evidence"
        reason = (
            draft.overall_reason.strip()
            or "No factual claims could be validated."
        )

    elif len(supported) == len(draft.claims):
        status = "supported"
        reason = (
            draft.overall_reason.strip()
            or "All extracted factual claims have evidence references."
        )

    elif supported:
        status = "partially_supported"
        reason = (
            draft.overall_reason.strip()
            or "Some factual claims lack sufficient evidence."
        )

    else:
        status = "unsupported"
        reason = (
            draft.overall_reason.strip()
            or "No extracted factual claims were supported."
        )

    return GroundingResult(
        status=status,
        supported_claims=supported,
        unsupported_claims=unsupported,
        supporting_evidence_ids=list(
            dict.fromkeys(all_supporting_ids)
        ),
        reason=reason,
    )


async def validate_answer_grounding(
    answer: str,
    evidence: Sequence[
        EvidenceRecord | dict[str, Any]
    ],
    *,
    question: str | None = None,
    model: str | None = None,
    client: Any | None = None,
) -> GroundingResult:
    """
    Validate factual answer claims against retrieved evidence.

    Uses the local Ollama generation model.

    This is a focused claim-to-evidence validation step, not a formal
    entailment proof or a replacement for the project's later answer
    evaluation framework.

    The optional client argument allows deterministic unit testing
    without requiring a running Ollama instance.
    """

    normalized = normalize_evidence(evidence)

    answer = (answer or "").strip()

    if not normalized:

        return GroundingResult(
            status="insufficient_evidence",
            reason=(
                "No retrieved evidence was available "
                "for grounding validation."
            ),
        )

    if not answer:

        return GroundingResult(
            status="insufficient_evidence",
            reason=(
                "No answer was available "
                "for grounding validation."
            ),
        )

    available_ids = {
        item.doc_id
        for item in normalized
    }

    evidence_context = format_evidence_context(
        normalized
    )

    prompt = f"""
        You are a factual grounding validator in a retrieval-augmented
        question-answering system.

        Your task is to identify factual claims in the ANSWER and check
        each claim against the supplied RETRIEVED EVIDENCE only.

        Rules:

        1. Split the answer into concise, independently checkable factual claims.
        2. Mark a claim supported only when the evidence directly supports it.
        3. Do not use outside knowledge or assume missing relationships.
        4. For every supported claim, provide one or more exact Evidence IDs
        from the supplied evidence that support the claim.
        5. Never invent an Evidence ID.
        6. If evidence is related but does not establish the claim,
        mark the claim unsupported.
        7. Give a short reason for each decision.
        8. Do not rewrite the answer.
        9. Do not include hidden reasoning.
        10. Return only the requested JSON structure.
        11. Every claim must be copied verbatim from the ANSWER.
        12. Never create a claim using information found only in the evidence.
        13. If the answer contains only a short entity name or a fragment
            without a factual assertion, return an empty claims list.
        14. Evaluate the answer in the context of the original question.
        15. For a short direct answer, such as an entity name, assess whether
            that answer correctly resolves the question using the evidence.
            The claim must still be copied verbatim from the answer.

        QUESTION:

        {question or "Not supplied."}

        ANSWER:

        {answer}

        RETRIEVED EVIDENCE:

        {evidence_context}
    """.strip()

    try:

        if client is None:

            from ollama import AsyncClient

            async with AsyncClient(
                host=Config.OLLAMA_BASE_URL
            ) as ollama_client:

                response = await ollama_client.chat(
                    model=(
                        model
                        or Config.OLLAMA_GENERATION_MODEL
                    ),
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    format=GroundingDraft.model_json_schema(),
                    options={
                        "temperature": 0.0,
                        "num_predict": 700,
                    },
                )

        else:

            response = await client.chat(
                model=(
                    model
                    or Config.OLLAMA_GENERATION_MODEL
                ),
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                format=GroundingDraft.model_json_schema(),
                options={
                    "temperature": 0.0,
                    "num_predict": 700,
                },
            )

        raw_content = _response_content(response)

        if not raw_content:

            raise ValueError(
                "Ollama returned an empty grounding response."
            )

        draft = GroundingDraft.model_validate(
            json.loads(raw_content)
        )

        return _result_from_draft(
            draft,
            available_ids,
            answer,
        )

    except Exception as exc:

        return GroundingResult(
            status="validation_error",
            reason=(
                "Grounding validation could not be completed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )