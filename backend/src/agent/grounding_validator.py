from __future__ import annotations

import json

from typing import (
    Any,
    Literal,
    Sequence,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from config import Config

from agent.answer_synthesis import (
    format_evidence_context,
    normalize_evidence,
)

from agent.task_models import EvidenceRecord


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


def _result_from_draft(
    draft: GroundingDraft,
    available_ids: set[str],
) -> GroundingResult:
    """
    Convert the model assessment into a validated result.

    A claim cannot be considered supported unless it references
    at least one evidence ID that was actually supplied.
    """

    supported = []
    unsupported = []

    all_supporting_ids = []

    for item in draft.claims:

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

        is_supported = (
            item.supported
            and bool(known_ids)
        )

        reason = (
            item.reason.strip()
            or "No validation reason was provided."
        )

        if unknown_ids:

            reason += (
                " Unknown evidence references were discarded: "
                + ", ".join(unknown_ids)
                + "."
            )

        claim = GroundingClaim(
            claim=item.claim.strip(),
            supporting_evidence_ids=known_ids,
            reason=reason,
        )

        if is_supported:

            supported.append(claim)

            all_supporting_ids.extend(
                known_ids
            )

        else:

            if item.supported and not known_ids:

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
        )

    except Exception as exc:

        return GroundingResult(
            status="validation_error",
            reason=(
                "Grounding validation could not be completed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )