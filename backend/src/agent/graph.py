from __future__ import annotations

import logging
import sys
import os
import time
from datetime import datetime
from typing import Any, Dict, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from ollama import AsyncClient
from langgraph.graph import StateGraph, END

# Preserve the existing local library resolution.
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../libs")
    )
)

from config import Config
from agent.classifier import run_classifier
from agent.state import MultiHopState
from agent.evidence_assessor import assess_evidence
from agent.models import TaskSpec
from agent.planner import create_followup_tasks, create_task_plan
from agent.task_manager import get_execution_status
from agent.task_dispatcher import (
    prepare_retrieval_batch,
    dispatch_retrieval_tasks,
    execute_retrieval_task_node,
    collect_retrieval_results,
)
from agent.synthesis import (
    AnswerSynthesis,
    format_evidence_context,
    normalize_evidence,
    parse_synthesis_response,
    validate_answer_grounding,
)

load_dotenv()

logger = logging.getLogger(__name__)


class Configuration(TypedDict, total=False):
    model: str
    temperature: float
    max_tokens: int


def _request_id(config: RunnableConfig) -> str:
    metadata = config.get("metadata") or {}
    return str(metadata.get("request_id", "unknown"))


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------
# Supervisor classification and routing
# ---------------------------------------------------------------------

async def supervisor_classify(
    state: MultiHopState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Classify the incoming user query."""

    question = state.get("original_question", "")
    request_id = _request_id(config)
    started_at = time.perf_counter()

    logger.info(
        "Supervisor classification started: request_id=%s",
        request_id,
    )

    result = await run_classifier(question)

    elapsed_ms = (time.perf_counter() - started_at) * 1000

    logger.info(
        "Supervisor classification completed: request_id=%s "
        "classification=%s elapsed_ms=%.2f",
        request_id,
        result.classification,
        elapsed_ms,
    )

    return {
        "classification": result.classification,
        "classification_reason": result.reason,
        "request_id": request_id,
        "trace": [
            {
                "step": "supervisor_classify",
                "timestamp": _timestamp(),
                "input": {"question": question},
                "output": {
                    "classification": result.classification,
                    "reason": result.reason,
                },
            }
        ],
    }


async def out_of_scope(
    state: MultiHopState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    message = (
        "I'm sorry, I can only answer factual, multi-hop knowledge questions."
    )

    return {
        "final_answer": message,
        "trace": [
            {
                "step": "out_of_scope",
                "timestamp": _timestamp(),
                "output": {"final_answer": message},
            }
        ],
    }


async def prompt_injection(
    state: MultiHopState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    message = "I'm sorry, I cannot process that request."

    return {
        "final_answer": message,
        "trace": [
            {
                "step": "prompt_injection",
                "timestamp": _timestamp(),
                "output": {"final_answer": message},
            }
        ],
    }


async def invalid_query(
    state: MultiHopState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    message = "I'm sorry, the query appears to be invalid. Please rephrase."

    return {
        "final_answer": message,
        "trace": [
            {
                "step": "invalid_query",
                "timestamp": _timestamp(),
                "output": {"final_answer": message},
            }
        ],
    }


async def needs_clarification(
    state: MultiHopState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    message = (
        "I need a little more clarification before I can answer. "
        "Could you elaborate?"
    )

    return {
        "final_answer": message,
        "trace": [
            {
                "step": "needs_clarification",
                "timestamp": _timestamp(),
                "output": {"final_answer": message},
            }
        ],
    }


async def supervisor_finalize(
    state: MultiHopState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Preserve the worker's answer and expose it as the final answer."""

    worker_answer = state.get("final_answer", "")

    if not worker_answer:
        worker_answer = "[No answer generated]"

    return {
        "final_answer": worker_answer,
        "worker_answer": worker_answer,
        "trace": [
            {
                "step": "supervisor_finalize",
                "timestamp": _timestamp(),
                "output": {"final_answer": worker_answer},
            }
        ],
    }


def route_based_on_classification(
    state: MultiHopState,
) -> Literal[
    "initialize_task_execution",
    "out_of_scope",
    "prompt_injection",
    "invalid_query",
    "needs_clarification",
]:
    label = state.get("classification", "invalid")

    mapping = {
        "in_scope": "initialize_task_execution",
        "out_of_scope": "out_of_scope",
        "prompt_injection": "prompt_injection",
        "invalid": "invalid_query",
        "needs_clarification": "needs_clarification",
    }

    return mapping.get(label, "invalid_query")


# ---------------------------------------------------------------------
# Evidence-based final answer generation
# ---------------------------------------------------------------------

async def generate_final_answer(
    state: MultiHopState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    Synthesize one structured answer from all accumulated evidence.

    This retains the existing graph node and routing. Grounding validation
    is implemented separately in Stage 6 and will be connected to graph
    routing during Stage 7.
    """

    request_id = _request_id(config)
    started_at = time.perf_counter()

    question = state.get("original_question", "")
    evidence = state.get("evidence", [])

    logger.info(
        "Final answer synthesis started: request_id=%s evidence_count=%d",
        request_id,
        len(evidence),
    )

    if not evidence:

        synthesis = AnswerSynthesis(
            answer=(
                "I don't have enough relevant information in my "
                "knowledge base to answer this question reliably."
            ),
            status="no_evidence",
            warnings=[
                "No evidence records were available for synthesis."
            ],
        )

        return {
            "final_answer": synthesis.answer,
            "answer_synthesis": synthesis.model_dump(
                mode="json"
            ),
            "trace": [
                {
                    "step": "generate_final_answer",
                    "timestamp": _timestamp(),
                    "input": {
                        "question": question,
                        "evidence_count": 0,
                    },
                    "output": {
                        "final_answer": synthesis.answer,
                        "synthesis_status": synthesis.status,
                        "used_evidence_ids": [],
                    },
                }
            ],
        }

    normalized_evidence = normalize_evidence(
        evidence
    )

    evidence_context = format_evidence_context(
        normalized_evidence
    )

    available_evidence_ids = {
        item.doc_id
        for item in normalized_evidence
    }

    configuration = config.get(
        "configurable",
        {},
    )

    prompt = ChatPromptTemplate.from_template(
    """
        You are the answer synthesis component of a factual
        multi-hop question-answering system.

        Generate one coherent answer to the ORIGINAL QUESTION using
        ONLY the ACCUMULATED RETRIEVED EVIDENCE.

        Requirements:

        1. Answer the original question directly.
        2. Combine evidence across independent retrieval tasks when
        necessary to resolve multi-hop relationships.
        3. Use evidence from both original retrieval and follow-up retrieval.
        4. Do not invent facts, relationships, dates, names, or explanations.
        5. If the evidence is contradictory or insufficient, explicitly
        communicate that limitation.
        6. Cite factual statements inline using this exact format:

        [source:doc_id]

        7. Use only Evidence IDs supplied below.
        8. Never invent an Evidence ID.
        9. Include cited IDs in used_evidence_ids.
        10. Do not mention internal task IDs or retrieval rounds.
        11. Keep the answer concise and focused.
        12. Return only JSON matching the required schema.

        ORIGINAL QUESTION:

        {question}

        ACCUMULATED RETRIEVED EVIDENCE:

        {evidence}
    """
    )

    prompt_text = prompt.format(
        question=question,
        evidence=evidence_context,
    )

    answer_chunks: list[str] = []

    async with AsyncClient(
        host=Config.OLLAMA_BASE_URL
    ) as client:

        response_stream = await client.chat(
            model=configuration.get(
                "model",
                Config.OLLAMA_GENERATION_MODEL,
            ),
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            format=AnswerSynthesis.model_json_schema(),
            options={
                "temperature": configuration.get(
                    "temperature",
                    0.2,
                ),
                "num_predict": configuration.get(
                    "max_tokens",
                    700,
                ),
            },
            stream=True,
        )

        async for response_chunk in response_stream:

            token = response_chunk.message.content

            if token:
                answer_chunks.append(token)

    raw_response = "".join(
        answer_chunks
    ).strip()

    synthesis = parse_synthesis_response(
        raw_response=raw_response,
        available_evidence_ids=available_evidence_ids,
    )

    elapsed_ms = (
        time.perf_counter() - started_at
    ) * 1000

    synthesis_payload = synthesis.model_dump(
        mode="json"
    )

    logger.info(
        "Final answer synthesis completed: request_id=%s "
        "elapsed_ms=%.2f status=%s cited_evidence=%d",
        request_id,
        elapsed_ms,
        synthesis.status,
        len(synthesis.used_evidence_ids),
    )

    return {
        # Preserve the existing final_answer contract.
        "final_answer": synthesis.answer,

        # Retain structured synthesis for Stage 7 grounding.
        "answer_synthesis": synthesis_payload,

        "trace": [
            {
                "step": "generate_final_answer",
                "timestamp": _timestamp(),
                "input": {
                    "question": question,
                    "evidence_count": len(normalized_evidence),
                    "evidence_assessment": state.get(
                        "evidence_assessment",
                        {},
                    ),
                    "followup_round": state.get(
                        "followup_round",
                        0,
                    ),
                    "evidence_ids": sorted(
                        available_evidence_ids
                    ),
                    "evidence_provenance": [
                        {
                            "doc_id": item.doc_id,
                            "source_task_ids": (
                                item.source_task_ids
                            ),
                            "retrieval_rounds": (
                                item.retrieval_rounds
                            ),
                        }
                        for item in normalized_evidence
                    ],
                },
                "output": {
                    "final_answer": synthesis.answer,
                    "synthesis_status": synthesis.status,
                    "used_evidence_ids": (
                        synthesis.used_evidence_ids
                    ),
                    "warnings": synthesis.warnings,
                },
            }
        ],
    }


# ---------------------------------------------------------------------
# Stage 7: Grounding validation and answer policy
# ---------------------------------------------------------------------

async def validate_final_answer_grounding(
    state: MultiHopState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    Validate the synthesized answer against accumulated evidence.
    """

    request_id = _request_id(config)
    answer = state.get("final_answer", "")
    evidence = state.get("evidence", [])

    started_at = time.perf_counter()

    result = await validate_answer_grounding(
        answer=answer,
        evidence=evidence,
        question=state.get("original_question", ""),
    )

    elapsed_ms = (
        time.perf_counter() - started_at
    ) * 1000

    result_payload = result.model_dump(mode="json")

    logger.info(
        "Grounding validation completed: request_id=%s "
        "status=%s supported_claims=%d unsupported_claims=%d "
        "elapsed_ms=%.2f",
        request_id,
        result.status,
        len(result.supported_claims),
        len(result.unsupported_claims),
        elapsed_ms,
    )

    return {
        "grounding_result": result_payload,
        "trace": [
            {
                "step": "validate_final_answer_grounding",
                "timestamp": _timestamp(),
                "input": {
                    "answer": answer,
                    "evidence_count": len(evidence),
                },
                "output": {
                    "status": result.status,
                    "supported_claim_count": len(
                        result.supported_claims
                    ),
                    "unsupported_claim_count": len(
                        result.unsupported_claims
                    ),
                    "supporting_evidence_ids": (
                        result.supporting_evidence_ids
                    ),
                    "reason": result.reason,
                },
            }
        ],
    }


def route_after_grounding_validation(
    state: MultiHopState,
) -> Literal[
    "supervisor_finalize",
    "apply_grounding_policy",
]:
    """
    Only fully supported answers proceed unchanged.
    All other results require the grounding policy.
    """

    grounding_result = state.get("grounding_result", {})
    status = grounding_result.get("status")

    if status == "supported":
        return "supervisor_finalize"

    return "apply_grounding_policy"


async def apply_grounding_policy(
    state: MultiHopState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    Preserve supported claims and citations from partially supported
    answers. Abstain when no supported claims are available.

    Validation errors and insufficient evidence fail closed.
    """

    grounding_result = state.get("grounding_result", {})
    status = grounding_result.get("status", "validation_error")

    supported_claims = grounding_result.get(
        "supported_claims",
        [],
    )

    grounded_statements = []

    if status == "partially_supported":
        for claim in supported_claims:
            claim_text = str(
                claim.get("claim", "")
            ).strip()

            evidence_ids = claim.get(
                "supporting_evidence_ids",
                [],
            )

            if not claim_text or not evidence_ids:
                continue

            citations = " ".join(
                f"[source:{evidence_id}]"
                for evidence_id in evidence_ids
            )

            grounded_statements.append(
                f"{claim_text} {citations}"
            )

    if grounded_statements:
        final_answer = " ".join(grounded_statements)

        policy_action = "returned_supported_claims"

    else:
        final_answer = (
            "I couldn't verify a reliable answer from the retrieved "
            "evidence, so I can't provide a factual answer with confidence."
        )

        policy_action = "abstained"

    logger.warning(
        "Grounding policy applied: request_id=%s "
        "validation_status=%s action=%s",
        _request_id(config),
        status,
        policy_action,
    )

    return {
        "final_answer": final_answer,
        "trace": [
            {
                "step": "apply_grounding_policy",
                "timestamp": _timestamp(),
                "input": {
                    "grounding_status": status,
                    "supported_claim_count": len(
                        supported_claims
                    ),
                },
                "output": {
                    "action": policy_action,
                    "final_answer": final_answer,
                },
            }
        ],
    }


# ---------------------------------------------------------------------
# Task planning and orchestration
# ---------------------------------------------------------------------

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

    tasks = await create_task_plan(question)

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
    """Create bounded follow-up tasks for remaining evidence gaps."""

    raw_tasks = state.get("tasks", [])
    raw_assessment = state.get("evidence_assessment")

    if not raw_assessment:
        return {
            "tasks": raw_tasks,
            "followup_tasks_created": False,
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
            "No follow-up tasks created: round=%d max_rounds=%d task_count=%d",
            current_round,
            Config.MAX_FOLLOWUP_ROUNDS,
            len(raw_tasks),
        )

        return {
            "tasks": raw_tasks,
            "followup_tasks_created": False,
            "execution_status": get_execution_status([
                TaskSpec.model_validate(task)
                for task in raw_tasks
            ]),
            "trace": [
                {
                    "step": "plan_followup_execution",
                    "timestamp": _timestamp(),
                    "output": {
                        "followup_tasks_created": False,
                        "reason": "No eligible follow-up tasks could be created.",
                    },
                }
            ],
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
        "tasks": [task.model_dump() for task in updated_tasks],
        "followup_round": current_round + 1,
        "followup_tasks_created": True,
        "execution_status": get_execution_status(updated_tasks),
        "trace": [
            {
                "step": "plan_followup_execution",
                "timestamp": _timestamp(),
                "output": {
                    "followup_round": current_round + 1,
                    "followup_tasks_created": True,
                    "created_task_ids": [
                        task.task_id for task in followup_tasks
                    ],
                },
            }
        ],
    }


def route_after_followup(
    state: dict[str, Any],
) -> str:
    """
    Return to retrieval only when new follow-up tasks were actually created.

    This prevents the graph from repeatedly scheduling an unchanged,
    already-terminal task list.
    """
    if not state.get("followup_tasks_created", False):
        return "generate_final_answer"

    tasks = [
        task if isinstance(task, TaskSpec)
        else TaskSpec.model_validate(task)
        for task in state.get("tasks", [])
    ]

    followup_tasks = [
        task for task in tasks
        if task.followup_round > 0
    ]

    has_pending_work = any(
        task.status in {"pending", "running"}
        for task in followup_tasks
    )

    return (
        "prepare_retrieval_batch"
        if has_pending_work
        else "generate_final_answer"
    )


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



# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------

workflow = StateGraph(
    MultiHopState,
    config_schema=Configuration,
)

# Supervisor nodes
workflow.add_node("supervisor_classify", supervisor_classify)
workflow.add_node("out_of_scope", out_of_scope)
workflow.add_node("prompt_injection", prompt_injection)
workflow.add_node("invalid_query", invalid_query)
workflow.add_node("needs_clarification", needs_clarification)
workflow.add_node("supervisor_finalize", supervisor_finalize)

# Planning and orchestration
workflow.add_node(
    "initialize_task_execution",
    initialize_task_execution,
)

# Retrieval execution
workflow.add_node(
    "prepare_retrieval_batch",
    prepare_retrieval_batch,
)

workflow.add_node(
    "execute_retrieval_task_node",
    execute_retrieval_task_node,
)

workflow.add_node(
    "collect_retrieval_results",
    collect_retrieval_results,
)

# Evidence assessment and adaptive retrieval
workflow.add_node(
    "assess_current_evidence",
    assess_current_evidence,
)

workflow.add_node(
    "plan_followup_execution",
    plan_followup_execution,
)

# Final answer and Stage 7 grounding
workflow.add_node(
    "generate_final_answer",
    generate_final_answer,
)

workflow.add_node(
    "validate_final_answer_grounding",
    validate_final_answer_grounding,
)

workflow.add_node(
    "apply_grounding_policy",
    apply_grounding_policy,
)

# Entry point
workflow.set_entry_point("supervisor_classify")

# Classification routing
workflow.add_conditional_edges(
    "supervisor_classify",
    route_based_on_classification,
    {
        "initialize_task_execution": "initialize_task_execution",
        "out_of_scope": "out_of_scope",
        "prompt_injection": "prompt_injection",
        "invalid_query": "invalid_query",
        "needs_clarification": "needs_clarification",
    },
)

# Initialize task plan and start retrieval
workflow.add_edge(
    "initialize_task_execution",
    "prepare_retrieval_batch",
)

# Dynamically dispatch ready tasks.
# A list of Send objects fans out to parallel retrieval workers.
workflow.add_conditional_edges(
    "prepare_retrieval_batch",
    dispatch_retrieval_tasks,
    {
        "no_tasks_to_dispatch": "assess_current_evidence",
    },
)

# Each worker contributes task_results and evidence through reducers.
workflow.add_edge(
    "execute_retrieval_task_node",
    "collect_retrieval_results",
)

# Assess evidence when the current retrieval plan reaches a terminal state.
workflow.add_conditional_edges(
    "collect_retrieval_results",
    route_after_retrieval,
    {
        "prepare_retrieval_batch": "prepare_retrieval_batch",
        "assess_current_evidence": "assess_current_evidence",
    },
)

# Decide whether the accumulated evidence is sufficient.
workflow.add_conditional_edges(
    "assess_current_evidence",
    route_after_assessment,
    {
        "plan_followup_execution": "plan_followup_execution",
        "generate_final_answer": "generate_final_answer",
    },
)

# Follow-up tasks return to the same bounded retrieval scheduler.
workflow.add_conditional_edges(
    "plan_followup_execution",
    route_after_followup,
    {
        "prepare_retrieval_batch": "prepare_retrieval_batch",
        "generate_final_answer": "generate_final_answer",
    },
)

# Supervisor owns finalization.
# Every generated answer must pass grounding validation.
workflow.add_edge(
    "generate_final_answer",
    "validate_final_answer_grounding",
)

# Route based on the structured grounding result.
workflow.add_conditional_edges(
    "validate_final_answer_grounding",
    route_after_grounding_validation,
    {
        "supervisor_finalize": "supervisor_finalize",
        "apply_grounding_policy": "apply_grounding_policy",
    },
)

# The policy produces a safe answer before supervisor finalization.
workflow.add_edge(
    "apply_grounding_policy",
    "supervisor_finalize",
)

# Rejected requests also go through supervisor finalization.
workflow.add_edge("out_of_scope", "supervisor_finalize")
workflow.add_edge("prompt_injection", "supervisor_finalize")
workflow.add_edge("invalid_query", "supervisor_finalize")
workflow.add_edge("needs_clarification", "supervisor_finalize")

workflow.add_edge("supervisor_finalize", END)

graph = workflow.compile(name="multi-hop-agent")