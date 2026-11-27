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
from agent.orchestrator import (
    initialize_task_execution,
    assess_current_evidence,
    plan_followup_execution,
    route_after_retrieval,
    route_after_assessment,
    route_after_followup,
)
from agent.task_dispatcher import (
    prepare_retrieval_batch,
    dispatch_retrieval_tasks,
    execute_retrieval_task_node,
    collect_retrieval_results,
)
from agent.answer_synthesis import (
    AnswerSynthesis,
    format_evidence_context,
    normalize_evidence,
    parse_synthesis_response,
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

# Final answer
workflow.add_node(
    "generate_final_answer",
    generate_final_answer,
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
workflow.add_edge(
    "generate_final_answer",
    "supervisor_finalize",
)

# Rejected requests also go through supervisor finalization.
workflow.add_edge("out_of_scope", "supervisor_finalize")
workflow.add_edge("prompt_injection", "supervisor_finalize")
workflow.add_edge("invalid_query", "supervisor_finalize")
workflow.add_edge("needs_clarification", "supervisor_finalize")

workflow.add_edge("supervisor_finalize", END)

graph = workflow.compile(name="multi-hop-agent")