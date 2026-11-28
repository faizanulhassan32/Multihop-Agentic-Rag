
from __future__ import annotations

import inspect
import unittest
from typing import Any
import asyncio
import inspect
from agent.retrieval import rag_system
from agent.graph import graph
from agent import synthesis as grounding_validator


QUESTION = (
    "Which magazine was started first, Arthur's Magazine "
    "or First for Women?"
)

UNSUPPORTED_ANSWER = (
    "Arthur's Magazine was started first, and it was founded in 1800 "
    "by a person named John Smith."
)


def _to_dict(value: Any) -> dict:
    """Convert common structured model outputs into dictionaries."""
    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    if hasattr(value, "__dict__"):
        return vars(value)

    return {}


def _get_value(data: Any, *keys: str) -> Any:
    """Read the first available field from a dictionary or model."""
    if not isinstance(data, dict):
        data = _to_dict(data)

    for key in keys:
        value = data.get(key)
        if value is not None:
            return value

    return None


def _document_id(document: Any) -> str | None:
    """Extract a stable document ID from a retrieved evidence item."""
    if isinstance(document, dict):
        metadata = document.get("metadata") or {}
        return (
            document.get("doc_id")
            or document.get("id")
            or metadata.get("doc_id")
            or metadata.get("id")
        )

    metadata = getattr(document, "metadata", {}) or {}

    return (
        getattr(document, "doc_id", None)
        or getattr(document, "id", None)
        or metadata.get("doc_id")
        or metadata.get("id")
    )


def _extract_evidence(result: dict) -> list:
    """
    Extract evidence from the graph's final state.

    The graph may expose evidence directly or through task results.
    """
    for key in (
        "evidence",
        "retrieved_evidence",
        "evidence_items",
        "documents",
        "retrieved_documents",
    ):
        value = result.get(key)
        if isinstance(value, list) and value:
            return value

    for key in ("task_results", "retrieval_results"):
        value = result.get(key)

        if isinstance(value, dict):
            collected = []

            for task_result in value.values():
                task_result = _to_dict(task_result)

                for evidence_key in (
                    "evidence",
                    "documents",
                    "retrieved_documents",
                ):
                    items = task_result.get(evidence_key)
                    if isinstance(items, list):
                        collected.extend(items)

            if collected:
                return collected

        elif isinstance(value, list):
            collected = []

            for task_result in value:
                task_result = _to_dict(task_result)

                for evidence_key in (
                    "evidence",
                    "documents",
                    "retrieved_documents",
                ):
                    items = task_result.get(evidence_key)
                    if isinstance(items, list):
                        collected.extend(items)

            if collected:
                return collected

    return []


def _extract_answer(result: dict) -> str:
    """Extract the synthesized answer from the graph's final state."""
    for key in (
        "final_answer",
        "answer",
        "response",
        "generated_answer",
    ):
        value = result.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

        if isinstance(value, dict):
            nested = (
                value.get("answer")
                or value.get("final_answer")
                or value.get("text")
            )
            if isinstance(nested, str) and nested.strip():
                return nested.strip()

    raise AssertionError(
        "Could not find the synthesized answer in the graph result. "
        f"Available state keys: {list(result.keys())}"
    )


def _argument_values(
    parameter_name: str,
    *,
    question: str,
    answer: str,
    evidence: list,
) -> Any:
    """Map validator parameters to the corresponding test data."""
    name = parameter_name.lower()

    if any(term in name for term in ("question", "query", "prompt")):
        return question

    if any(term in name for term in ("answer", "response", "completion")):
        return answer

    if any(term in name for term in ("evidence", "documents", "context", "sources")):
        return evidence

    if "citation" in name:
        return [_document_id(item) for item in evidence if _document_id(item)]

    return inspect.Parameter.empty



async def _invoke_validator(
    *,
    question: str,
    answer: str,
    evidence: list,
) -> Any:
    """Invoke the asynchronous grounding validator."""

    validator = grounding_validator.validate_answer_grounding
    signature = inspect.signature(validator)
    kwargs = {}

    for parameter_name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        value = _argument_values(
            parameter_name,
            question=question,
            answer=answer,
            evidence=evidence,
        )

        if value is not inspect.Parameter.empty:
            kwargs[parameter_name] = value
        elif parameter.default is inspect.Parameter.empty:
            raise TypeError(
                f"Cannot resolve required validator parameter: "
                f"{parameter_name}"
            )

    result = validator(**kwargs)

    if inspect.isawaitable(result):
        result = await result

    return result


def _grounding_status(result: Any) -> str:
    """Normalize common structured grounding status fields."""
    data = _to_dict(result)

    status = (
        data.get("status")
        or data.get("grounding_status")
        or data.get("validation_status")
        or data.get("verdict")
    )

    if status is not None:
        return str(status).lower()

    for key in ("is_grounded", "grounded", "valid", "is_valid"):
        if key in data:
            return "grounded" if data[key] is True else "unsupported"

    return ""


def _unsupported_claims(result: Any) -> list:
    data = _to_dict(result)

    for key in (
        "unsupported_claims",
        "unsupported",
        "unverified_claims",
        "unsupported_statements",
    ):
        value = data.get(key)
        if value:
            return value if isinstance(value, list) else [value]

    return []


class TestStage6PipelineIntegration(unittest.TestCase):
    """
    Integration tests for real retrieval, synthesis, attribution,
    and standalone grounding validation.

    Grounding validation is deliberately invoked outside the graph.
    """

    @classmethod

    @classmethod
    def setUpClass(cls):
        cls.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(cls.loop)

        async def initialize():
            # Initialize retrieval infrastructure first.
            startup_result = rag_system.startup()

            if inspect.isawaitable(startup_result):
                await startup_result

            # Run the actual asynchronous LangGraph workflow.
            result = await graph.ainvoke(
                {"original_question": QUESTION},
                config={"recursion_limit": 30},
            )

            if not isinstance(result, dict):
                result = _to_dict(result)

            return result

        try:
            cls.graph_result = cls.loop.run_until_complete(
                initialize()
            )

            # Extract final answer and evidence.
            cls.answer = _extract_answer(cls.graph_result)
            cls.evidence = _extract_evidence(cls.graph_result)

            # Extract the original synthesized answer before
            # Stage 7 grounding policy potentially modifies it.
            synthesis = cls.graph_result.get(
                "answer_synthesis",
                {},
            )

            cls.synthesized_answer = synthesis.get(
                "answer",
                cls.answer,
            )

            # Ensure actual evidence was retrieved.
            if not cls.evidence:
                raise AssertionError(
                    "The graph completed without retrieving evidence. "
                    "Check the retrieval logs."
                )

        except Exception:
            cls.loop.run_until_complete(
                rag_system.shutdown()
            )
            cls.loop.close()
            raise
    
    @classmethod
    def tearDownClass(cls):
        try:
            shutdown_result = rag_system.shutdown()

            if inspect.isawaitable(shutdown_result):
                cls.loop.run_until_complete(shutdown_result)

        finally:
            cls.loop.close()
            asyncio.set_event_loop(None)

    def test_01_graph_returns_synthesized_answer(self):
        self.assertTrue(
            self.answer.strip(),
            "The graph returned an empty synthesized answer.",
        )

        print("\n[PASS] Graph produced a synthesized answer:")
        print(self.answer)

    def test_02_graph_returns_real_retrieved_evidence(self):
        self.assertTrue(
            self.evidence,
            "No evidence was found in the graph's final state.",
        )

        print(
            f"\n[PASS] Graph returned {len(self.evidence)} evidence items."
        )

    def test_03_evidence_has_stable_document_ids(self):
        ids = [
            _document_id(item)
            for item in self.evidence
        ]

        missing_ids = [
            index
            for index, doc_id in enumerate(ids)
            if not doc_id
        ]

        self.assertFalse(
            missing_ids,
            f"Evidence items are missing stable document IDs "
            f"at indexes: {missing_ids}",
        )

        print(
            f"\n[PASS] All evidence items have document IDs: {ids}"
        )

    def test_04_evidence_preserves_task_provenance(self):
        provenance_found = False

        for item in self.evidence:
            data = _to_dict(item)
            metadata = data.get("metadata") or {}

            source_task_ids = (
                data.get("source_task_ids")
                or metadata.get("source_task_ids")
            )

            if source_task_ids:
                provenance_found = True
                break

        self.assertTrue(
            provenance_found,
            "No source_task_ids provenance was found in the retrieved "
            "evidence. Check the evidence merge output.",
        )

        print("\n[PASS] Evidence contains task provenance.")

    def test_05_grounding_validator_accepts_real_synthesis(self):
        result = self.loop.run_until_complete(
            _invoke_validator(
                question=QUESTION,
                answer=self.synthesized_answer,
                evidence=self.evidence,
            )
        )

        self.assertIsNotNone(
            result,
            "Grounding validator returned no result.",
        )

        print("\n[GROUNDING RESULT]")
        print(_to_dict(result) or result)

        status = _grounding_status(result)

        self.assertTrue(
            status or _to_dict(result),
            "Grounding validator did not return a recognizable "
            "structured result.",
        )

    def test_06_grounding_validator_detects_unsupported_content(self):
        result = self.loop.run_until_complete(
            _invoke_validator(
                question=QUESTION,
                answer=UNSUPPORTED_ANSWER,
                evidence=self.evidence,
            )
        )

        self.assertIsNotNone(
            result,
            "Grounding validator returned no result for unsupported text.",
        )

        status = _grounding_status(result)
        unsupported = _unsupported_claims(result)

        print("\n[UNSUPPORTED CONTENT RESULT]")
        print(_to_dict(result) or result)

        detected = (
            bool(unsupported)
            or status in {
                "unsupported",
                "partially_supported",
                "not_grounded",
                "ungrounded",
                "invalid",
                "failed",
                "rejected",
            }
        )

        self.assertTrue(
            detected,
            "The validator did not identify the fabricated John Smith "
            "claim as unsupported. Inspect the validator result above.",
        )

    def test_07_graph_persists_grounding_result(self):
        grounding_result = self.graph_result.get(
            "grounding_result"
        )

        self.assertIsNotNone(
            grounding_result,
            "Graph did not persist the grounding result.",
        )

        self.assertIn(
            grounding_result.get("status"),
            {
                "supported",
                "partially_supported",
                "unsupported",
                "insufficient_evidence",
                "validation_error",
            },
        )

        trace_steps = [
            item.get("step")
            for item in self.graph_result.get("trace", [])
        ]

        self.assertIn(
            "validate_final_answer_grounding",
            trace_steps,
        )

        print("\n[PASS] Grounding validation executed inside LangGraph.")
        print(
            "[GROUNDING STATUS]",
            grounding_result["status"],
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
