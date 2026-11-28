
from __future__ import annotations

import importlib
import unittest
from unittest.mock import AsyncMock, patch

from agent.synthesis import (
    GroundingClaim,
    GroundingClaimDraft,
    GroundingDraft,
    GroundingResult,
    _result_from_draft,
)

graph_module = importlib.import_module("agent.graph")


class TestStage7GroundingIntegration(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.question = (
            "Which magazine was started first, Arthur's Magazine "
            "or First for Women?"
        )

        self.answer = "Arthur's Magazine"

        self.evidence = [
            {
                "doc_id": "doc_arthur",
                "content": "Arthur's Magazine began publication in 1844.",
                "metadata": {},
                "source_task_ids": ["task_001"],
                "retrieval_rounds": [0],
            }
        ]

    async def test_grounding_node_persists_result_in_graph_state(self):
        grounding_result = GroundingResult(
            status="supported",
            supported_claims=[
                GroundingClaim(
                    claim="Arthur's Magazine",
                    supporting_evidence_ids=["doc_arthur"],
                    reason="The evidence supports the answer.",
                )
            ],
            supporting_evidence_ids=["doc_arthur"],
            reason="The answer is supported.",
        )

        with patch.object(
            graph_module,
            "validate_answer_grounding",
            new=AsyncMock(return_value=grounding_result),
        ) as validator:

            update = await graph_module.validate_final_answer_grounding(
                {
                    "original_question": self.question,
                    "final_answer": self.answer,
                    "evidence": self.evidence,
                },
                {"metadata": {"request_id": "stage7-test"}},
            )

        validator.assert_awaited_once_with(
            answer=self.answer,
            evidence=self.evidence,
            question=self.question,
        )

        self.assertEqual(
            update["grounding_result"]["status"],
            "supported",
        )

        self.assertEqual(
            update["grounding_result"]["supporting_evidence_ids"],
            ["doc_arthur"],
        )

        self.assertEqual(
            update["trace"][0]["step"],
            "validate_final_answer_grounding",
        )

    def test_supported_answer_routes_directly_to_finalization(self):
        destination = graph_module.route_after_grounding_validation(
            {
                "grounding_result": {
                    "status": "supported",
                }
            }
        )

        self.assertEqual(destination, "supervisor_finalize")

    def test_partially_supported_answer_routes_to_policy(self):
        destination = graph_module.route_after_grounding_validation(
            {
                "grounding_result": {
                    "status": "partially_supported",
                }
            }
        )

        self.assertEqual(destination, "apply_grounding_policy")

    def test_unsupported_answer_routes_to_policy(self):
        destination = graph_module.route_after_grounding_validation(
            {
                "grounding_result": {
                    "status": "unsupported",
                }
            }
        )

        self.assertEqual(destination, "apply_grounding_policy")

    def test_validation_error_routes_to_policy(self):
        destination = graph_module.route_after_grounding_validation(
            {
                "grounding_result": {
                    "status": "validation_error",
                }
            }
        )

        self.assertEqual(destination, "apply_grounding_policy")

    async def test_partial_answer_preserves_only_supported_claims(self):
        update = await graph_module.apply_grounding_policy(
            {
                "grounding_result": {
                    "status": "partially_supported",
                    "supported_claims": [
                        {
                            "claim": (
                                "Arthur's Magazine began publication in 1844."
                            ),
                            "supporting_evidence_ids": [
                                "doc_arthur"
                            ],
                        }
                    ],
                    "unsupported_claims": [
                        {
                            "claim": (
                                "It was the most popular magazine of its time."
                            ),
                            "supporting_evidence_ids": [],
                        }
                    ],
                }
            },
            {"metadata": {"request_id": "stage7-test"}},
        )

        final_answer = update["final_answer"]

        self.assertIn(
            "Arthur's Magazine began publication in 1844.",
            final_answer,
        )

        self.assertIn(
            "[source:doc_arthur]",
            final_answer,
        )

        self.assertNotIn(
            "most popular",
            final_answer,
        )

    async def test_unsupported_answer_is_replaced_with_safe_response(self):
        update = await graph_module.apply_grounding_policy(
            {
                "grounding_result": {
                    "status": "unsupported",
                    "supported_claims": [],
                    "unsupported_claims": [
                        {
                            "claim": "The magazine was founded in 1800.",
                            "supporting_evidence_ids": [],
                        }
                    ],
                }
            },
            {},
        )

        self.assertIn(
            "couldn't verify a reliable answer",
            update["final_answer"],
        )

        self.assertNotIn(
            "founded in 1800",
            update["final_answer"],
        )

    async def test_validation_error_fails_closed(self):
        update = await graph_module.apply_grounding_policy(
            {
                "grounding_result": {
                    "status": "validation_error",
                    "supported_claims": [],
                    "unsupported_claims": [],
                }
            },
            {},
        )

        self.assertIn(
            "couldn't verify a reliable answer",
            update["final_answer"],
        )

    def test_validator_rejects_claims_not_present_in_answer(self):
        draft = GroundingDraft(
            claims=[
                GroundingClaimDraft(
                    claim=(
                        "Arthur's Magazine was an American literary periodical."
                    ),
                    supported=True,
                    supporting_evidence_ids=["doc_arthur"],
                    reason="The evidence supports this statement.",
                )
            ],
            overall_reason="The claim appears to be supported.",
        )

        result = _result_from_draft(
            draft=draft,
            available_ids={"doc_arthur"},
            answer="Arthur's Magazine",
        )

        self.assertEqual(
            result.status,
            "unsupported",
        )

        self.assertEqual(
            len(result.supported_claims),
            0,
        )

        self.assertEqual(
            result.supporting_evidence_ids,
            [],
        )

    def test_validator_accepts_verbatim_short_answer_claim(self):
        draft = GroundingDraft(
            claims=[
                GroundingClaimDraft(
                    claim="Arthur's Magazine",
                    supported=True,
                    supporting_evidence_ids=["doc_arthur"],
                    reason="The evidence supports the answer.",
                )
            ],
            overall_reason="The answer is supported.",
        )

        result = _result_from_draft(
            draft=draft,
            available_ids={"doc_arthur"},
            answer="Arthur's Magazine",
        )

        self.assertEqual(
            result.status,
            "supported",
        )

        self.assertEqual(
            result.supporting_evidence_ids,
            ["doc_arthur"],
        )

    def test_compiled_graph_contains_grounding_nodes(self):
        node_names = set(
            graph_module.graph.get_graph().nodes.keys()
        )

        self.assertIn(
            "validate_final_answer_grounding",
            node_names,
        )

        self.assertIn(
            "apply_grounding_policy",
            node_names,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
