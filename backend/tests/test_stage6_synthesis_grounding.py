from __future__ import annotations
import json
import unittest
from langchain_core.documents import Document
from agent.state import merge_evidence
from agent.models import (
    EvidenceRecord,
    TaskSpec,
)
from agent.retrieval import (
    _document_to_evidence,
)
from agent.synthesis import (
    AnswerSynthesis,
    format_evidence_context,
    parse_synthesis_response,
    validate_answer_grounding
)

class EvidenceAggregationTests(unittest.TestCase):

    def test_parallel_updates_preserve_all_task_ids(self):

        existing = [
            EvidenceRecord(
                doc_id="doc_shared",
                content="Canonical document content.",
                metadata={
                    "title": "Example source",
                },
                source_task_ids=["task_001"],
                retrieval_rounds=[0],
            ).model_dump(mode="json")
        ]

        updates = [
            EvidenceRecord(
                doc_id="doc_shared",
                content="Canonical document content.",
                metadata={
                    "url": "https://example.test/source",
                },
                source_task_ids=["task_002"],
                retrieval_rounds=[0],
            ).model_dump(mode="json"),

            EvidenceRecord(
                doc_id="doc_shared",
                content="",
                metadata={
                    "section": "History",
                },
                source_task_ids=[
                    "task_001_followup_1",
                ],
                retrieval_rounds=[1],
            ).model_dump(mode="json"),
        ]

        merged = merge_evidence(
            existing,
            updates,
        )

        self.assertEqual(len(merged), 1)

        self.assertEqual(
            merged[0]["source_task_ids"],
            [
                "task_001",
                "task_002",
                "task_001_followup_1",
            ],
        )

        self.assertEqual(
            merged[0]["retrieval_rounds"],
            [0, 1],
        )

        self.assertEqual(
            merged[0]["content"],
            "Canonical document content.",
        )

        self.assertEqual(
            merged[0]["metadata"],
            {
                "title": "Example source",
                "url": "https://example.test/source",
                "section": "History",
            },
        )

    def test_multiple_new_updates_for_same_document_are_merged(self):

        updates = [
            {
                "doc_id": "doc_1",
                "content": "Document content.",
                "metadata": {
                    "title": "Example",
                },
                "source_task_ids": ["task_a"],
                "retrieval_rounds": [0],
            },
            {
                "doc_id": "doc_1",
                "content": "Document content.",
                "metadata": {
                    "url": "https://example.test",
                },
                "source_task_ids": ["task_b"],
                "retrieval_rounds": [0],
            },
            {
                "doc_id": "doc_1",
                "content": "",
                "metadata": {
                    "section": "Background",
                },
                "source_task_ids": ["task_a_followup_1"],
                "retrieval_rounds": [1],
            },
        ]

        merged = merge_evidence(
            [],
            updates,
        )

        self.assertEqual(len(merged), 1)

        self.assertEqual(
            merged[0]["source_task_ids"],
            [
                "task_a",
                "task_b",
                "task_a_followup_1",
            ],
        )

        self.assertEqual(
            merged[0]["retrieval_rounds"],
            [0, 1],
        )

        self.assertEqual(
            merged[0]["metadata"],
            {
                "title": "Example",
                "url": "https://example.test",
                "section": "Background",
            },
        )

    def test_duplicate_existing_records_are_consolidated(self):

        merged = merge_evidence(
            [
                {
                    "doc_id": "doc_1",
                    "content": "Text",
                    "metadata": {},
                    "source_task_ids": ["task_a"],
                    "retrieval_rounds": [0],
                },
                {
                    "doc_id": "doc_1",
                    "content": "Text",
                    "metadata": {
                        "title": "Example",
                    },
                    "source_task_ids": ["task_b"],
                    "retrieval_rounds": [1],
                },
            ],
            [],
        )

        self.assertEqual(len(merged), 1)

        self.assertEqual(
            merged[0]["source_task_ids"],
            ["task_a", "task_b"],
        )

        self.assertEqual(
            merged[0]["retrieval_rounds"],
            [0, 1],
        )

        self.assertEqual(
            merged[0]["metadata"]["title"],
            "Example",
        )

    def test_missing_document_id_is_rejected(self):

        with self.assertRaises(ValueError):

            merge_evidence(
                [],
                [
                    {
                        "content": "Missing ID",
                    }
                ],
            )

    def test_existing_input_is_not_mutated(self):

        existing = [
            {
                "doc_id": "doc_1",
                "content": "Original",
                "metadata": {
                    "title": "Original title",
                },
                "source_task_ids": ["task_a"],
                "retrieval_rounds": [0],
            }
        ]

        original_snapshot = json.loads(
            json.dumps(existing)
        )

        merge_evidence(
            existing,
            [
                {
                    "doc_id": "doc_1",
                    "content": "Updated",
                    "metadata": {
                        "url": "https://example.test",
                    },
                    "source_task_ids": ["task_b"],
                    "retrieval_rounds": [1],
                }
            ],
        )

        self.assertEqual(
            existing,
            original_snapshot,
        )


class EvidenceProvenanceTests(unittest.TestCase):

    def test_evidence_record_preserves_retrieval_rounds(self):

        record = EvidenceRecord(
            doc_id="doc_1",
            content="Example content.",
            source_task_ids=[
                "task_001",
                "task_001_followup_1",
            ],
            retrieval_rounds=[0, 1, 1],
        )

        self.assertEqual(
            record.source_task_ids,
            [
                "task_001",
                "task_001_followup_1",
            ],
        )

        self.assertEqual(
            record.retrieval_rounds,
            [0, 1],
        )

    def test_negative_retrieval_round_is_rejected(self):

        with self.assertRaises(ValueError):

            EvidenceRecord(
                doc_id="doc_1",
                content="Example",
                retrieval_rounds=[-1],
            )

    def test_document_conversion_preserves_followup_round(self):

        document = Document(
            page_content="Example document content.",
            metadata={
                "id": "doc_123",
                "title": "Example document",
            },
        )

        evidence = _document_to_evidence(
            document=document,
            task_id="task_001_followup_1",
            retrieval_round=1,
        )

        self.assertEqual(
            evidence.doc_id,
            "doc_123",
        )

        self.assertEqual(
            evidence.source_task_ids,
            ["task_001_followup_1"],
        )

        self.assertEqual(
            evidence.retrieval_rounds,
            [1],
        )

        self.assertEqual(
            evidence.metadata["title"],
            "Example document",
        )

    def test_document_without_stable_id_is_rejected(self):

        document = Document(
            page_content="Example content.",
            metadata={
                "title": "Missing ID",
            },
        )

        with self.assertRaises(ValueError):

            _document_to_evidence(
                document,
                "task_001",
            )


class AnswerSynthesisTests(unittest.TestCase):

    def setUp(self):

        self.evidence = [
            EvidenceRecord(
                doc_id="doc_arthur",
                content=(
                    "Arthur's Magazine began publication in 1844."
                ),
                metadata={
                    "title": "Arthur's Magazine",
                    "source": "HotpotQA",
                },
                source_task_ids=["task_001"],
                retrieval_rounds=[0],
            ),

            EvidenceRecord(
                doc_id="doc_first",
                content=(
                    "First for Women was first published in 1989."
                ),
                metadata={
                    "title": "First for Women",
                },
                source_task_ids=[
                    "task_002_followup_1",
                ],
                retrieval_rounds=[1],
            ),
        ]

    def test_evidence_context_contains_stable_ids(self):

        context = format_evidence_context(
            self.evidence
        )

        self.assertIn(
            "Evidence ID: doc_arthur",
            context,
        )

        self.assertIn(
            "Evidence ID: doc_first",
            context,
        )

    def test_evidence_context_contains_source_metadata(self):

        context = format_evidence_context(
            self.evidence
        )

        self.assertIn(
            "title=Arthur's Magazine",
            context,
        )

        self.assertIn(
            "source=HotpotQA",
            context,
        )

    def test_evidence_context_contains_task_provenance(self):

        context = format_evidence_context(
            self.evidence
        )

        self.assertIn(
            "Source task IDs: task_001",
            context,
        )

        self.assertIn(
            "Source task IDs: task_002_followup_1",
            context,
        )

    def test_evidence_context_contains_retrieval_rounds(self):

        context = format_evidence_context(
            self.evidence
        )

        self.assertIn(
            "Retrieval rounds: 0",
            context,
        )

        self.assertIn(
            "Retrieval rounds: 1",
            context,
        )

    def test_structured_response_keeps_only_known_evidence_ids(self):

        raw = json.dumps(
            {
                "answer": (
                    "Arthur's Magazine began in 1844 "
                    "[source:doc_arthur]."
                ),
                "used_evidence_ids": [
                    "doc_arthur",
                    "doc_not_retrieved",
                ],
                "status": "structured",
                "warnings": [],
            }
        )

        result = parse_synthesis_response(
            raw_response=raw,
            available_evidence_ids={
                "doc_arthur",
                "doc_first",
            },
        )

        self.assertEqual(
            result.status,
            "structured",
        )

        self.assertEqual(
            result.used_evidence_ids,
            ["doc_arthur"],
        )

        self.assertTrue(
            any(
                "doc_not_retrieved" in warning
                for warning in result.warnings
            )
        )

    def test_unknown_inline_citation_is_flagged(self):

        raw = json.dumps(
            {
                "answer": (
                    "The answer is 1844 "
                    "[source:invented_doc]."
                ),
                "used_evidence_ids": [],
                "status": "structured",
                "warnings": [],
            }
        )

        result = parse_synthesis_response(
            raw,
            {"doc_arthur"},
        )

        self.assertTrue(
            any(
                "invented_doc" in warning
                for warning in result.warnings
            )
        )

    def test_plain_text_response_is_preserved_as_fallback(self):

        result = parse_synthesis_response(
            raw_response=(
                "Arthur's Magazine began publication in 1844."
            ),
            available_evidence_ids={
                "doc_arthur",
            },
        )

        self.assertEqual(
            result.status,
            "unstructured_fallback",
        )

        self.assertIn(
            "1844",
            result.answer,
        )

        self.assertEqual(
            result.used_evidence_ids,
            [],
        )

    def test_markdown_json_response_is_supported(self):

        raw = """```json
{
    "answer": "Arthur's Magazine began in 1844 [source:doc_arthur].",
    "used_evidence_ids": ["doc_arthur"],
    "status": "structured",
    "warnings": []
}
```"""

        result = parse_synthesis_response(
            raw,
            {"doc_arthur"},
        )

        self.assertEqual(
            result.status,
            "structured",
        )

        self.assertEqual(
            result.used_evidence_ids,
            ["doc_arthur"],
        )

    def test_synthesis_schema_rejects_unexpected_fields(self):

        with self.assertRaises(Exception):

            AnswerSynthesis.model_validate(
                {
                    "answer": "Answer",
                    "unexpected": True,
                }
            )


class FakeOllamaClient:

    def __init__(self, content: str):

        self.content = content

        self.calls = []

    async def chat(self, **kwargs):

        self.calls.append(kwargs)

        return {
            "message": {
                "content": self.content,
            }
        }


class GroundingValidationTests(
    unittest.IsolatedAsyncioTestCase
):

    def setUp(self):

        self.evidence = [
            EvidenceRecord(
                doc_id="doc_1",
                content=(
                    "Arthur's Magazine began publication in 1844."
                ),
                metadata={
                    "title": "Arthur's Magazine",
                },
                source_task_ids=["task_001"],
                retrieval_rounds=[0],
            )
        ]

    async def test_supported_claim_requires_valid_evidence_id(self):

        response = json.dumps(
            {
                "claims": [
                    {
                        "claim": (
                            "Arthur's Magazine began publication in 1844."
                        ),
                        "supported": True,
                        "supporting_evidence_ids": [
                            "doc_1",
                        ],
                        "reason": (
                            "The evidence directly states the publication year."
                        ),
                    }
                ],
                "overall_reason": (
                    "The factual claim is directly supported."
                ),
            }
        )

        client = FakeOllamaClient(response)

        result = await validate_answer_grounding(
            answer=(
                "Arthur's Magazine began publication in 1844."
            ),
            evidence=self.evidence,
            client=client,
        )

        self.assertEqual(
            result.status,
            "supported",
        )

        self.assertEqual(
            len(result.supported_claims),
            1,
        )

        self.assertEqual(
            result.supporting_evidence_ids,
            ["doc_1"],
        )

        self.assertEqual(
            len(client.calls),
            1,
        )

        self.assertIn(
            "format",
            client.calls[0],
        )

    async def test_unknown_evidence_id_cannot_support_claim(self):

        response = json.dumps(
            {
                "claims": [
                    {
                        "claim": (
                            "The magazine was founded in 1800."
                        ),
                        "supported": True,
                        "supporting_evidence_ids": [
                            "invented_doc",
                        ],
                        "reason": (
                            "The model asserted support."
                        ),
                    }
                ],
                "overall_reason": (
                    "One claim was assessed."
                ),
            }
        )

        result = await validate_answer_grounding(
            answer=(
                "The magazine was founded in 1800."
            ),
            evidence=self.evidence,
            client=FakeOllamaClient(response),
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
            len(result.unsupported_claims),
            1,
        )

        self.assertEqual(
            result.supporting_evidence_ids,
            [],
        )

        self.assertIn(
            "invented_doc",
            result.unsupported_claims[0].reason,
        )

    async def test_partially_supported_answer_is_identified(self):

        response = json.dumps(
            {
                "claims": [
                    {
                        "claim": (
                            "Arthur's Magazine began publication in 1844."
                        ),
                        "supported": True,
                        "supporting_evidence_ids": [
                            "doc_1",
                        ],
                        "reason": "Directly supported.",
                    },
                    {
                        "claim": (
                            "It was the most popular magazine of its time."
                        ),
                        "supported": False,
                        "supporting_evidence_ids": [],
                        "reason": (
                            "The supplied evidence does not establish popularity."
                        ),
                    },
                ],
                "overall_reason": (
                    "One claim is supported and one is not."
                ),
            }
        )

        result = await validate_answer_grounding(
            answer=(
                "Arthur's Magazine began publication in 1844. "
                "It was the most popular magazine of its time."
            ),
            evidence=self.evidence,
            client=FakeOllamaClient(response),
        )

        self.assertEqual(
            result.status,
            "partially_supported",
        )

        self.assertEqual(
            len(result.supported_claims),
            1,
        )

        self.assertEqual(
            len(result.unsupported_claims),
            1,
        )

    async def test_no_evidence_returns_without_model_call(self):

        client = FakeOllamaClient("{}")

        result = await validate_answer_grounding(
            answer="A factual answer.",
            evidence=[],
            client=client,
        )

        self.assertEqual(
            result.status,
            "insufficient_evidence",
        )

        self.assertEqual(
            client.calls,
            [],
        )

    async def test_empty_answer_returns_without_model_call(self):

        client = FakeOllamaClient("{}")

        result = await validate_answer_grounding(
            answer="",
            evidence=self.evidence,
            client=client,
        )

        self.assertEqual(
            result.status,
            "insufficient_evidence",
        )

        self.assertEqual(
            client.calls,
            [],
        )

    async def test_invalid_model_response_fails_closed(self):

        result = await validate_answer_grounding(
            answer="A factual answer.",
            evidence=self.evidence,
            client=FakeOllamaClient(
                "not valid JSON"
            ),
        )

        self.assertEqual(
            result.status,
            "validation_error",
        )

        self.assertIn(
            "could not be completed",
            result.reason,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)