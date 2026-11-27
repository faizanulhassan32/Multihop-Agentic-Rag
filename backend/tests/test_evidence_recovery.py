import unittest

from agent.evidence_assessor import assess_evidence
from agent.followup_planner import create_followup_tasks
from agent.orchestrator import route_after_followup
from agent.task_models import EvidenceRecord, TaskSpec
from agent.evidence_models import (
    EvidenceAssessment,
    TaskEvidenceAssessment,
)


class EvidenceAssessmentTests(unittest.TestCase):

    def test_missing_named_entity_does_not_force_insufficient(self):
        task = TaskSpec(
            task_id="task_001",
            question="Who directed Serendipity?",
            retrieval_query="Serendipity film director",
        )
        evidence = EvidenceRecord(
            doc_id="doc_001",
            content=(
                "Serendipity is a romantic comedy film directed "
                "by Peter Chelsom."
            ),
            source_task_ids=["task_001"],
        )

        result = assess_evidence([task], [evidence])

        self.assertEqual(result.status, "sufficient")
        self.assertEqual(result.gap_task_ids, [])

    def test_no_evidence_marks_task_as_gap(self):
        task = TaskSpec(
            task_id="task_001",
            question="Who directed Serendipity?",
            retrieval_query="Serendipity film director",
        )

        result = assess_evidence([task], [])

        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.gap_task_ids, ["task_001"])

    def test_followup_evidence_is_attributed_to_original_task(self):
        task = TaskSpec(
            task_id="task_001",
            question="Who directed Serendipity?",
            retrieval_query="Serendipity film director",
        )
        followup_evidence = EvidenceRecord(
            doc_id="doc_followup",
            content=(
                "Serendipity is a romantic comedy film directed "
                "by Peter Chelsom."
            ),
            source_task_ids=["task_001_followup_1"],
        )

        result = assess_evidence([task], [followup_evidence])

        self.assertEqual(result.status, "sufficient")
        self.assertEqual(result.task_assessments[0].evidence_ids, ["doc_followup"])


class FollowupPlannerTests(unittest.TestCase):

    def test_followup_query_uses_missing_terms(self):
        task = TaskSpec(
            task_id="task_001",
            question="Who directed Serendipity?",
            retrieval_query="Serendipity film",
            status="completed",
        )
        assessment = EvidenceAssessment(
            status="partial",
            coverage_score=0.5,
            task_assessments=[
                TaskEvidenceAssessment(
                    task_id="task_001",
                    status="partial",
                    coverage_score=0.5,
                    matched_terms=["serendipity"],
                    missing_terms=["directed", "peter", "chelsom"],
                    evidence_ids=["doc_001"],
                    reason="Partial evidence.",
                )
            ],
            gap_task_ids=["task_001"],
            evidence_count=1,
            assessment_method="lexical_coverage",
            reason="Evidence gap remains.",
        )

        followups = create_followup_tasks(
            tasks=[task],
            assessment=assessment,
            followup_round=0,
            max_followup_rounds=1,
            max_planned_tasks=4,
        )

        self.assertEqual(len(followups), 1)
        self.assertIn("directed", followups[0].retrieval_query)
        self.assertIn("peter", followups[0].retrieval_query)
        self.assertNotIn(
            "additional factual evidence",
            followups[0].retrieval_query,
        )

    def test_followup_respects_round_limit(self):
        task = TaskSpec(
            task_id="task_001",
            question="Who directed Serendipity?",
            retrieval_query="Serendipity film",
        )
        assessment = EvidenceAssessment(
            status="insufficient",
            coverage_score=0.0,
            task_assessments=[],
            gap_task_ids=["task_001"],
            evidence_count=0,
            reason="No evidence.",
        )

        followups = create_followup_tasks(
            tasks=[task],
            assessment=assessment,
            followup_round=1,
            max_followup_rounds=1,
            max_planned_tasks=4,
        )

        self.assertEqual(followups, [])


class FollowupRoutingTests(unittest.TestCase):

    def test_route_to_final_answer_when_no_followup_was_created(self):
        state = {
            "followup_tasks_created": False,
            "tasks": [],
        }

        self.assertEqual(
            route_after_followup(state),
            "generate_final_answer",
        )

    def test_route_to_retrieval_when_followup_is_pending(self):
        state = {
            "followup_tasks_created": True,
            "tasks": [
                {
                    "task_id": "task_001",
                    "question": "Original question",
                    "retrieval_query": "Original query",
                    "status": "completed",
                    "followup_round": 0,
                },
                {
                    "task_id": "task_001_followup_1",
                    "question": "Original question",
                    "retrieval_query": "Targeted query",
                    "status": "pending",
                    "followup_round": 1,
                },
            ],
        }

        self.assertEqual(
            route_after_followup(state),
            "prepare_retrieval_batch",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)