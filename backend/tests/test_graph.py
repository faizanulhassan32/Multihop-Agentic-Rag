import asyncio
import json
import threading
import time
import unittest
from collections import deque
from unittest.mock import AsyncMock, patch

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from agent.classifier import ClassificationResult
from agent.graph import (
    EvidenceGap,
    EvidenceAssessment,
    RAGSystem,
    RecoveryPlan,
    build_answer_target,
    _ainvoke_with_model_timeout,
    graph,
    rag_system,
    supervisor_classify,
)
from ingestion.config import Config


class ScriptedLLM:
    def __init__(self, questions, assessments, plans=(), final_answer="answer"):
        self.questions = questions
        self.assessments = deque(assessments)
        self.plans = deque(plans)
        self.final_answer = final_answer
        self.generation_calls = 0
        self.assessment_calls = 0
        self.recovery_calls = 0
        self.assessment_prompts = []
        self.recovery_prompts = []
        self.final_prompts = []

    def __call__(self, prompt):
        self.generation_calls += 1
        prompt_text = prompt.to_string()
        if "Generate at most" in prompt_text:
            return json.dumps({"questions": self.questions})
        self.final_prompts.append(prompt_text)
        return self.final_answer

    def with_structured_output(self, schema, method):
        if method != "json_schema":
            raise AssertionError("Unexpected structured-output schema or method")

        if schema is EvidenceAssessment:
            def assess(_prompt):
                self.assessment_calls += 1
                self.assessment_prompts.append(_prompt.to_string())
                return self.assessments.popleft()

            return RunnableLambda(assess)
        if schema is RecoveryPlan:
            def plan(_prompt):
                self.recovery_calls += 1
                self.recovery_prompts.append(_prompt.to_string())
                return self.plans.popleft()

            return RunnableLambda(plan)
        raise AssertionError("Unexpected structured-output schema or method")


class GraphArchitectureTests(unittest.IsolatedAsyncioTestCase):
    async def execute_graph(self, llm, retrieve, question="test question"):
        classifier = ClassificationResult(classification="in_scope", reason="test")
        with (
            patch("agent.graph.arun_classifier", new_callable=AsyncMock, return_value=classifier) as classify,
            patch("agent.graph.get_generator_llm", return_value=llm),
            patch.object(rag_system, "retrieve_documents", new=retrieve),
        ):
            state = await graph.ainvoke({"original_question": question}, config={})
        return state, classify

    async def test_independent_queries_are_retrieved_concurrently(self):
        llm = ScriptedLLM(
            ["independent fact one", "independent fact two"],
            [EvidenceAssessment(relevant_document_ids=[0, 1, 2], sufficient=True)],
        )
        active = 0
        max_active = 0

        async def retrieve(query):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            return [Document(page_content=f"Evidence for {query}", metadata={"title": query})]

        state, classifier = await self.execute_graph(llm, retrieve)

        self.assertEqual(max_active, 3)
        self.assertEqual(state["final_answer"], "answer")
        self.assertEqual(state["followup_rounds"], 0)
        self.assertEqual(llm.assessment_calls, 1)
        self.assertEqual(llm.generation_calls, 2)
        self.assertEqual(classifier.await_count, 1)

    async def test_partially_supported_multi_hop_relationship_triggers_recovery(self):
        llm = ScriptedLLM(
            [],
            [
                EvidenceAssessment(
                    relevant_document_ids=[0, 1],
                    sufficient=False,
                    established_entities=["James Henry Miller", "Ewan MacColl"],
                    established_relationships=["James Henry Miller is also known as Ewan MacColl"],
                    gaps=[EvidenceGap(
                        missing_relationship="the target's spouse and the spouse's nationality",
                        retrieval_query="Who was Ewan MacColl's wife and what nationality was she?",
                    )],
                ),
                EvidenceAssessment(
                    relevant_document_ids=[0, 1, 2],
                    sufficient=True,
                    supported_relationship_ids=["requested-related-entity"],
                    answer_attribute_supported=True,
                    resolved_answer_entity="Peggy Seeger",
                    answer_value="American",
                ),
            ],
        )
        seen_queries = []

        async def retrieve(query):
            seen_queries.append(query)
            if "ewan maccoll" not in query.casefold():
                content = "James Henry Miller was also known as Ewan MacColl."
            else:
                content = "Peggy Seeger is an American folksinger and was married to Ewan MacColl."
            return [Document(page_content=content, metadata={"title": query})]

        state, _ = await self.execute_graph(
            llm,
            retrieve,
            question="What nationality was James Henry Miller's wife?",
        )

        self.assertEqual(
            seen_queries,
            [
                "What nationality was James Henry Miller's wife?",
                "What nationality was Ewan MacColl's wife?",
            ],
        )
        self.assertEqual(state["followup_rounds"], 1)
        self.assertTrue(state["evidence_sufficient"], json.dumps(state["trace"]))
        self.assertEqual(llm.assessment_calls, 2)
        self.assertEqual(llm.recovery_calls, 0)
        self.assertEqual(llm.generation_calls, 2)
        self.assertIn(
            "Retrieved for query: What nationality was Ewan MacColl's wife?",
            llm.assessment_prompts[-1],
        )
        self.assertIn("American folksinger", llm.assessment_prompts[-1])
        self.assertNotIn("the target's spouse", json.dumps(state["trace"]))

    async def test_insufficient_evidence_with_empty_queries_uses_fallback_planner(self):
        llm = ScriptedLLM(
            [],
            [
                EvidenceAssessment(relevant_document_ids=[0], sufficient=False),
                EvidenceAssessment(relevant_document_ids=[0, 1, 2], sufficient=True),
            ],
            plans=[RecoveryPlan(queries=["targeted missing fact query"])],
        )
        seen_queries = []

        async def retrieve(query):
            seen_queries.append(query)
            return [Document(page_content=f"Evidence for {query}", metadata={"title": query})]

        state, _ = await self.execute_graph(llm, retrieve)

        self.assertEqual(seen_queries, ["test question", "targeted missing fact query"])
        self.assertTrue(state["evidence_sufficient"])
        self.assertEqual(llm.recovery_calls, 1)
        self.assertEqual(llm.assessment_calls, 2)
        self.assertIn("Original question: test question", llm.recovery_prompts[0])
        self.assertIn("Previous retrieval queries:", llm.recovery_prompts[0])

    async def test_missing_question_relation_blocks_premature_sufficiency(self):
        question = "What nationality was Ewan MacColl's wife?"
        llm = ScriptedLLM(
            [],
            [
                EvidenceAssessment(
                    relevant_document_ids=[0],
                    sufficient=True,
                    established_relationships=[
                        "Ewan MacColl was married to James Henry Miller",
                    ],
                ),
                EvidenceAssessment(
                    relevant_document_ids=[0, 1],
                    sufficient=True,
                    supported_relationship_ids=["requested-related-entity"],
                    answer_attribute_supported=True,
                    resolved_answer_entity="Peggy Seeger",
                    answer_value="American",
                ),
            ],
            plans=[RecoveryPlan(queries=["Ewan MacColl spouse nationality biography"])],
        )
        seen_queries = []

        async def retrieve(query):
            seen_queries.append(query)
            if len(seen_queries) == 1:
                content = "Ewan MacColl was an English folk singer."
            else:
                content = "Peggy Seeger is an American folksinger and was married to Ewan MacColl."
            return [Document(page_content=content, metadata={"title": query})]

        state, _ = await self.execute_graph(llm, retrieve, question=question)

        self.assertEqual(
            seen_queries,
            [question, "Who was Ewan MacColl's wife?"],
        )
        self.assertTrue(state["evidence_sufficient"])
        self.assertEqual(state["followup_rounds"], 1)
        self.assertEqual(llm.recovery_calls, 0)

    async def test_irrelevant_documents_cannot_be_graded_sufficient(self):
        llm = ScriptedLLM(
            [],
            [
                EvidenceAssessment(relevant_document_ids=[], sufficient=True),
                EvidenceAssessment(relevant_document_ids=[0], sufficient=True),
            ],
            plans=[RecoveryPlan(queries=["query targeting the missing fact"])],
        )
        seen_queries = []

        async def retrieve(query):
            seen_queries.append(query)
            return [Document(page_content=f"Evidence for {query}", metadata={"title": query})]

        state, _ = await self.execute_graph(llm, retrieve)

        self.assertEqual(
            seen_queries,
            ["test question", "query targeting the missing fact"],
        )
        self.assertTrue(state["evidence_sufficient"])
        self.assertEqual(llm.recovery_calls, 1)

    async def test_q4_recovers_target_nationality_from_partial_evidence(self):
        question = "What nationality was James Henry Miller's wife?"
        llm = ScriptedLLM(
            [],
            [
                EvidenceAssessment(
                    relevant_document_ids=[0],
                    sufficient=False,
                    established_entities=["James Henry Miller", "Ewan MacColl"],
                    established_relationships=[
                        "James Henry Miller was also known as Ewan MacColl",
                    ],
                    gaps=[EvidenceGap(
                        missing_relationship="the requested person's spouse and nationality",
                        retrieval_query="",
                    )],
                ),
                EvidenceAssessment(
                    relevant_document_ids=[0, 1],
                    sufficient=True,
                    supported_relationship_ids=["requested-related-entity"],
                    answer_attribute_supported=True,
                    resolved_answer_entity="Peggy Seeger",
                    answer_value="American",
                ),
            ],
            plans=[RecoveryPlan(queries=[])],
            final_answer="American",
        )
        seen_queries = []

        async def retrieve(query):
            seen_queries.append(query)
            if "Ewan MacColl" in query:
                content = "Peggy Seeger was married to Ewan MacColl and is an American singer."
                return [Document(page_content=content, metadata={"title": "Peggy Seeger"})]
            return [
                Document(
                    page_content="June Miller was the second wife of Henry Miller.",
                    metadata={"title": "June Miller"},
                ),
                Document(
                    page_content="James Henry Miller was a former Australian rules footballer.",
                    metadata={"title": "Jim Miller"},
                ),
                Document(
                    page_content="James Henry Miller was better known by his stage name Ewan MacColl.",
                    metadata={"title": "Ewan MacColl"},
                ),
                Document(
                    page_content="Irma Haubold was married to Frank Haubold.",
                    metadata={"title": "Irma Haubold"},
                ),
            ]

        state, _ = await self.execute_graph(llm, retrieve, question=question)

        self.assertEqual(
            seen_queries,
            [
                question,
                "What nationality was Ewan MacColl's wife?",
            ],
        )
        self.assertEqual(state["final_answer"], "American")
        self.assertTrue(state["evidence_sufficient"])
        self.assertEqual(state["followup_rounds"], 1)
        self.assertEqual(llm.recovery_calls, 0)
        self.assertIn("Ewan MacColl", llm.assessment_prompts[0])
        self.assertNotIn("June Miller", llm.assessment_prompts[0])
        self.assertNotIn("Irma Haubold", llm.assessment_prompts[0])

    async def test_contract_query_is_not_repeated_by_semantically_duplicate_plan(self):
        question = "What nationality was Ewan MacColl's wife?"
        llm = ScriptedLLM(
            [],
            [
                EvidenceAssessment(
                    relevant_document_ids=[0],
                    sufficient=False,
                    gaps=[EvidenceGap(
                        missing_fact="the wife's nationality",
                        retrieval_query="Ewan MacColl wife nationality",
                    )],
                ),
                EvidenceAssessment(
                    relevant_document_ids=[0, 1],
                    sufficient=False,
                ),
            ],
            plans=[RecoveryPlan(queries=["What was the nationality of Ewan MacColl's spouse?"])],
        )
        seen_queries = []

        async def retrieve(query):
            seen_queries.append(query)
            return [Document(page_content="Partial evidence", metadata={"title": query})]

        state, _ = await self.execute_graph(llm, retrieve, question=question)

        self.assertEqual(seen_queries, [question, "Who was Ewan MacColl's wife?"])
        self.assertEqual(state["final_answer"], "I don't have enough information in my knowledge base to answer this question.")
        self.assertFalse(state["evidence_sufficient"])
        self.assertEqual(llm.recovery_calls, 0)
        self.assertEqual(len(seen_queries), len(set(seen_queries)))
        self.assertEqual(llm.generation_calls, 1)

    async def test_nonempty_but_insufficient_evidence_triggers_followup(self):
        llm = ScriptedLLM(
            ["bridge entity"],
            [
                EvidenceAssessment(
                    relevant_document_ids=[0, 1],
                    sufficient=False,
                    gaps=[EvidenceGap(
                        missing_fact="requested property",
                        retrieval_query="bridge entity requested property",
                    )],
                ),
                EvidenceAssessment(relevant_document_ids=[0, 1, 2], sufficient=True),
            ],
        )
        seen_queries = []

        async def retrieve(query):
            seen_queries.append(query)
            return [Document(page_content=f"Some evidence for {query}", metadata={"title": query})]

        state, _ = await self.execute_graph(llm, retrieve)

        self.assertEqual(len(seen_queries), 3)
        self.assertEqual(state["followup_rounds"], 1)
        self.assertEqual(llm.assessment_calls, 2)

    async def test_sufficient_evidence_skips_followup(self):
        llm = ScriptedLLM(
            ["answer fact"],
            [EvidenceAssessment(relevant_document_ids=[0, 1], sufficient=True)],
        )
        seen_queries = []

        async def retrieve(query):
            seen_queries.append(query)
            return [Document(page_content="Complete evidence", metadata={"title": "source"})]

        state, _ = await self.execute_graph(llm, retrieve)

        self.assertEqual(seen_queries, ["test question", "answer fact"])
        self.assertTrue(state["evidence_sufficient"])
        self.assertEqual(state["followup_rounds"], 0)
        self.assertEqual(llm.assessment_calls, 1)

    async def test_q3_contract_preserves_named_after_direction_and_recovers_answer(self):
        question = (
            "Musician and satirist Allie Goertz wrote a song about the 'The Simpsons' "
            "character Milhouse, who Matt Groening named after who?"
        )
        llm = ScriptedLLM(
            ["Who did Matt Groening name after Milhouse?"],
            [
                EvidenceAssessment(
                    relevant_document_ids=[0],
                    sufficient=False,
                    gaps=[EvidenceGap(
                        missing_relationship="the person whose name inspired Milhouse",
                        retrieval_query="Who did Matt Groening name after Milhouse?",
                    )],
                ),
                EvidenceAssessment(
                    relevant_document_ids=[0, 1],
                    sufficient=True,
                    supported_relationship_ids=["named-after-target"],
                    answer_attribute_supported=True,
                    resolved_answer_entity="President Richard Nixon",
                    answer_value="President Richard Nixon",
                ),
            ],
            final_answer="President Richard Nixon",
        )
        seen_queries = []

        async def retrieve(query):
            seen_queries.append(query)
            if "Milhouse after" in query:
                content = "Matt Groening named Milhouse after President Richard Nixon."
            else:
                content = "Milhouse is a character in The Simpsons."
            return [Document(page_content=content, metadata={"title": query})]

        target = build_answer_target(question)
        relationship = target.required_relationships[0]
        self.assertEqual(relationship.subject, "Milhouse")
        self.assertEqual(relationship.object, "$answer_entity")
        self.assertEqual(relationship.direction, "named entity to inspiration")
        self.assertEqual(
            relationship.retrieval_query,
            "Who did Matt Groening name Milhouse after?",
        )

        state, _ = await self.execute_graph(llm, retrieve, question=question)

        self.assertEqual(
            seen_queries,
            [question, "Who did Matt Groening name Milhouse after?"],
        )
        self.assertTrue(state["evidence_sufficient"])
        self.assertEqual(state["final_answer"], "President Richard Nixon")
        self.assertEqual(state["followup_rounds"], 1)
        self.assertEqual(
            state["answer_target"]["resolved_answer_entity"],
            "President Richard Nixon",
        )
        self.assertIn("named entity to inspiration", llm.final_prompts[0])
        self.assertIn("President Richard Nixon", llm.final_prompts[0])

    async def test_q4_answer_is_grounded_to_wife_not_intermediate_entity(self):
        question = "What nationality was James Henry Miller's wife?"
        self.assertEqual(
            build_answer_target(question).answer_entity,
            "wife of James Henry Miller",
        )
        self.assertEqual(
            build_answer_target(question).required_relationships[0].subject,
            "James Henry Miller",
        )

        async def run_with_final_answer(final_answer):
            llm = ScriptedLLM(
                [],
                [
                    EvidenceAssessment(
                        relevant_document_ids=[0],
                        sufficient=False,
                        established_entities=["James Henry Miller", "Ewan MacColl"],
                        established_relationships=[
                            "James Henry Miller was also known as Ewan MacColl",
                        ],
                    ),
                    EvidenceAssessment(
                        relevant_document_ids=[0, 1],
                        sufficient=True,
                        supported_relationship_ids=["requested-related-entity"],
                        answer_attribute_supported=True,
                        resolved_answer_entity="Peggy Seeger",
                        answer_value="American",
                    ),
                ],
                final_answer=final_answer,
            )

            async def retrieve(query):
                if query == question:
                    content = (
                        "James Henry Miller was better known by his stage name Ewan MacColl."
                    )
                    title = "Ewan MacColl"
                else:
                    content = (
                        "Peggy Seeger was married to Ewan MacColl. "
                        "Peggy Seeger was an American folksinger."
                    )
                    title = "Peggy Seeger"
                return [Document(page_content=content, metadata={"title": title})]

            return await self.execute_graph(llm, retrieve, question=question)

        state, _ = await run_with_final_answer("English")
        self.assertTrue(state["evidence_sufficient"], json.dumps(state["trace"]))
        self.assertEqual(
            state["final_answer"],
            "I couldn't verify a response for the answer target in the available evidence.",
        )

        correct_state, _ = await run_with_final_answer("American")
        self.assertTrue(correct_state["evidence_sufficient"])
        self.assertEqual(correct_state["final_answer"], "American")

    async def test_retriever_semaphore_honors_configured_concurrency(self):
        active = 0
        max_active = 0
        lock = threading.Lock()

        class BlockingRetriever:
            def invoke(self, query):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.02)
                with lock:
                    active -= 1
                return [Document(page_content=query, metadata={"title": query})]

        with patch.object(Config, "MAX_RETRIEVAL_CONCURRENCY", 2):
            retriever_system = RAGSystem()
        retriever_system._initialize_if_needed = lambda: None
        retriever_system._retriever = BlockingRetriever()

        results = await asyncio.gather(*(
            retriever_system.retrieve_documents(f"query {index}")
            for index in range(5)
        ))

        self.assertEqual(len(results), 5)
        self.assertEqual(max_active, 2)

    async def test_round_and_followup_query_limits_terminate_graph(self):
        llm = ScriptedLLM(
            ["initial query"],
            [
                EvidenceAssessment(
                    relevant_document_ids=[0, 1],
                    sufficient=False,
                    gaps=[
                        EvidenceGap(missing_fact="missing facts", retrieval_query="follow-up one"),
                        EvidenceGap(missing_fact="other missing fact", retrieval_query="follow-up two"),
                        EvidenceGap(missing_fact="third missing fact", retrieval_query="follow-up three"),
                    ],
                ),
                EvidenceAssessment(
                    relevant_document_ids=[0, 1, 2, 3],
                    sufficient=False,
                    gaps=[EvidenceGap(missing_fact="still missing", retrieval_query="another query")],
                ),
            ],
        )
        seen_queries = []

        async def retrieve(query):
            seen_queries.append(query)
            return [Document(page_content=f"Evidence for {query}", metadata={"title": query})]

        with (
            patch.object(Config, "MAX_FOLLOWUP_ROUNDS", 1),
            patch.object(Config, "MAX_FOLLOWUP_QUERIES", 2),
        ):
            state, _ = await self.execute_graph(llm, retrieve)

        self.assertEqual(
            seen_queries,
            ["test question", "initial query", "follow-up one", "follow-up two"],
        )
        self.assertEqual(state["followup_rounds"], 1)
        self.assertEqual(llm.assessment_calls, 2)
        self.assertEqual(
            state["final_answer"],
            "I don't have enough information in my knowledge base to answer this question.",
        )
        self.assertEqual(llm.generation_calls, 1)

    async def test_exhausted_followup_limit_returns_controlled_response(self):
        llm = ScriptedLLM(
            [],
            [EvidenceAssessment(
                relevant_document_ids=[0],
                sufficient=False,
                gaps=[EvidenceGap(
                    missing_fact="a required fact",
                    retrieval_query="targeted recovery query",
                )],
            )],
        )
        seen_queries = []

        async def retrieve(query):
            seen_queries.append(query)
            return [Document(page_content=f"Partial evidence for {query}", metadata={"title": query})]

        with patch.object(Config, "MAX_FOLLOWUP_ROUNDS", 0):
            state, _ = await self.execute_graph(llm, retrieve)

        self.assertEqual(seen_queries, ["test question"])
        self.assertEqual(state["final_answer"], "I don't have enough information in my knowledge base to answer this question.")
        self.assertFalse(state["evidence_sufficient"])
        self.assertEqual(llm.recovery_calls, 0)
        self.assertEqual(llm.generation_calls, 1)

    async def test_classifier_timeout_is_applied_and_does_not_block_loop(self):
        async def slow_classifier(_question):
            await asyncio.sleep(0.1)
            return ClassificationResult(classification="in_scope", reason="late")

        with (
            patch("agent.graph.arun_classifier", new=slow_classifier),
            patch.object(Config, "MODEL_TIMEOUT", 0.01),
        ):
            with self.assertRaises(TimeoutError):
                await supervisor_classify({"original_question": "test question"}, {})

    async def test_generator_invocation_timeout_is_applied(self):
        async def slow_invoke(_input):
            await asyncio.sleep(0.1)

        class SlowRunnable:
            ainvoke = staticmethod(slow_invoke)

        with patch.object(Config, "MODEL_TIMEOUT", 0.01):
            with self.assertRaises(TimeoutError):
                await _ainvoke_with_model_timeout(SlowRunnable(), {})


if __name__ == "__main__":
    unittest.main()
