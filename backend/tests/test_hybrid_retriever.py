import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from langchain_core.documents import Document
from agent import retrieval
from agent.models import TaskSpec

def document(document_id, content=None, **metadata):
    return Document(
        page_content=content or f"content for {document_id}",
        metadata={"id": document_id, **metadata},
    )


class TestReciprocalRankFusion(unittest.TestCase):
    def test_rrf_ranks_overlap_and_deduplicates_by_stable_id(self):
        dense = [document("a"), document("b")]
        bm25 = [document("b", "sparse b"), document("c")]

        results = retrieval.reciprocal_rank_fusion(
            dense, bm25, result_limit=3, rrf_constant=60
        )

        self.assertEqual([item.metadata["id"] for item in results], ["b", "a", "c"])
        self.assertEqual(
            sum(item.metadata["id"] == "b" for item in results),
            1,
        )
        self.assertEqual(results[0].page_content, dense[1].page_content)

    def test_rrf_handles_one_empty_backend_and_requires_stable_ids(self):
        self.assertEqual(
            [
                item.metadata["id"]
                for item in retrieval.reciprocal_rank_fusion(
                    [], [document("sparse-only")], result_limit=5
                )
            ],
            ["sparse-only"],
        )
        self.assertEqual(
            [
                item.metadata["id"]
                for item in retrieval.reciprocal_rank_fusion(
                    [document("dense-only")], [], result_limit=5
                )
            ],
            ["dense-only"],
        )
        self.assertEqual(
            retrieval.reciprocal_rank_fusion([], [], result_limit=5), []
        )
        with self.assertRaisesRegex(ValueError, "stable string ID"):
            retrieval.reciprocal_rank_fusion(
                [Document(page_content="missing id")], [], result_limit=1
            )

    def test_rrf_respects_final_limit(self):
        results = retrieval.reciprocal_rank_fusion(
            [document("a"), document("b"), document("c")],
            [document("d"), document("e")],
            result_limit=2,
        )
        self.assertEqual(len(results), 2)


class TestHybridSearch(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vectorstore = object()
        self.db_path = Path(self.temp_dir.name) / "bm25.sqlite3"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parallel_retrieval_and_candidate_limits(self):
        barrier = threading.Barrier(2)
        dense = [document("dense")]
        sparse = [document("sparse")]

        def wait_for_peer(result):
            barrier.wait(timeout=5)
            return result

        with patch.object(
            retrieval,
            "_retrieve_dense",
            side_effect=lambda *args: wait_for_peer(dense),
        ) as dense_mock, patch.object(
            retrieval,
            "_retrieve_bm25",
            side_effect=lambda *args: wait_for_peer(sparse),
        ) as bm25_mock:
            results = asyncio.run(
                retrieval.hybrid_search(
                    "query",
                    self.vectorstore,
                    dense_limit=7,
                    bm25_limit=9,
                    result_limit=1,
                    rrf_constant=60,
                    bm25_db_path=self.db_path,
                )
            )

        dense_mock.assert_called_once_with(
            self.vectorstore,
            "query",
            7,
            "unknown",
        )
        bm25_mock.assert_called_once_with(
            "query",
            9,
            self.db_path,
            "unknown",
        )
        self.assertEqual(len(results), 1)

    def test_single_backend_failures_use_the_other_backend(self):
        dense = [document("dense-only")]
        sparse = [document("sparse-only")]
        with patch.object(
            retrieval, "_retrieve_dense", side_effect=RuntimeError("Chroma down")
        ), patch.object(retrieval, "_retrieve_bm25", return_value=sparse):
            with self.assertLogs(retrieval.logger, level="WARNING"):
                results = asyncio.run(
                    retrieval.hybrid_search(
                        "query", self.vectorstore, 2, 2, 5, 60, self.db_path
                    )
                )
        self.assertEqual([item.metadata["id"] for item in results], ["sparse-only"])

        with patch.object(retrieval, "_retrieve_dense", return_value=dense), patch.object(
            retrieval,
            "_retrieve_bm25",
            side_effect=RuntimeError("SQLite unavailable"),
        ):
            with self.assertLogs(retrieval.logger, level="WARNING"):
                results = asyncio.run(
                    retrieval.hybrid_search(
                        "query", self.vectorstore, 2, 2, 5, 60, self.db_path
                    )
                )
        self.assertEqual([item.metadata["id"] for item in results], ["dense-only"])

    def test_empty_backend_results_use_the_nonempty_results(self):
        dense = [document("dense-only")]
        sparse = [document("sparse-only")]
        with patch.object(
            retrieval, "_retrieve_dense", return_value=[]
        ), patch.object(retrieval, "_retrieve_bm25", return_value=sparse):
            results = asyncio.run(
                retrieval.hybrid_search(
                    "query", self.vectorstore, 2, 2, 5, 60, self.db_path
                )
            )
        self.assertEqual([item.metadata["id"] for item in results], ["sparse-only"])

        with patch.object(
            retrieval, "_retrieve_dense", return_value=dense
        ), patch.object(retrieval, "_retrieve_bm25", return_value=[]):
            results = asyncio.run(
                retrieval.hybrid_search(
                    "query", self.vectorstore, 2, 2, 5, 60, self.db_path
                )
            )
        self.assertEqual([item.metadata["id"] for item in results], ["dense-only"])

    def test_both_backend_failures_raise_informative_error(self):
        with patch.object(
            retrieval, "_retrieve_dense", side_effect=RuntimeError("Chroma down")
        ), patch.object(
            retrieval,
            "_retrieve_bm25",
            side_effect=FileNotFoundError("BM25 missing"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "both backends.*Chroma down.*BM25 missing"
            ):
                asyncio.run(
                    retrieval.hybrid_search(
                        "query", self.vectorstore, 2, 2, 5, 60, self.db_path
                    )
                )

    def test_dense_and_bm25_documents_expose_the_same_stable_id(self):
        class FakeEmbeddings:
            def embed_query(self, query):
                return [0.25, 0.75]

        class FakeCollection:
            def query(self, **kwargs):
                self.kwargs = kwargs
                return {
                    "ids": [["chunk-a"]],
                    "documents": [["same content"]],
                    "metadatas": [[{"title": "Test"}]],
                }

        class FakeVectorStore:
            _embedding_function = FakeEmbeddings()
            _collection = FakeCollection()

        dense_result = retrieval._retrieve_dense(
            FakeVectorStore(), "query", limit=4
        )
        self.assertEqual(dense_result[0].metadata["id"], "chunk-a")
        self.assertEqual(FakeVectorStore._collection.kwargs["n_results"], 4)

        with patch.object(
            retrieval,
            "search_bm25",
            return_value=[
                {
                    "id": "chunk-a",
                    "content": "same content",
                    "metadata": {"title": "Test"},
                    "score": -1.0,
                }
            ],
        ):
            bm25_result = retrieval._retrieve_bm25(
                "query", 4, self.db_path
            )
            retrieval.search_bm25.assert_called_once_with(
                "query", limit=4, db_path=self.db_path
            )
        self.assertEqual(bm25_result[0].metadata["id"], "chunk-a")
        self.assertEqual(bm25_result[0].metadata["title"], "Test")


class TestTaskExecutorIntegration(unittest.TestCase):
    def test_task_executor_preserves_retrieved_documents_as_evidence(self):
        docs = [
            document(
                "hybrid-a",
                "hybrid answer context",
                source="test",
            )
        ]

        task = TaskSpec(
            task_id="task-1",
            question="question",
            retrieval_query="search query",
        )

        with patch.object(
            retrieval.rag_system,
            "retrieve_documents",
            new=AsyncMock(return_value=docs),
        ) as retrieve_mock:

            result, evidence = asyncio.run(
                retrieval.execute_retrieval_task(
                    task=task,
                    request_id="test-request",
                )
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.evidence_ids, ["hybrid-a"])

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].doc_id, "hybrid-a")
        self.assertEqual(
            evidence[0].content,
            "hybrid answer context",
        )
        self.assertEqual(
            evidence[0].source_task_ids,
            ["task-1"],
        )

        retrieve_mock.assert_awaited_once_with(
            query="search query",
            request_id="test-request",
        )


if __name__ == "__main__":
    unittest.main()
