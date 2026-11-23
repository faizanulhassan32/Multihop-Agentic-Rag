import asyncio
import importlib
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langchain_core.documents import Document

from agent import hybrid_retriever

graph_module = importlib.import_module("agent.graph")


def document(document_id, content=None, **metadata):
    return Document(
        page_content=content or f"content for {document_id}",
        metadata={"id": document_id, **metadata},
    )


class TestReciprocalRankFusion(unittest.TestCase):
    def test_rrf_ranks_overlap_and_deduplicates_by_stable_id(self):
        dense = [document("a"), document("b")]
        bm25 = [document("b", "sparse b"), document("c")]

        results = hybrid_retriever.reciprocal_rank_fusion(
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
                for item in hybrid_retriever.reciprocal_rank_fusion(
                    [], [document("sparse-only")], result_limit=5
                )
            ],
            ["sparse-only"],
        )
        self.assertEqual(
            [
                item.metadata["id"]
                for item in hybrid_retriever.reciprocal_rank_fusion(
                    [document("dense-only")], [], result_limit=5
                )
            ],
            ["dense-only"],
        )
        self.assertEqual(
            hybrid_retriever.reciprocal_rank_fusion([], [], result_limit=5), []
        )
        with self.assertRaisesRegex(ValueError, "stable string ID"):
            hybrid_retriever.reciprocal_rank_fusion(
                [Document(page_content="missing id")], [], result_limit=1
            )

    def test_rrf_respects_final_limit(self):
        results = hybrid_retriever.reciprocal_rank_fusion(
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
            hybrid_retriever,
            "_retrieve_dense",
            side_effect=lambda *args: wait_for_peer(dense),
        ) as dense_mock, patch.object(
            hybrid_retriever,
            "_retrieve_bm25",
            side_effect=lambda *args: wait_for_peer(sparse),
        ) as bm25_mock:
            results = asyncio.run(
                hybrid_retriever.hybrid_search(
                    "query",
                    self.vectorstore,
                    dense_limit=7,
                    bm25_limit=9,
                    result_limit=1,
                    rrf_constant=60,
                    bm25_db_path=self.db_path,
                )
            )

        dense_mock.assert_called_once_with(self.vectorstore, "query", 7)
        bm25_mock.assert_called_once_with("query", 9, self.db_path)
        self.assertEqual(len(results), 1)

    def test_single_backend_failures_use_the_other_backend(self):
        dense = [document("dense-only")]
        sparse = [document("sparse-only")]
        with patch.object(
            hybrid_retriever, "_retrieve_dense", side_effect=RuntimeError("Chroma down")
        ), patch.object(hybrid_retriever, "_retrieve_bm25", return_value=sparse):
            with self.assertLogs(hybrid_retriever.logger, level="WARNING"):
                results = asyncio.run(
                    hybrid_retriever.hybrid_search(
                        "query", self.vectorstore, 2, 2, 5, 60, self.db_path
                    )
                )
        self.assertEqual([item.metadata["id"] for item in results], ["sparse-only"])

        with patch.object(hybrid_retriever, "_retrieve_dense", return_value=dense), patch.object(
            hybrid_retriever,
            "_retrieve_bm25",
            side_effect=RuntimeError("SQLite unavailable"),
        ):
            with self.assertLogs(hybrid_retriever.logger, level="WARNING"):
                results = asyncio.run(
                    hybrid_retriever.hybrid_search(
                        "query", self.vectorstore, 2, 2, 5, 60, self.db_path
                    )
                )
        self.assertEqual([item.metadata["id"] for item in results], ["dense-only"])

    def test_empty_backend_results_use_the_nonempty_results(self):
        dense = [document("dense-only")]
        sparse = [document("sparse-only")]
        with patch.object(
            hybrid_retriever, "_retrieve_dense", return_value=[]
        ), patch.object(hybrid_retriever, "_retrieve_bm25", return_value=sparse):
            results = asyncio.run(
                hybrid_retriever.hybrid_search(
                    "query", self.vectorstore, 2, 2, 5, 60, self.db_path
                )
            )
        self.assertEqual([item.metadata["id"] for item in results], ["sparse-only"])

        with patch.object(
            hybrid_retriever, "_retrieve_dense", return_value=dense
        ), patch.object(hybrid_retriever, "_retrieve_bm25", return_value=[]):
            results = asyncio.run(
                hybrid_retriever.hybrid_search(
                    "query", self.vectorstore, 2, 2, 5, 60, self.db_path
                )
            )
        self.assertEqual([item.metadata["id"] for item in results], ["dense-only"])

    def test_both_backend_failures_raise_informative_error(self):
        with patch.object(
            hybrid_retriever, "_retrieve_dense", side_effect=RuntimeError("Chroma down")
        ), patch.object(
            hybrid_retriever,
            "_retrieve_bm25",
            side_effect=FileNotFoundError("BM25 missing"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "both backends.*Chroma down.*BM25 missing"
            ):
                asyncio.run(
                    hybrid_retriever.hybrid_search(
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

        dense_result = hybrid_retriever._retrieve_dense(
            FakeVectorStore(), "query", limit=4
        )
        self.assertEqual(dense_result[0].metadata["id"], "chunk-a")
        self.assertEqual(FakeVectorStore._collection.kwargs["n_results"], 4)

        with patch.object(
            hybrid_retriever,
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
            bm25_result = hybrid_retriever._retrieve_bm25(
                "query", 4, self.db_path
            )
            hybrid_retriever.search_bm25.assert_called_once_with(
                "query", limit=4, db_path=self.db_path
            )
        self.assertEqual(bm25_result[0].metadata["id"], "chunk-a")
        self.assertEqual(bm25_result[0].metadata["title"], "Test")


class TestGraphHybridIntegration(unittest.TestCase):
    def test_graph_retrieval_node_uses_hybrid_documents(self):
        docs = [document("hybrid-a", "hybrid answer context", source="test")]
        state = {
            "decomposed_questions": ["question"],
            "rewritten_query": "search query",
            "current_question_index": 0,
        }
        vectorstore = object()

        with patch.object(
            graph_module.rag_system, "_initialize_if_needed", return_value=None
        ), patch.object(
            graph_module.rag_system, "_vectorstore", vectorstore
        ), patch.object(
            graph_module, "hybrid_search", new_callable=AsyncMock, return_value=docs
        ) as hybrid_mock:
            result = asyncio.run(graph_module.retrieve_documents(state, {}))

        self.assertEqual(result["retrieved_docs"], docs)
        hybrid_mock.assert_awaited_once_with(
            query="search query",
            vectorstore=vectorstore,
            dense_limit=graph_module.Config.DENSE_RETRIEVAL_K,
            bm25_limit=graph_module.Config.BM25_RETRIEVAL_K,
            result_limit=graph_module.Config.HYBRID_RETRIEVAL_K,
            rrf_constant=graph_module.Config.RRF_K,
            bm25_db_path=Path(graph_module.Config.BM25_DB_PATH),
        )


if __name__ == "__main__":
    unittest.main()
