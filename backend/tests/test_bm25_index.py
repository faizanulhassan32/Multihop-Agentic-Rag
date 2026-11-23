import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.ingestion.build_bm25_index import (
    build_bm25_index,
    get_bm25_ids,
    search_bm25,
)


class TestBM25Index(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "bm25" / "index.sqlite3"

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def chunk(chunk_id, content, title="Example", **metadata):
        return {
            "id": chunk_id,
            "title": title,
            "content": content,
            "metadata": metadata,
        }

    def test_keyword_retrieval_persists_stable_ids_and_metadata(self):
        chunks = [
            self.chunk(
                "article_hash_chunk_0",
                "Quasars emit bright radio signals.",
                title="Astronomy",
                source="test",
            ),
            self.chunk("article_hash_chunk_20", "Ocean currents move heat."),
        ]

        self.assertEqual(
            build_bm25_index(chunks, self.db_path),
            {"article_hash_chunk_0", "article_hash_chunk_20"},
        )
        results = search_bm25("quasars radio", db_path=self.db_path)

        self.assertEqual([result["id"] for result in results], ["article_hash_chunk_0"])
        self.assertEqual(results[0]["content"], chunks[0]["content"])
        self.assertEqual(
            results[0]["metadata"],
            {"title": "Astronomy", "source": "test"},
        )
        self.assertEqual(get_bm25_ids(self.db_path), {
            "article_hash_chunk_0",
            "article_hash_chunk_20",
        })

    def test_repeated_sync_is_idempotent_and_updates_adds_and_deletes(self):
        original = [
            self.chunk("stable_a", "The old fox crossed the valley."),
            self.chunk("stable_b", "A removable record contains obsolete words."),
        ]
        expected_ids = {"stable_a", "stable_b"}

        self.assertEqual(build_bm25_index(original, self.db_path), expected_ids)
        self.assertEqual(build_bm25_index(original, self.db_path), expected_ids)

        updated = [
            self.chunk("stable_a", "The new fox crossed the mountain.", revision=2),
            self.chunk("stable_c", "Added entry discusses glaciers."),
        ]
        self.assertEqual(
            build_bm25_index(updated, self.db_path), {"stable_a", "stable_c"}
        )
        self.assertEqual(get_bm25_ids(self.db_path), {"stable_a", "stable_c"})
        self.assertEqual(search_bm25("obsolete", db_path=self.db_path), [])
        self.assertEqual(
            [result["id"] for result in search_bm25("mountain", db_path=self.db_path)],
            ["stable_a"],
        )
        self.assertEqual(
            search_bm25("fox", db_path=self.db_path)[0]["metadata"]["revision"], 2
        )

        with sqlite3.connect(self.db_path) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM bm25_documents").fetchone()[0],
                2,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM bm25_fts").fetchone()[0],
                2,
            )

    def test_rare_term_partial_query_and_empty_or_unmatched_queries(self):
        chunks = [
            self.chunk(
                "rare_term_chunk_0",
                "A xylophagous beetle consumes wood fibers.",
            ),
            self.chunk("common_term_chunk_0", "A beetle crawled across a leaf."),
        ]
        build_bm25_index(chunks, self.db_path)

        self.assertEqual(
            [result["id"] for result in search_bm25("xylophagous", db_path=self.db_path)],
            ["rare_term_chunk_0"],
        )
        self.assertEqual(
            [result["id"] for result in search_bm25(
                "xylophagous unrelatedword", db_path=self.db_path
            )],
            ["rare_term_chunk_0"],
        )
        self.assertEqual(search_bm25("", db_path=self.db_path), [])
        self.assertEqual(search_bm25("!?...", db_path=self.db_path), [])
        self.assertEqual(search_bm25("unrelatedword", db_path=self.db_path), [])


if __name__ == "__main__":
    unittest.main()
