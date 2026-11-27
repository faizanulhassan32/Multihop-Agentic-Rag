import hashlib
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_chroma import Chroma

from agent.hybrid_retriever import _retrieve_dense
from src.ingestion import build_vectorstore as vectorstore_module
from src.ingestion.build_bm25_index import build_bm25_index, get_bm25_ids, search_bm25
from config import Config

class DeterministicEmbeddings:
    def _embed(self, text):
        vector = [0.0] * 128
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % len(vector)] += 1.0
        return vector

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


class TestIndexSynchronization(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.collection_name = f"audit_{self.root.name.replace('-', '_')}"
        self.db_path = self.root / "bm25.sqlite3"
        self.embeddings = DeterministicEmbeddings()

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _chunk(chunk_id, title, content, **metadata):
        return {
            "id": chunk_id,
            "title": title,
            "content": content,
            "metadata": {"source": "isolated-test", **metadata},
        }

    def test_repeated_and_mutating_sync_keeps_both_indexes_aligned(self):
        initial = [
            self._chunk("audit_maple_chunk_0", "Maple", "maple amber leaves"),
            self._chunk("audit_glacier_chunk_0", "Glacier", "glacier melt alpine ice"),
        ]
        updated = [
            self._chunk(
                "audit_maple_chunk_0",
                "Maple",
                "maple volcanic amber ash",
                revision=2,
            ),
            self._chunk("audit_volcano_chunk_0", "Volcano", "eruption plume volcanic ash"),
        ]

        with patch.object(Config, "CHROMA_PERSIST_DIR", str(self.root / "chroma")), \
             patch.object(Config, "CHROMA_COLLECTION_NAME", self.collection_name), \
             patch.object(
                 vectorstore_module,
                 "OllamaEmbeddings",
                 return_value=self.embeddings,
             ):
            expected_ids = {chunk["id"] for chunk in initial}
            self.assertEqual(
                vectorstore_module.build_vectorstore(chunks=initial), expected_ids
            )
            self.assertEqual(build_bm25_index(initial, self.db_path), expected_ids)

            self.assertEqual(
                vectorstore_module.build_vectorstore(chunks=initial, force=True),
                expected_ids,
            )
            self.assertEqual(build_bm25_index(initial, self.db_path), expected_ids)

            expected_updated_ids = {chunk["id"] for chunk in updated}
            self.assertEqual(
                vectorstore_module.build_vectorstore(chunks=updated, force=True),
                expected_updated_ids,
            )
            self.assertEqual(
                build_bm25_index(updated, self.db_path), expected_updated_ids
            )

            store = Chroma(
                persist_directory=str(self.root / "chroma"),
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            chroma_ids = set(store._collection.get(include=[])["ids"])
            self.assertEqual(chroma_ids, expected_updated_ids)
            self.assertEqual(get_bm25_ids(self.db_path), expected_updated_ids)

            dense_results = _retrieve_dense(store, "eruption plume", limit=1)
            self.assertEqual(dense_results[0].page_content, updated[1]["content"])
            self.assertEqual(
                dense_results[0].metadata["id"], "audit_volcano_chunk_0"
            )
            self.assertEqual(dense_results[0].metadata["title"], "Volcano")
            self.assertEqual(dense_results[0].metadata["source"], "isolated-test")
            sparse_results = search_bm25("eruption plume", db_path=self.db_path)
            self.assertEqual(sparse_results[0]["id"], "audit_volcano_chunk_0")

    def test_force_does_not_delete_legacy_chroma_records(self):
        with patch.object(Config, "CHROMA_PERSIST_DIR", str(self.root / "chroma")), \
             patch.object(Config, "CHROMA_COLLECTION_NAME", self.collection_name), \
             patch.object(
                 vectorstore_module,
                 "OllamaEmbeddings",
                 return_value=self.embeddings,
             ):
            store = Chroma(
                persist_directory=str(self.root / "chroma"),
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            store.add_texts(
                ["legacy record that must remain"],
                ids=["legacy-auto-id"],
                metadatas=[{"source": "legacy"}],
            )
            before = store._collection.get(include=["documents", "metadatas"])

            with self.assertRaisesRegex(RuntimeError, "no data was changed"):
                vectorstore_module.build_vectorstore(
                    force=True,
                    chunks=[self._chunk("audit_new_chunk_0", "New", "new record")],
                )

            after = store._collection.get(include=["documents", "metadatas"])
            self.assertEqual(after["ids"], before["ids"])
            self.assertEqual(after["documents"], before["documents"])
            self.assertEqual(after["metadatas"], before["metadatas"])


if __name__ == "__main__":
    unittest.main()
