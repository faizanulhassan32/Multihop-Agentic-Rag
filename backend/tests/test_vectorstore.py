import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.config import Config

class TestVectorstoreIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        persist_directory = Path(Config.CHROMA_PERSIST_DIR).expanduser().resolve()
        if not persist_directory.is_dir():
            raise unittest.SkipTest(f"Chroma directory not found: {persist_directory}")

        cls.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        cls.client.get_collection(Config.CHROMA_COLLECTION_NAME)
        cls.vectorstore = Chroma(
            client=cls.client,
            embedding_function=OllamaEmbeddings(
                base_url=Config.OLLAMA_BASE_URL,
                model=Config.OLLAMA_EMBEDDING_MODEL,
            ),
            collection_name=Config.CHROMA_COLLECTION_NAME,
            create_collection_if_not_exists=False,
        )

    def test_collection_count_and_similarity_search(self):
        ids = self.vectorstore.get(include=[])["ids"]
        self.assertTrue(ids, "Configured Chroma collection is empty.")

        query = "What is the solubility of cadmium chloride in alcohol?"
        results = self.vectorstore.similarity_search(query, k=3)
        self.assertTrue(results, "Chroma similarity search returned no documents.")

        required_keys = {
            "title",
            "source_dataset",
            "source_config",
            "source_split",
            "sentence_count",
            "article_id",
        }
        missing = required_keys - results[0].metadata.keys()
        self.assertFalse(missing, f"Retrieved document is missing metadata: {sorted(missing)}")

        print(f"Collection contains {len(ids)} documents.")
        print(f"Top result: {results[0].metadata.get('title')}")
        print(f"Chunk ID: {results[0].metadata.get('article_id')}")


if __name__ == "__main__":
    unittest.main()
