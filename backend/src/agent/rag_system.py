import logging

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import Config
from services.reranker_client import RerankerClient
from agent.hybrid_retriever import hybrid_search

logger = logging.getLogger(__name__)


class RAGSystem:
    """Owns the vector store, embeddings, reranker and retrieval lifecycle."""

    def __init__(self):
        self._vectorstore = None
        self._reranker_client = None

    def startup(self):
        """Initialize the reranker client."""
        if self._reranker_client is None:
            self._reranker_client = RerankerClient()

    async def shutdown(self):
        """Close the reranker client."""
        if self._reranker_client is not None:
            await self._reranker_client.aclose()
            self._reranker_client = None

    def _initialize_if_needed(self):
        """Initialize the persistent Chroma vector store."""
        if self._vectorstore is not None:
            return

        embeddings = OllamaEmbeddings(
            model=Config.OLLAMA_EMBEDDING_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
        )

        self._vectorstore = Chroma(
            collection_name=Config.CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR,
        )

    async def retrieve_documents(
        self,
        query: str,
        request_id: str = "unknown",
    ):
        """Retrieve and rerank documents using hybrid retrieval."""
        self._initialize_if_needed()

        if self._reranker_client is None:
            raise RuntimeError(
                "Reranker client is not initialized. "
                "Call rag_system.startup() before retrieval."
            )

        return await hybrid_search(
            query=query,
            vectorstore=self._vectorstore,
            dense_limit=Config.DENSE_RETRIEVAL_K,
            bm25_limit=Config.BM25_RETRIEVAL_K,
            result_limit=20,
            rrf_constant=Config.RRF_K,
            bm25_db_path=Config.BM25_DB_PATH,
            reranker_client=self._reranker_client,
            rerank_limit=Config.TASK_RERANK_LIMIT,
            request_id=request_id,
        )


rag_system = RAGSystem()