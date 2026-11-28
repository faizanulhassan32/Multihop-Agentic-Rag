import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

from langchain_core.documents import Document

from ingestion.build_bm25_index import search_bm25
from services.reranker_client import RerankerClient

from config import Config
from agent.models import EvidenceRecord, TaskResult, TaskSpec


logger = logging.getLogger(__name__)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _document_id(document: Document) -> str:
    document_id = document.metadata.get("id")
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("Hybrid retrieval requires a stable string ID in document metadata.")
    return document_id


def reciprocal_rank_fusion(
    dense_results: Sequence[Document],
    bm25_results: Sequence[Document],
    result_limit: int,
    rrf_constant: int = 60,
) -> List[Document]:
    """Fuse ranked documents by stable ID using reciprocal rank, not raw scores."""
    if result_limit < 0:
        raise ValueError("result_limit must not be negative.")
    if rrf_constant <= 0:
        raise ValueError("rrf_constant must be greater than zero.")
    if result_limit == 0:
        return []

    scores = {}
    documents = {}
    for results in (dense_results, bm25_results):
        seen_ids = set()
        for rank, document in enumerate(results, start=1):
            document_id = _document_id(document)
            if document_id in seen_ids:
                continue
            seen_ids.add(document_id)
            scores[document_id] = scores.get(document_id, 0.0) + (
                1.0 / (rrf_constant + rank)
            )
            documents.setdefault(document_id, document)

    ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)
    return [documents[document_id] for document_id in ranked_ids[:result_limit]]


def _retrieve_dense(
    vectorstore, query: str, limit: int, request_id: str = "unknown"
) -> List[Document]:
    """Query Chroma directly so each returned document retains its Chroma ID."""
    started_at = time.perf_counter()
    documents: List[Document] = []
    print(
        f"[backend] timestamp={_timestamp()} request={request_id} "
        f"stage=dense_retrieval "
        f"event=started limit={limit}",
        flush=True,
    )
    try:
        if limit <= 0:
            return documents
        embedding_function = vectorstore._embedding_function
        if embedding_function is None:
            raise RuntimeError("Chroma dense retrieval has no embedding function configured.")

        query_embedding = embedding_function.embed_query(query)
        results = vectorstore._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas"],
        )
        ids = results["ids"][0]
        contents = results["documents"][0]
        metadatas = results["metadatas"][0]
        if not (len(ids) == len(contents) == len(metadatas)):
            raise RuntimeError("Chroma returned misaligned IDs, documents, and metadata.")

        for document_id, content, metadata in zip(ids, contents, metadatas):
            if not isinstance(document_id, str) or not document_id:
                raise RuntimeError("Chroma returned a document without a stable string ID.")
            if not isinstance(content, str):
                raise RuntimeError(
                    f"Chroma returned no content for document {document_id!r}."
                )
            document_metadata = dict(metadata or {})
            document_metadata["id"] = document_id
            documents.append(
                Document(page_content=content, metadata=document_metadata)
            )
        return documents
    finally:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        print(
            f"[backend] timestamp={_timestamp()} request={request_id} "
            f"stage=dense_retrieval "
            f"event=completed candidates={len(documents)} elapsed_ms={elapsed_ms:.2f}",
            flush=True,
        )


def _retrieve_bm25(
    query: str, limit: int, db_path: Path, request_id: str = "unknown"
) -> List[Document]:
    started_at = time.perf_counter()
    documents: List[Document] = []
    print(
        f"[backend] timestamp={_timestamp()} request={request_id} "
        f"stage=bm25_retrieval "
        f"event=started limit={limit}",
        flush=True,
    )
    try:
        results = search_bm25(query, limit=limit, db_path=db_path)
        for result in results:
            metadata = dict(result["metadata"])
            metadata["id"] = result["id"]
            documents.append(
                Document(page_content=result["content"], metadata=metadata)
            )
        return documents
    finally:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        print(
            f"[backend] timestamp={_timestamp()} request={request_id} "
            f"stage=bm25_retrieval "
            f"event=completed candidates={len(documents)} elapsed_ms={elapsed_ms:.2f}",
            flush=True,
        )


async def hybrid_search(
    query: str,
    vectorstore,
    dense_limit: int,
    bm25_limit: int,
    result_limit: int,
    rrf_constant: int,
    bm25_db_path: Path,
    reranker_client: RerankerClient | None = None,
    rerank_limit: int = 5,
    request_id: str = "unknown",
) -> List[Document]:
    """Run dense and sparse retrieval, fuse candidates, and optionally rerank them."""
    if dense_limit <= 0 or bm25_limit <= 0:
        raise ValueError("Dense and BM25 candidate limits must be greater than zero.")
    if result_limit < 0:
        raise ValueError("result_limit must not be negative.")
    if rrf_constant <= 0:
        raise ValueError("rrf_constant must be greater than zero.")
    if rerank_limit <= 0:
        raise ValueError("rerank_limit must be greater than zero.")

    dense_result, bm25_result = await asyncio.gather(
        asyncio.to_thread(
            _retrieve_dense, vectorstore, query, dense_limit, request_id
        ),
        asyncio.to_thread(
            _retrieve_bm25, query, bm25_limit, bm25_db_path, request_id
        ),
        return_exceptions=True,
    )

    for backend_result in (dense_result, bm25_result):
        if isinstance(backend_result, asyncio.CancelledError):
            raise backend_result
        if isinstance(backend_result, BaseException) and not isinstance(
            backend_result, Exception
        ):
            raise backend_result

    dense_error = dense_result if isinstance(dense_result, Exception) else None
    bm25_error = bm25_result if isinstance(bm25_result, Exception) else None
    if dense_error and bm25_error:
        raise RuntimeError(
            "Hybrid retrieval failed in both backends "
            f"(dense: {type(dense_error).__name__}: {dense_error}; "
            f"BM25: {type(bm25_error).__name__}: {bm25_error})."
        ) from dense_error
    if dense_error:
        logger.warning(
            "Dense retrieval failed; returning BM25-only results: %s: %s",
            type(dense_error).__name__,
            dense_error,
        )
        dense_result = []
    if bm25_error:
        logger.warning(
            "BM25 retrieval failed; returning dense-only results: %s: %s",
            type(bm25_error).__name__,
            bm25_error,
        )
        bm25_result = []

    rrf_started = time.perf_counter()
    print(
        f"[backend] timestamp={_timestamp()} request={request_id} "
        f"stage=reciprocal_rank_fusion "
        f"event=started dense={len(dense_result)} bm25={len(bm25_result)} "
        f"candidate_limit={result_limit}",
        flush=True,
    )
    fused_documents = reciprocal_rank_fusion(
        dense_result,
        bm25_result,
        result_limit=result_limit,
        rrf_constant=rrf_constant,
    )
    rrf_elapsed_ms = (time.perf_counter() - rrf_started) * 1000
    print(
        f"[backend] timestamp={_timestamp()} request={request_id} "
        f"stage=reciprocal_rank_fusion "
        f"event=completed candidates={len(fused_documents)} "
        f"elapsed_ms={rrf_elapsed_ms:.2f}",
        flush=True,
    )
    if reranker_client is None or not fused_documents:
        return fused_documents

    documents_by_id = {
        _document_id(document): document for document in fused_documents
    }
    reranked_results = await reranker_client.rerank(
        query=query,
        documents=[
            {"id": document_id, "text": document.page_content}
            for document_id, document in documents_by_id.items()
        ],
        top_n=rerank_limit,
        request_id=request_id,
    )

    mapping_started = time.perf_counter()
    print(
        f"[backend] timestamp={_timestamp()} request={request_id} "
        f"stage=reranked_document_mapping "
        f"event=started results={len(reranked_results)}",
        flush=True,
    )
    reranked_documents = []
    for result in reranked_results:
        document = documents_by_id.get(result.id)
        if document is None:
            raise RuntimeError(
                f"Reranker returned an unknown candidate ID: {result.id!r}."
            )
        reranked_documents.append(document)
    mapping_elapsed_ms = (time.perf_counter() - mapping_started) * 1000
    print(
        f"[backend] timestamp={_timestamp()} request={request_id} "
        f"stage=reranked_document_mapping "
        f"event=completed mapped={len(reranked_documents)} "
        f"elapsed_ms={mapping_elapsed_ms:.2f}",
        flush=True,
    )
    return reranked_documents


# ============================================================
# RAG system singleton
# ============================================================

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

        from langchain_ollama import OllamaEmbeddings
        from langchain_chroma import Chroma

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


# ============================================================
# Task execution
# ============================================================

def _document_to_evidence(
    document: Document,
    task_id: str,
    retrieval_round: int = 0,
) -> EvidenceRecord:
    """
    Convert a retrieved document into structured evidence.

    Preserve stable document identity, metadata, originating task,
    and retrieval round.
    """

    document_id = document.metadata.get("id")

    if (
        not isinstance(document_id, str)
        or not document_id.strip()
    ):
        raise ValueError(
            "Retrieved document is missing a stable string ID."
        )

    return EvidenceRecord(
        doc_id=document_id,
        content=document.page_content,
        metadata=dict(document.metadata),
        source_task_ids=[task_id],
        retrieval_rounds=[retrieval_round],
    )


async def execute_retrieval_task(
    task: TaskSpec,
    request_id: str = "unknown",
) -> tuple[TaskResult, list[EvidenceRecord]]:
    """
    Execute one retrieval task.

    A task with no retrieved evidence is considered failed so that
    the orchestrator can distinguish an unsuccessful retrieval from
    a successful retrieval that produced supporting evidence.
    """

    started_at = time.perf_counter()

    try:
        documents = await rag_system.retrieve_documents(
            query=task.retrieval_query,
            request_id=request_id,
        )

        evidence = [
            _document_to_evidence(
                document=document,
                task_id=task.task_id,
                retrieval_round=task.followup_round,
            )
            for document in documents
        ]

        elapsed_ms = (
            time.perf_counter() - started_at
        ) * 1000

        if not evidence:

            result = TaskResult(
                task_id=task.task_id,
                status="failed",
                evidence_ids=[],
                attempts=task.attempts,
                error="Retrieval returned no documents.",
                latency_ms=elapsed_ms,
            )

            logger.warning(
                "Retrieval task returned no evidence: task_id=%s",
                task.task_id,
            )

            return result, []

        result = TaskResult(
            task_id=task.task_id,
            status="completed",
            evidence_ids=[
                item.doc_id
                for item in evidence
            ],
            attempts=task.attempts,
            latency_ms=elapsed_ms,
        )

        logger.info(
            "Retrieval task completed: task_id=%s "
            "documents=%d elapsed_ms=%.2f",
            task.task_id,
            len(evidence),
            elapsed_ms,
        )

        return result, evidence

    except Exception as exc:

        elapsed_ms = (
            time.perf_counter() - started_at
        ) * 1000

        logger.exception(
            "Retrieval task failed: task_id=%s",
            task.task_id,
        )

        result = TaskResult(
            task_id=task.task_id,
            status="failed",
            evidence_ids=[],
            attempts=task.attempts,
            error=f"{type(exc).__name__}: {exc}",
            latency_ms=elapsed_ms,
        )

        return result, []