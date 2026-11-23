import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

from langchain_core.documents import Document

from ingestion.build_bm25_index import search_bm25
from services.reranker_client import RerankerClient


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
