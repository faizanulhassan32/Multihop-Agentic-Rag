import asyncio
import logging
from pathlib import Path
from typing import List, Sequence

from langchain_core.documents import Document

from ingestion.build_bm25_index import search_bm25


logger = logging.getLogger(__name__)


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


def _retrieve_dense(vectorstore, query: str, limit: int) -> List[Document]:
    """Query Chroma directly so each returned document retains its Chroma ID."""
    if limit <= 0:
        return []
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

    documents = []
    for document_id, content, metadata in zip(ids, contents, metadatas):
        if not isinstance(document_id, str) or not document_id:
            raise RuntimeError("Chroma returned a document without a stable string ID.")
        if not isinstance(content, str):
            raise RuntimeError(f"Chroma returned no content for document {document_id!r}.")
        document_metadata = dict(metadata or {})
        document_metadata["id"] = document_id
        documents.append(Document(page_content=content, metadata=document_metadata))
    return documents


def _retrieve_bm25(query: str, limit: int, db_path: Path) -> List[Document]:
    results = search_bm25(query, limit=limit, db_path=db_path)
    documents = []
    for result in results:
        metadata = dict(result["metadata"])
        metadata["id"] = result["id"]
        documents.append(Document(page_content=result["content"], metadata=metadata))
    return documents


async def hybrid_search(
    query: str,
    vectorstore,
    dense_limit: int,
    bm25_limit: int,
    result_limit: int,
    rrf_constant: int,
    bm25_db_path: Path,
) -> List[Document]:
    """Run dense and sparse retrieval concurrently, then fuse their ranked results."""
    if dense_limit <= 0 or bm25_limit <= 0:
        raise ValueError("Dense and BM25 candidate limits must be greater than zero.")
    if result_limit < 0:
        raise ValueError("result_limit must not be negative.")
    if rrf_constant <= 0:
        raise ValueError("rrf_constant must be greater than zero.")

    dense_result, bm25_result = await asyncio.gather(
        asyncio.to_thread(_retrieve_dense, vectorstore, query, dense_limit),
        asyncio.to_thread(_retrieve_bm25, query, bm25_limit, bm25_db_path),
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

    return reciprocal_rank_fusion(
        dense_result,
        bm25_result,
        result_limit=result_limit,
        rrf_constant=rrf_constant,
    )
