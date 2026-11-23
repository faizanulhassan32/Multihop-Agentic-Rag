from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import time
import uuid

import httpx

from ingestion.config import Config


MAX_DOCUMENTS_PER_REQUEST = 20


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


class RerankerClientError(RuntimeError):
    """Base exception for reranker client failures."""


class RerankerConnectionError(RerankerClientError):
    """The reranker service could not be reached."""


class RerankerTimeoutError(RerankerClientError):
    """The reranker service did not respond before the configured timeout."""


class RerankerServiceError(RerankerClientError):
    """The reranker service returned an unsuccessful HTTP status."""


class RerankerResponseError(RerankerClientError):
    """The reranker service returned an invalid response."""


@dataclass(frozen=True, slots=True)
class RerankedResult:
    id: str
    score: float
    rank: int


class RerankerClient:
    """Async client for the standalone reranker service."""

    def __init__(
        self,
        service_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        configured_url = (
            Config.RERANKER_SERVICE_URL if service_url is None else service_url
        )
        if not isinstance(configured_url, str) or not configured_url.strip():
            raise ValueError("Reranker service URL must not be empty.")
        self._service_url = configured_url.strip().rstrip("/")

        configured_timeout = (
            Config.RERANKER_REQUEST_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        if (
            isinstance(configured_timeout, bool)
            or not isinstance(configured_timeout, (int, float))
            or not math.isfinite(configured_timeout)
            or configured_timeout <= 0
        ):
            raise ValueError("Reranker request timeout must be a positive number.")
        self._timeout_seconds = float(configured_timeout)
        self._http = httpx.AsyncClient(timeout=self._timeout_seconds)

    async def __aenter__(self) -> RerankerClient:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if not self._http.is_closed:
            await self._http.aclose()

    async def rerank(
        self,
        query: str,
        documents: Sequence[Mapping[str, str]],
        top_n: int | None = None,
        request_id: str | None = None,
    ) -> list[RerankedResult]:
        payload = self._build_payload(query, documents, top_n)
        url = f"{self._service_url}/v1/rerank"
        request_id = request_id or uuid.uuid4().hex[:12]
        request_started = time.perf_counter()
        print(
            f"[backend] timestamp={_timestamp()} request={request_id} "
            f"stage=reranker_http "
            f"event=request_started candidates={len(payload['documents'])} top_n={top_n}",
            flush=True,
        )

        try:
            response = await self._http.post(
                url,
                json=payload,
                headers={"X-Request-ID": request_id},
            )
        except httpx.TimeoutException as exc:
            elapsed_ms = (time.perf_counter() - request_started) * 1000
            print(
                f"[backend] timestamp={_timestamp()} request={request_id} "
                f"stage=reranker_http "
                f"event=timeout elapsed_ms={elapsed_ms:.2f}",
                flush=True,
            )
            raise RerankerTimeoutError(
                f"Reranker request to {url} timed out after "
                f"{self._timeout_seconds:g} seconds."
            ) from exc
        except httpx.RequestError as exc:
            elapsed_ms = (time.perf_counter() - request_started) * 1000
            print(
                f"[backend] timestamp={_timestamp()} request={request_id} "
                f"stage=reranker_http "
                f"event=connection_error elapsed_ms={elapsed_ms:.2f}",
                flush=True,
            )
            raise RerankerConnectionError(
                f"Could not reach reranker service at {url}: {exc}."
            ) from exc

        elapsed_ms = (time.perf_counter() - request_started) * 1000
        print(
            f"[backend] timestamp={_timestamp()} request={request_id} "
            f"stage=reranker_http "
            f"event=response_received status={response.status_code} "
            f"elapsed_ms={elapsed_ms:.2f}",
            flush=True,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text.strip()
            if len(detail) > 500:
                detail = f"{detail[:500]}..."
            message = f"Reranker service returned HTTP {response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            raise RerankerServiceError(message) from exc

        try:
            response_body = response.json()
        except ValueError as exc:
            raise RerankerResponseError(
                "Reranker service response was not valid JSON."
            ) from exc

        results = self._parse_results(response_body, payload["documents"], top_n)
        ranking = [
            {"id": result.id, "score": result.score, "rank": result.rank}
            for result in results
        ]
        print(
            f"[backend] timestamp={_timestamp()} request={request_id} "
            f"stage=reranker_http "
            f"event=ranking results={ranking}",
            flush=True,
        )
        return results

    @staticmethod
    def _build_payload(
        query: str,
        documents: Sequence[Mapping[str, str]],
        top_n: int | None,
    ) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Reranker query must not be blank.")
        if isinstance(documents, (str, bytes)) or not isinstance(
            documents, Sequence
        ):
            raise ValueError("Reranker documents must be a sequence of ID/text mappings.")
        if len(documents) > MAX_DOCUMENTS_PER_REQUEST:
            raise ValueError(
                f"Reranker accepts at most {MAX_DOCUMENTS_PER_REQUEST} documents."
            )
        if top_n is not None and (
            isinstance(top_n, bool)
            or not isinstance(top_n, int)
            or not 1 <= top_n <= MAX_DOCUMENTS_PER_REQUEST
        ):
            raise ValueError(
                f"top_n must be between 1 and {MAX_DOCUMENTS_PER_REQUEST}."
            )

        request_documents = []
        seen_ids = set()
        for document in documents:
            if not isinstance(document, Mapping):
                raise ValueError("Each reranker document must contain an id and text.")
            document_id = document.get("id")
            text = document.get("text")
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError("Each reranker document must have a non-blank string id.")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Reranker document {document_id!r} must have non-blank text."
                )
            if document_id in seen_ids:
                raise ValueError(f"Duplicate reranker document ID: {document_id!r}.")
            seen_ids.add(document_id)
            request_documents.append({"id": document_id, "text": text})

        return {
            "query": query,
            "documents": request_documents,
            "top_n": top_n,
        }

    @staticmethod
    def _parse_results(
        response_body: object,
        request_documents: list[dict[str, str]],
        top_n: int | None,
    ) -> list[RerankedResult]:
        if not isinstance(response_body, dict) or not isinstance(
            response_body.get("results"), list
        ):
            raise RerankerResponseError(
                "Reranker response must contain a results array."
            )

        results = response_body["results"]
        expected_count = min(
            len(request_documents) if top_n is None else top_n,
            len(request_documents),
        )
        if len(results) != expected_count:
            raise RerankerResponseError(
                f"Reranker returned {len(results)} results; expected {expected_count}."
            )

        submitted_ids = {document["id"] for document in request_documents}
        parsed_results = []
        seen_result_ids = set()
        previous_score = math.inf
        for expected_rank, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                raise RerankerResponseError(
                    f"Reranker result at position {expected_rank} must be an object."
                )

            document_id = result.get("id")
            score = result.get("score")
            rank = result.get("rank")
            if not isinstance(document_id, str) or document_id not in submitted_ids:
                raise RerankerResponseError(
                    f"Reranker returned an unknown document ID at rank {expected_rank}."
                )
            if document_id in seen_result_ids:
                raise RerankerResponseError(
                    f"Reranker returned duplicate document ID {document_id!r}."
                )
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise RerankerResponseError(
                    f"Reranker returned an invalid score at rank {expected_rank}."
                )
            if isinstance(rank, bool) or not isinstance(rank, int) or rank != expected_rank:
                raise RerankerResponseError(
                    f"Reranker returned an invalid rank at position {expected_rank}."
                )
            if score > previous_score:
                raise RerankerResponseError(
                    "Reranker results were not sorted by descending score."
                )

            parsed_results.append(
                RerankedResult(id=document_id, score=float(score), rank=rank)
            )
            seen_result_ids.add(document_id)
            previous_score = score

        return parsed_results
