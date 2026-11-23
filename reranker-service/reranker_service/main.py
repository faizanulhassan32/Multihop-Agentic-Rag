from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .config import ServiceSettings
from .model import load_cross_encoder
from .schemas import RerankRequest, RerankResponse, RerankResult

logger = logging.getLogger("uvicorn.error")
ModelLoader = Callable[[ServiceSettings], Any]


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _to_float_scores(raw_scores: Any, expected_count: int) -> list[float]:
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    if isinstance(raw_scores, (int, float)):
        raw_scores = [raw_scores]
    if not isinstance(raw_scores, (list, tuple)) or len(raw_scores) != expected_count:
        raise RuntimeError(
            f"CrossEncoder returned an unexpected number of scores; expected {expected_count}."
        )

    scores: list[float] = []
    for raw_score in raw_scores:
        if hasattr(raw_score, "tolist"):
            raw_score = raw_score.tolist()
        if isinstance(raw_score, (list, tuple)):
            if len(raw_score) != 1:
                raise RuntimeError("CrossEncoder must return one relevance score per pair.")
            raw_score = raw_score[0]
        score = float(raw_score)
        if not math.isfinite(score):
            raise RuntimeError("CrossEncoder returned a non-finite relevance score.")
        scores.append(score)
    return scores


def create_app(
    settings: ServiceSettings | None = None,
    model_loader: ModelLoader | None = None,
) -> FastAPI:
    service_settings = settings or ServiceSettings.from_env()
    load_model = model_loader or load_cross_encoder

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.reranker = None
        app.state.model_load_error = None
        app.state.inference_semaphore = asyncio.Semaphore(
            service_settings.max_concurrent_requests
        )

        started_at = time.perf_counter()
        try:
            app.state.reranker = await run_in_threadpool(load_model, service_settings)
        except Exception as exc:
            app.state.model_load_error = str(exc)
            elapsed = time.perf_counter() - started_at
            logger.exception(
                "Failed to load reranker model %s after %.3f seconds.",
                service_settings.model_name,
                elapsed,
            )
        else:
            elapsed = time.perf_counter() - started_at
            logger.info(
                "Loaded reranker model %s on configured device %s in %.3f seconds.",
                service_settings.model_name,
                service_settings.device,
                elapsed,
            )
        yield
        app.state.reranker = None

    app = FastAPI(
        title="Standalone BGE Reranker",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def readiness(request: Request) -> dict[str, str]:
        if request.app.state.reranker is None:
            detail = {"status": "not_ready"}
            if request.app.state.model_load_error:
                detail["error"] = request.app.state.model_load_error
            raise HTTPException(status_code=503, detail=detail)
        return {"status": "ready", "model": service_settings.model_name}

    @app.post("/v1/rerank", response_model=RerankResponse)
    async def rerank(payload: RerankRequest, request: Request) -> RerankResponse:
        request_id = request.headers.get("x-request-id", "")
        if not request_id or len(request_id) > 64 or not all(
            character.isascii()
            and (character.isalnum() or character in "-_.")
            for character in request_id
        ):
            request_id = uuid.uuid4().hex[:12]
        request_started = time.perf_counter()
        print(
            f"[reranker] timestamp={_timestamp()} request={request_id} "
            f"event=request_received "
            f"documents={len(payload.documents)} top_n={payload.top_n}",
            flush=True,
        )

        try:
            model = request.app.state.reranker
            if model is None:
                raise HTTPException(
                    status_code=503, detail="Reranker model is not ready."
                )

            if not payload.documents:
                response = RerankResponse(results=[])
                print(
                    f"[reranker] timestamp={_timestamp()} request={request_id} "
                    f"event=inference_skipped "
                    "reason=no_documents",
                    flush=True,
                )
                print(
                    f"[reranker] timestamp={_timestamp()} request={request_id} "
                    f"event=results_returned ranking=[]",
                    flush=True,
                )
                return response

            pairs = [(payload.query, document.text) for document in payload.documents]

            async with request.app.state.inference_semaphore:
                inference_started = time.perf_counter()
                print(
                    f"[reranker] timestamp={_timestamp()} request={request_id} "
                    f"event=inference_started "
                    f"documents={len(payload.documents)}",
                    flush=True,
                )
                try:
                    raw_scores = await run_in_threadpool(
                        model.predict,
                        pairs,
                        batch_size=service_settings.batch_size,
                        show_progress_bar=False,
                        apply_softmax=False,
                        convert_to_numpy=True,
                    )
                except Exception:
                    inference_elapsed_ms = (
                        time.perf_counter() - inference_started
                    ) * 1000
                    print(
                        f"[reranker] timestamp={_timestamp()} request={request_id} "
                        f"event=inference_failed "
                        f"elapsed_ms={inference_elapsed_ms:.2f}",
                        flush=True,
                    )
                    logger.exception("Reranker inference failed.")
                    raise

                inference_elapsed_ms = (
                    time.perf_counter() - inference_started
                ) * 1000
                print(
                    f"[reranker] timestamp={_timestamp()} request={request_id} "
                    f"event=inference_completed "
                    f"elapsed_ms={inference_elapsed_ms:.2f}",
                    flush=True,
                )

            try:
                scores = _to_float_scores(raw_scores, len(payload.documents))
                ranked = sorted(
                    zip(payload.documents, scores),
                    key=lambda item: -item[1],
                )
                result_limit = (
                    payload.top_n if payload.top_n is not None else len(ranked)
                )
                results = [
                    RerankResult(id=document.id, score=score, rank=rank)
                    for rank, (document, score) in enumerate(
                        ranked[:result_limit], start=1
                    )
                ]
                response = RerankResponse(results=results)
            except Exception:
                logger.exception("Failed to process reranker scores.")
                raise

            ranking = [
                {"id": result.id, "score": result.score, "rank": result.rank}
                for result in results
            ]
            print(
                f"[reranker] timestamp={_timestamp()} request={request_id} "
                f"event=results_returned "
                f"ranking={ranking}",
                flush=True,
            )
            return response

        finally:
            total_elapsed_ms = (time.perf_counter() - request_started) * 1000
            print(
                f"[reranker] timestamp={_timestamp()} request={request_id} "
                f"event=request_completed "
                f"elapsed_ms={total_elapsed_ms:.2f}",
                flush=True,
            )

    return app


app = create_app()
