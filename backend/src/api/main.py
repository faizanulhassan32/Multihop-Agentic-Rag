import os
import sys
import uuid
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

# Add backend/libs to import path for packaged deps
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "libs")))

# Ensure the backend src directory is on the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Disable LangSmith tracing (no internet)
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "")

# Import the LangGraph workflow
from agent.graph import graph, rag_system


@asynccontextmanager
async def lifespan(_app: FastAPI):
    rag_system.startup()
    try:
        yield
    finally:
        await rag_system.shutdown()


app = FastAPI(lifespan=lifespan)

# Allow the frontend (likely running on http://localhost:5173) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] ,  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In‑memory store for thread results (simple for demo)
_thread_store: Dict[str, Dict[str, Any]] = {}


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _question_preview(question: Any) -> str:
    question_text = question if isinstance(question, str) else repr(question)
    question_text = " ".join(question_text.split())
    if len(question_text) > 200:
        question_text = f"{question_text[:197]}..."
    return question_text


@app.get("/assistants")
async def list_assistants() -> list:
    """Placeholder – the UI only checks that this endpoint exists.
    Returning an empty list is sufficient for now.
    """
    return []

@app.post("/threads")
async def create_thread() -> Dict[str, str]:
    """Create a new thread identifier. The frontend expects a JSON object
    containing ``thread_id``.
    """
    thread_id = str(uuid.uuid4())
    # Initialise empty store entry
    _thread_store[thread_id] = {}
    return {"thread_id": thread_id}

@app.post("/threads/{thread_id}/runs/stream")
async def run_stream(
    thread_id: str,
    payload: Dict[str, Any]
) -> StreamingResponse:
    """Execute the graph for the supplied thread and stream SSE events."""

    # The frontend now sends a plain JSON payload with a single field "question"
    question = payload.get("question")
    request_id = uuid.uuid4().hex[:12]
    request_started = time.perf_counter()
    print(
        f"[backend] timestamp={_timestamp()} request={request_id} thread={thread_id} "
        f"event=request_received question={_question_preview(question)!r}",
        flush=True,
    )
    if not question:
        elapsed_ms = (time.perf_counter() - request_started) * 1000
        print(
            f"[backend] timestamp={_timestamp()} request={request_id} thread={thread_id} "
            f"event=response_completed status=400 elapsed_ms={elapsed_ms:.2f}",
            flush=True,
        )
        raise HTTPException(status_code=400, detail="Missing 'question' in request payload")
    config = {"metadata": {"request_id": request_id}}

    async def event_generator():
        status = "completed"
        try:
            # Run the LangGraph workflow
            result = await graph.ainvoke({"original_question": question}, config=config)
            # Store the result for later state queries
            _thread_store[thread_id] = result

            # Build a single SSE event that matches what the frontend parses
            event = {
                "event": "values",
                "data": {
                    "final_answer": result.get("final_answer"),
                    "trace": result.get("trace", []),
                    # Include classifier fields for completeness
                    "classification": result.get("classification"),
                    "classification_reason": result.get("classification_reason"),
                },
            }
            # SSE format: "data: <json>\n\n"
            yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            status = "error"
            # In case of unexpected failure, send an error event
            error_event = {"event": "error", "data": {"message": str(exc)}}
            yield f"data: {json.dumps(error_event)}\n\n"
        finally:
            elapsed_ms = (time.perf_counter() - request_started) * 1000
            print(
                f"[backend] timestamp={_timestamp()} request={request_id} thread={thread_id} "
                f"event=response_completed status={status} elapsed_ms={elapsed_ms:.2f}",
                flush=True,
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/threads/{thread_id}/state")
async def get_thread_state(thread_id: str) -> JSONResponse:
    """Return the final state of a thread. Used as a fallback when streaming
    does not provide a final answer.
    """
    state = _thread_store.get(thread_id)
    if not state:
        return JSONResponse(content={"values": {}}, status_code=200)
    return JSONResponse(content={"values": state})
