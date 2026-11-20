import asyncio
import os
import sys
import uuid
import json
from typing import Dict, Any

# Add backend/libs to import path for packaged deps
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "libs")))

# Ensure the backend src directory is on the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Disable LangSmith tracing (no internet)
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "")

# Import the LangGraph workflow
from agent.graph import graph

app = FastAPI()

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
_thread_locks: Dict[str, asyncio.Lock] = {}


def _get_thread_lock(thread_id: str) -> asyncio.Lock:
    lock = _thread_locks.get(thread_id)
    if lock is None:
        lock = asyncio.Lock()
        _thread_locks[thread_id] = lock
    return lock



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
    _thread_locks[thread_id] = asyncio.Lock()
    return {"thread_id": thread_id}

@app.post("/threads/{thread_id}/runs/stream")
async def run_stream(
    thread_id: str,
    payload: Dict[str, Any]
) -> StreamingResponse:
    """Execute the graph for the supplied thread and stream SSE events."""

    # The frontend now sends a plain JSON payload with a single field "question"
    question = payload.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' in request payload")
    if thread_id not in _thread_store:
        raise HTTPException(status_code=404, detail="Thread not found")
    thread_lock = _get_thread_lock(thread_id)
    config: Dict[str, Any] = {}

    async def event_generator():
        try:
            final_accumulated = ""
            streamed_answer_tokens = ""
            verified_answer_value = ""
            latest_state = {}
            async with thread_lock:
                _thread_store[thread_id] = {}
                # Stream graph events using a single graph execution; never rerun on missing output.
                async for event in graph.astream_events(
                    {"original_question": question},
                    config=config,
                    version="v2",
                ):
                    kind = event["event"]
                    name = event.get("name", "")
                    metadata = event.get("metadata", {})
                    node = metadata.get("langgraph_node", "")

                    if kind == "on_chain_start" and name in [
                        "supervisor_classify", "decompose_question", "parallel_retrieve",
                        "assess_evidence", "followup_retrieve", "generate_final_answer",
                        "supervisor_finalize"
                    ]:
                        progress_messages = {
                            "supervisor_classify": "🔍 Classifying query intent and scope...",
                            "decompose_question": "🧠 Decomposing into parallel sub-questions...",
                            "parallel_retrieve": "⚡ Retrieving evidence chunks concurrently...",
                            "assess_evidence": "🧭 Assessing retrieved evidence...",
                            "followup_retrieve": "🔄 Performing follow-up evidence retrieval...",
                            "generate_final_answer": "✍️ Synthesizing final answer...",
                            "supervisor_finalize": "🛡️ Finalizing supervisor response...",
                        }
                        progress_event = {
                            "event": "progress",
                            "data": {"message": progress_messages[name], "node": name}
                        }
                        yield f"data: {json.dumps(progress_event)}\n\n"

                    elif kind == "on_chat_model_stream" and node == "generate_final_answer":
                        chunk = event.get("data", {}).get("chunk")
                        token = getattr(chunk, "content", "")
                        if token:
                            final_accumulated += token
                            target = verified_answer_value.strip()
                            if target:
                                normalized_candidate = "".join(
                                    char.casefold() for char in final_accumulated
                                    if char.isalnum()
                                )
                                normalized_target = "".join(
                                    char.casefold() for char in target if char.isalnum()
                                )
                                if normalized_target.startswith(normalized_candidate):
                                    streamed_answer_tokens += token
                                    yield (
                                        f"data: {json.dumps({'event': 'token', 'data': {'token': token}})}\n\n"
                                    )
                            else:
                                streamed_answer_tokens += token
                                yield (
                                    f"data: {json.dumps({'event': 'token', 'data': {'token': token}})}\n\n"
                                )

                    elif kind == "on_chain_end" and node:
                        output_data = event.get("data", {}).get("output")
                        if isinstance(output_data, dict):
                            latest_state.update(output_data)
                            target = output_data.get("answer_target")
                            if isinstance(target, dict):
                                verified_answer_value = str(
                                    target.get("verified_answer_value", "")
                                )

                    elif kind == "on_chain_end" and name in ["low-latency-multi-hop-agent", "LangGraph"]:
                        output_data = event.get("data", {}).get("output")
                        if isinstance(output_data, dict):
                            latest_state.update(output_data)

                if "final_answer" not in latest_state:
                    raise RuntimeError(
                        "Graph event stream completed without a final state; graph was not rerun."
                    )

                final_answer = str(latest_state.get("final_answer") or final_accumulated)
                target = verified_answer_value.strip()
                if target:
                    emitted_normalized = "".join(
                        char.casefold() for char in streamed_answer_tokens if char.isalnum()
                    )
                    final_normalized = "".join(
                        char.casefold() for char in final_answer if char.isalnum()
                    )
                    if not final_normalized.startswith(emitted_normalized):
                        streamed_answer_tokens = ""
                    if final_answer and not streamed_answer_tokens:
                        yield (
                            f"data: {json.dumps({'event': 'token', 'data': {'token': final_answer}})}\n\n"
                        )
                    elif emitted_normalized != final_normalized:
                        suffix_start = 0
                        for index in range(len(final_answer) + 1):
                            candidate = "".join(
                                char.casefold()
                                for char in final_answer[:index]
                                if char.isalnum()
                            )
                            if candidate == emitted_normalized:
                                suffix_start = index
                                break
                        suffix = final_answer[suffix_start:]
                        if suffix:
                            yield (
                                f"data: {json.dumps({'event': 'token', 'data': {'token': suffix}})}\n\n"
                            )

                _thread_store[thread_id] = latest_state

                final_event = {
                    "event": "values",
                    "data": {
                        "final_answer": final_answer,
                        "trace": latest_state.get("trace", []),
                        "classification": latest_state.get("classification"),
                        "classification_reason": latest_state.get("classification_reason"),
                        "decomposed_questions": latest_state.get("decomposed_questions", []),
                    },
                }
                yield f"data: {json.dumps(final_event)}\n\n"

        except Exception as exc:
            error_event = {"event": "error", "data": {"message": str(exc)}}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/threads/{thread_id}/state")
async def get_thread_state(thread_id: str) -> JSONResponse:
    """Return the final state of a thread. Used as a fallback when streaming
    does not provide a final answer.
    """
    if thread_id not in _thread_store:
        raise HTTPException(status_code=404, detail="Thread not found")
    async with _get_thread_lock(thread_id):
        state = _thread_store[thread_id]
        public_values = {
            "final_answer": state.get("final_answer"),
            "trace": state.get("trace", []),
            "classification": state.get("classification"),
            "classification_reason": state.get("classification_reason"),
            "decomposed_questions": state.get("decomposed_questions", []),
        }
        return JSONResponse(content={"values": jsonable_encoder(public_values)})
