import asyncio
import json
import unittest
from types import SimpleNamespace

import httpx
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from api import main as api


class EventGraph:
    def __init__(self):
        self.stream_calls = 0
        self.invoke_calls = 0
        self.active_streams = 0
        self.max_active_streams = 0

    async def astream_events(self, input, config, version):
        self.stream_calls += 1
        question = input["original_question"]
        self.active_streams += 1
        self.max_active_streams = max(self.max_active_streams, self.active_streams)
        try:
            yield {
                "event": "on_chain_start",
                "name": "decompose_question",
                "metadata": {"langgraph_node": "decompose_question"},
                "data": {},
            }
            yield {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "decompose_question"},
                "data": {"chunk": SimpleNamespace(content="internal")},
            }
            yield {
                "event": "on_chain_start",
                "name": "generate_final_answer",
                "metadata": {"langgraph_node": "generate_final_answer"},
                "data": {},
            }
            for token in ("final ", question):
                await asyncio.sleep(0.01)
                yield {
                    "event": "on_chat_model_stream",
                    "metadata": {"langgraph_node": "generate_final_answer"},
                    "data": {"chunk": SimpleNamespace(content=token)},
                }
            yield {
                "event": "on_chain_end",
                "name": "supervisor_finalize",
                "metadata": {"langgraph_node": "supervisor_finalize"},
                "data": {
                    "output": {
                        "final_answer": f"final {question}",
                        "trace": [],
                        "classification": "in_scope",
                        "classification_reason": "test",
                        "decomposed_questions": [question],
                    }
                },
            }
        finally:
            self.active_streams -= 1

    async def ainvoke(self, *_args, **_kwargs):
        self.invoke_calls += 1
        raise AssertionError("API must not rerun the graph through ainvoke.")


class WrongTargetEventGraph(EventGraph):
    async def astream_events(self, input, config, version):
        self.stream_calls += 1
        yield {
            "event": "on_chain_end",
            "name": "assess_evidence",
            "metadata": {"langgraph_node": "assess_evidence"},
            "data": {"output": {
                "answer_target": {"verified_answer_value": "American"},
            }},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "generate_final_answer"},
            "data": {"chunk": SimpleNamespace(content="English")},
        }
        yield {
            "event": "on_chain_end",
            "name": "supervisor_finalize",
            "metadata": {"langgraph_node": "supervisor_finalize"},
            "data": {"output": {
                "final_answer": "I couldn't verify a response for the answer target in the available evidence.",
                "trace": [],
                "classification": "in_scope",
                "classification_reason": "test",
                "decomposed_questions": [],
            }},
        }


class StreamingAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_graph = api.graph
        self.fake_graph = EventGraph()
        api.graph = self.fake_graph
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        api.graph = self.original_graph

    async def create_thread(self):
        response = await self.client.post("/threads", json={})
        self.assertEqual(response.status_code, 200)
        return response.json()["thread_id"]

    @staticmethod
    def parse_events(response):
        return [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]

    async def test_sse_filters_non_answer_tokens_and_executes_graph_once(self):
        thread_id = await self.create_thread()

        response = await self.client.post(
            f"/threads/{thread_id}/runs/stream",
            json={"question": "answer"},
        )

        self.assertEqual(response.status_code, 200)
        events = self.parse_events(response)
        tokens = [event["data"]["token"] for event in events if event["event"] == "token"]
        self.assertEqual(tokens, ["final ", "answer"])
        self.assertNotIn("internal", tokens)
        self.assertEqual(self.fake_graph.stream_calls, 1)
        self.assertEqual(self.fake_graph.invoke_calls, 0)
        values = next(event["data"] for event in events if event["event"] == "values")
        self.assertEqual(values["final_answer"], "final answer")

    async def test_sse_does_not_expose_answer_tokens_that_conflict_with_target(self):
        self.fake_graph = WrongTargetEventGraph()
        api.graph = self.fake_graph
        thread_id = await self.create_thread()

        response = await self.client.post(
            f"/threads/{thread_id}/runs/stream",
            json={"question": "wife nationality"},
        )

        events = self.parse_events(response)
        tokens = [event["data"]["token"] for event in events if event["event"] == "token"]
        self.assertEqual(
            "".join(tokens),
            "I couldn't verify a response for the answer target in the available evidence.",
        )
        self.assertNotIn("English", tokens)
        self.assertEqual(self.fake_graph.stream_calls, 1)
        values = next(event["data"] for event in events if event["event"] == "values")
        self.assertEqual(
            values["final_answer"],
            "I couldn't verify a response for the answer target in the available evidence.",
        )

    async def test_concurrent_threads_keep_request_state_isolated(self):
        thread_one, thread_two = await asyncio.gather(
            self.create_thread(),
            self.create_thread(),
        )

        response_one, response_two = await asyncio.gather(
            self.client.post(
                f"/threads/{thread_one}/runs/stream",
                json={"question": "first"},
            ),
            self.client.post(
                f"/threads/{thread_two}/runs/stream",
                json={"question": "second"},
            ),
        )

        self.assertEqual(self.fake_graph.stream_calls, 2)
        self.assertEqual(self.fake_graph.invoke_calls, 0)
        self.assertEqual(self.fake_graph.max_active_streams, 2)
        values_one = next(
            event["data"] for event in self.parse_events(response_one) if event["event"] == "values"
        )
        values_two = next(
            event["data"] for event in self.parse_events(response_two) if event["event"] == "values"
        )
        self.assertEqual(values_one["final_answer"], "final first")
        self.assertEqual(values_two["final_answer"], "final second")

        state_one, state_two = await asyncio.gather(
            self.client.get(f"/threads/{thread_one}/state"),
            self.client.get(f"/threads/{thread_two}/state"),
        )
        self.assertEqual(state_one.json()["values"]["final_answer"], "final first")
        self.assertEqual(state_two.json()["values"]["final_answer"], "final second")

    async def test_concurrent_runs_on_one_thread_are_serialized(self):
        thread_id = await self.create_thread()

        response_one, response_two = await asyncio.gather(
            self.client.post(
                f"/threads/{thread_id}/runs/stream",
                json={"question": "first"},
            ),
            self.client.post(
                f"/threads/{thread_id}/runs/stream",
                json={"question": "second"},
            ),
        )

        self.assertEqual(self.fake_graph.stream_calls, 2)
        self.assertEqual(self.fake_graph.max_active_streams, 1)
        results = {
            next(
                event["data"]["final_answer"]
                for event in self.parse_events(response)
                if event["event"] == "values"
            )
            for response in (response_one, response_two)
        }
        self.assertEqual(results, {"final first", "final second"})

    async def test_missing_graph_state_emits_error_without_second_execution(self):
        thread_id = await self.create_thread()

        async def incomplete_stream(*_args, **_kwargs):
            self.fake_graph.stream_calls += 1
            yield {
                "event": "on_chain_start",
                "name": "generate_final_answer",
                "metadata": {"langgraph_node": "generate_final_answer"},
                "data": {},
            }

        self.fake_graph.astream_events = incomplete_stream
        response = await self.client.post(
            f"/threads/{thread_id}/runs/stream",
            json={"question": "missing state"},
        )

        events = self.parse_events(response)
        self.assertEqual([event["event"] for event in events][-1], "error")
        self.assertEqual(self.fake_graph.stream_calls, 1)
        self.assertEqual(self.fake_graph.invoke_calls, 0)

    async def test_thread_state_excludes_non_public_langchain_values(self):
        thread_id = await self.create_thread()
        api._thread_store[thread_id] = {
            "final_answer": "answer",
            "messages": [AIMessage(content="answer")],
            "retrieved_docs": [Document(page_content="evidence", metadata={"title": "source"})],
        }

        response = await self.client.get(f"/threads/{thread_id}/state")

        self.assertEqual(response.status_code, 200)
        values = response.json()["values"]
        self.assertEqual(values["final_answer"], "answer")
        self.assertNotIn("messages", values)
        self.assertNotIn("retrieved_docs", values)


if __name__ == "__main__":
    unittest.main()
