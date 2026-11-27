
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent.answer_synthesis import AnswerSynthesis
from agent.graph import generate_final_answer


class TestNaturalLanguageStreaming(unittest.IsolatedAsyncioTestCase):

    @patch("agent.graph.AsyncClient")
    async def test_final_answer_uses_native_structured_token_stream(
        self,
        mock_client_type,
    ):
        response_json = (
            '{"answer":"The answer is 42. [source:doc-42]",'
            '"used_evidence_ids":["doc-42"],'
            '"status":"structured",'
            '"warnings":[]}'
        )

        async def token_stream():
            for content in (
                response_json[:30],
                response_json[30:65],
                response_json[65:],
            ):
                yield SimpleNamespace(
                    message=SimpleNamespace(content=content)
                )

        mock_client = mock_client_type.return_value
        mock_client.chat = AsyncMock(
            return_value=token_stream()
        )
        mock_client_type.return_value.__aenter__.return_value = mock_client

        result = await generate_final_answer(
            {
                "original_question": "What is the answer?",
                "evidence": [
                    {
                        "doc_id": "doc-42",
                        "content": "The answer is 42.",
                        "metadata": {},
                        "source_task_ids": ["task-1"],
                        "retrieval_rounds": [0],
                    }
                ],
                "trace": [],
            },
            {
                "configurable": {
                    "model": "test-model",
                    "temperature": 0.2,
                    "max_tokens": 64,
                }
            },
        )

        self.assertEqual(
            result["final_answer"],
            "The answer is 42. [source:doc-42]",
        )

        kwargs = mock_client.chat.await_args.kwargs

        self.assertTrue(kwargs["stream"])
        self.assertEqual(
            kwargs["format"],
            AnswerSynthesis.model_json_schema(),
        )
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(
            kwargs["options"],
            {
                "temperature": 0.2,
                "num_predict": 64,
            },
        )


if __name__ == "__main__":
    unittest.main()
