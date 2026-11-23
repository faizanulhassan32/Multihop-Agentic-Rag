import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from agent.graph import (
    QuestionDecomposition,
    decompose_question,
    generate_final_answer,
)


class TestQuestionDecomposition(unittest.IsolatedAsyncioTestCase):
    @patch("agent.graph.AsyncClient")
    async def test_uses_native_schema_and_validates_json(self, mock_client_type):
        mock_client = mock_client_type.return_value
        mock_client.chat = AsyncMock(
            return_value=SimpleNamespace(
                message=SimpleNamespace(
                    content='{"questions":["  Who founded Acme?  ","When was it founded?"]}'
                )
            )
        )
        mock_client_type.return_value.__aenter__.return_value = mock_client

        result = await decompose_question(
            {"original_question": "Who founded Acme and when?"},
            {"configurable": {"model": "test-model"}},
        )

        self.assertEqual(result["decomposed_questions"], ["Who founded Acme?", "When was it founded?"])
        kwargs = mock_client.chat.await_args.kwargs
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["format"], QuestionDecomposition.model_json_schema())

    @patch("agent.graph.AsyncClient")
    async def test_invalid_structured_response_is_not_silently_accepted(self, mock_client_type):
        mock_client = mock_client_type.return_value
        mock_client.chat = AsyncMock(
            return_value=SimpleNamespace(
                message=SimpleNamespace(content='{"questions":[]}'),
            )
        )
        mock_client_type.return_value.__aenter__.return_value = mock_client

        with self.assertRaisesRegex(ValueError, "invalid question decomposition"):
            await decompose_question(
                {"original_question": "Who founded Acme?"},
                {"configurable": {}},
            )

    def test_schema_requires_at_least_one_question(self):
        with self.assertRaises(ValidationError):
            QuestionDecomposition.model_validate({"questions": []})


class TestNaturalLanguageStreaming(unittest.IsolatedAsyncioTestCase):
    @patch("agent.graph.AsyncClient")
    async def test_final_answer_uses_unstructured_token_stream(
        self,
        mock_client_type,
    ):
        async def token_stream():
            for content in ("The answer", " is 42."):
                yield SimpleNamespace(message=SimpleNamespace(content=content))

        mock_client = mock_client_type.return_value
        mock_client.chat = AsyncMock(return_value=token_stream())
        mock_client_type.return_value.__aenter__.return_value = mock_client

        result = await generate_final_answer(
            {
                "original_question": "What is the answer?",
                "intermediate_answers": ["The answer is 42."],
                "trace": [],
            },
            {"configurable": {"model": "test-model", "temperature": 0.2, "max_tokens": 64}},
        )

        self.assertEqual(result["final_answer"], "The answer is 42.")
        kwargs = mock_client.chat.await_args.kwargs
        self.assertTrue(kwargs["stream"])
        self.assertNotIn("format", kwargs)
        self.assertEqual(kwargs["options"], {"temperature": 0.2, "num_predict": 64})


if __name__ == "__main__":
    unittest.main()
