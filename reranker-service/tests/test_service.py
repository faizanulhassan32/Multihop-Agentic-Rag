from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import unittest
from types import ModuleType, SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from reranker_service.config import ServiceSettings
from reranker_service.main import create_app
from reranker_service.model import load_cross_encoder, resolve_device


class TestConfiguration(unittest.TestCase):
    def test_environment_variables_configure_model_and_inference(self):
        values = {
            "RERANKER_MODEL_NAME": "local/model",
            "RERANKER_MODEL_REVISION": "v1",
            "RERANKER_CACHE_DIR": "~/model-cache",
            "RERANKER_DEVICE": "cuda:0",
            "RERANKER_BATCH_SIZE": "8",
            "RERANKER_MAX_INPUT_LENGTH": "384",
            "RERANKER_MAX_CONCURRENT_REQUESTS": "2",
        }
        with patch.dict(os.environ, values, clear=False):
            configured = ServiceSettings.from_env()

        self.assertEqual(configured.model_name, "local/model")
        self.assertEqual(configured.model_revision, "v1")
        self.assertTrue(configured.cache_dir.endswith("model-cache"))
        self.assertEqual(configured.device, "cuda:0")
        self.assertEqual(configured.batch_size, 8)
        self.assertEqual(configured.max_input_length, 384)
        self.assertEqual(configured.max_concurrent_requests, 2)


class FakeReranker:
    def __init__(self, scores: list[float]):
        self.scores = scores
        self.calls = []

    def predict(
        self,
        pairs,
        *,
        batch_size,
        show_progress_bar,
        apply_softmax,
        convert_to_numpy,
    ):
        self.calls.append(
            {
                "pairs": pairs,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "apply_softmax": apply_softmax,
                "convert_to_numpy": convert_to_numpy,
            }
        )
        return self.scores


def settings(**overrides) -> ServiceSettings:
    values = {
        "device": "cpu",
        "batch_size": 4,
        "max_input_length": 128,
        "max_concurrent_requests": 1,
    }
    values.update(overrides)
    return ServiceSettings(**values)


class TestRerankerService(unittest.TestCase):
    def test_rerank_sorts_raw_scores_and_preserves_ids(self):
        model = FakeReranker([-3.5, 2.25, 0.0])
        load_model = Mock(return_value=model)
        app = create_app(settings(), model_loader=load_model)
        documents = [
            {"id": "id/first", "text": "first passage"},
            {"id": "  exact-id Ω  ", "text": "second passage"},
            {"id": "id:third", "text": "third passage"},
        ]

        with TestClient(app) as client:
            response = client.post(
                "/v1/rerank",
                json={"query": "question", "documents": documents, "top_n": 2},
            )

        self.assertEqual(response.status_code, 200)
        results = response.json()["results"]
        self.assertEqual(
            results,
            [
                {"id": "  exact-id Ω  ", "score": 2.25, "rank": 1},
                {"id": "id:third", "score": 0.0, "rank": 2},
            ],
        )
        self.assertEqual(
            model.calls[0]["pairs"],
            [
                ("question", "first passage"),
                ("question", "second passage"),
                ("question", "third passage"),
            ],
        )
        self.assertEqual(model.calls[0]["batch_size"], 4)
        self.assertFalse(model.calls[0]["apply_softmax"])
        self.assertTrue(model.calls[0]["convert_to_numpy"])
        self.assertFalse(model.calls[0]["show_progress_bar"])
        load_model.assert_called_once()

    def test_default_top_n_returns_all_documents_and_empty_input_skips_inference(self):
        model = FakeReranker([1.0])
        app = create_app(settings(), model_loader=lambda _: model)

        with TestClient(app) as client:
            all_results = client.post(
                "/v1/rerank",
                json={
                    "query": "question",
                    "documents": [{"id": "exact", "text": "passage"}],
                },
            )
            empty_results = client.post(
                "/v1/rerank",
                json={"query": "question", "documents": []},
            )

        self.assertEqual(all_results.status_code, 200)
        self.assertEqual(len(all_results.json()["results"]), 1)
        self.assertEqual(empty_results.status_code, 200)
        self.assertEqual(empty_results.json(), {"results": []})
        self.assertEqual(len(model.calls), 1)

    def test_request_validation_enforces_document_and_top_n_limits(self):
        model = FakeReranker([])
        app = create_app(settings(), model_loader=lambda _: model)
        too_many_documents = [
            {"id": f"id-{index}", "text": "passage"} for index in range(21)
        ]

        with TestClient(app) as client:
            too_many = client.post(
                "/v1/rerank",
                json={"query": "question", "documents": too_many_documents},
            )
            invalid_top_n = client.post(
                "/v1/rerank",
                json={"query": "question", "documents": [], "top_n": 0},
            )
            blank_query = client.post(
                "/v1/rerank",
                json={"query": "   ", "documents": []},
            )

        self.assertEqual(too_many.status_code, 422)
        self.assertEqual(invalid_top_n.status_code, 422)
        self.assertEqual(blank_query.status_code, 422)
        self.assertEqual(model.calls, [])

    def test_health_and_readiness_report_model_state(self):
        app = create_app(settings(), model_loader=lambda _: FakeReranker([]))
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            ready = client.get("/ready")

        self.assertEqual(ready.status_code, 200)
        self.assertEqual(
            ready.json(),
            {"status": "ready", "model": "BAAI/bge-reranker-base"},
        )

    def test_model_loading_and_inference_timings_are_logged(self):
        app = create_app(settings(), model_loader=lambda _: FakeReranker([0.5]))

        with self.assertLogs("uvicorn.error", level=logging.INFO) as captured:
            with TestClient(app) as client:
                response = client.post(
                    "/v1/rerank",
                    json={
                        "query": "question",
                        "documents": [{"id": "id", "text": "passage"}],
                    },
                )

        self.assertEqual(response.status_code, 200)
        messages = "\n".join(captured.output)
        self.assertIn("Loaded reranker model", messages)
        self.assertIn("Reranker inference completed", messages)

    def test_readiness_is_unavailable_when_model_loading_fails(self):
        def fail_loading(_):
            raise RuntimeError("model unavailable")

        app = create_app(settings(), model_loader=fail_loading)
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            response = client.get("/ready")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            {"status": "not_ready", "error": "model unavailable"},
        )

    def test_inference_is_off_loop_and_concurrency_is_bounded(self):
        class TrackingModel:
            def __init__(self):
                self.lock = threading.Lock()
                self.active = 0
                self.max_active = 0
                self.called_outside_event_loop = True

            def predict(
                self,
                pairs,
                *,
                batch_size,
                show_progress_bar,
                apply_softmax,
                convert_to_numpy,
            ):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    pass
                else:
                    self.called_outside_event_loop = False

                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.02)
                with self.lock:
                    self.active -= 1
                return [1.0 for _ in pairs]

        model = TrackingModel()
        app = create_app(
            settings(max_concurrent_requests=1),
            model_loader=lambda _: model,
        )
        payload = {
            "query": "question",
            "documents": [{"id": "id", "text": "passage"}],
        }

        with TestClient(app) as client:
            with ThreadPoolExecutor(max_workers=4) as pool:
                responses = list(
                    pool.map(
                        lambda _: client.post("/v1/rerank", json=payload),
                        range(4),
                    )
                )

        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(model.max_active, 1)
        self.assertTrue(model.called_outside_event_loop)


class TestModelLoading(unittest.TestCase):
    @staticmethod
    def _torch(cuda: bool, mps: bool) -> SimpleNamespace:
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: mps),
            ),
        )

    def test_device_auto_prefers_cuda_then_mps_then_cpu(self):
        self.assertEqual(resolve_device("auto", self._torch(cuda=True, mps=True)), "cuda")
        self.assertEqual(resolve_device("auto", self._torch(cuda=False, mps=True)), "mps")
        self.assertEqual(resolve_device("auto", self._torch(cuda=False, mps=False)), "cpu")

    def test_unavailable_explicit_device_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            resolve_device("cuda", self._torch(cuda=False, mps=False))
        with self.assertRaisesRegex(RuntimeError, "MPS is unavailable"):
            resolve_device("mps", self._torch(cuda=False, mps=False))

    def test_loader_uses_supported_cross_encoder_arguments_and_raw_scores(self):
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False)
        )

        class FakeIdentity:
            pass

        fake_torch.nn = SimpleNamespace(Identity=FakeIdentity)
        cross_encoder = Mock(return_value=object())
        fake_sentence_transformers = ModuleType("sentence_transformers")
        fake_sentence_transformers.CrossEncoder = cross_encoder
        model_settings = settings(
            model_name="BAAI/bge-reranker-base",
            model_revision="model-revision",
            cache_dir="/tmp/reranker-cache",
            device="cpu",
            max_input_length=256,
        )

        with patch.dict(
            "sys.modules",
            {
                "torch": fake_torch,
                "sentence_transformers": fake_sentence_transformers,
            },
        ):
            load_cross_encoder(model_settings)

        kwargs = cross_encoder.call_args.kwargs
        self.assertEqual(cross_encoder.call_args.args, ("BAAI/bge-reranker-base",))
        self.assertEqual(kwargs["revision"], "model-revision")
        self.assertEqual(kwargs["cache_dir"], "/tmp/reranker-cache")
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["max_length"], 256)
        self.assertIsInstance(kwargs["default_activation_function"], FakeIdentity)


if __name__ == "__main__":
    unittest.main()
