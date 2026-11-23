from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


@dataclass(frozen=True)
class ServiceSettings:
    model_name: str = "BAAI/bge-reranker-base"
    model_revision: str | None = None
    cache_dir: str | None = None
    device: str = "auto"
    batch_size: int = 16
    max_input_length: int = 512
    max_concurrent_requests: int = 1

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.max_input_length < 1:
            raise ValueError("max_input_length must be positive.")
        if self.max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be positive.")

        device = self.device.strip().lower()
        if device not in {"auto", "cpu", "cuda", "mps"} and not (
            device.startswith("cuda:") and device.removeprefix("cuda:").isdigit()
        ):
            raise ValueError("device must be auto, cpu, cuda, cuda:N, or mps.")

    @classmethod
    def from_env(cls) -> ServiceSettings:
        revision = os.getenv("RERANKER_MODEL_REVISION", "").strip() or None
        cache_dir = os.getenv("RERANKER_CACHE_DIR", "").strip() or None
        if cache_dir is not None:
            cache_dir = str(Path(cache_dir).expanduser())

        return cls(
            model_name=os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base"),
            model_revision=revision,
            cache_dir=cache_dir,
            device=os.getenv("RERANKER_DEVICE", "auto"),
            batch_size=_positive_int("RERANKER_BATCH_SIZE", 16),
            max_input_length=_positive_int("RERANKER_MAX_INPUT_LENGTH", 512),
            max_concurrent_requests=_positive_int("RERANKER_MAX_CONCURRENT_REQUESTS", 1),
        )
