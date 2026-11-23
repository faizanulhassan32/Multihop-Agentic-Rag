from __future__ import annotations

from typing import Any

from .config import ServiceSettings


def resolve_device(requested: str, torch_module: Any | None = None) -> str:
    if torch_module is None:
        import torch as torch_module

    preference = requested.strip().lower()
    cuda_available = torch_module.cuda.is_available()
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if preference == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if preference == "cpu":
        return "cpu"
    if preference == "mps":
        if not mps_available:
            raise RuntimeError("RERANKER_DEVICE=mps was requested, but MPS is unavailable.")
        return "mps"
    if preference == "cuda" or preference.startswith("cuda:"):
        if not cuda_available:
            raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
        return preference
    raise ValueError("device must be auto, cpu, cuda, cuda:N, or mps.")


def load_cross_encoder(settings: ServiceSettings) -> Any:
    import torch
    from sentence_transformers import CrossEncoder

    device = resolve_device(settings.device, torch)
    return CrossEncoder(
        settings.model_name,
        revision=settings.model_revision,
        cache_dir=settings.cache_dir,
        device=device,
        max_length=settings.max_input_length,
        default_activation_function=torch.nn.Identity(),
    )
