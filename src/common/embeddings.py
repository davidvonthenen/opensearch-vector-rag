"""Embedding utilities using sentence-transformers."""
from __future__ import annotations

import os
# Force pure-Python protobuf to suppress MessageFactory C++ extension errors
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import platform
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence, Union, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import Settings
from .logging import get_logger

LOGGER = get_logger(__name__)

_DTYPE_BY_NAME = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _is_qwen3_embedding(model_name: str) -> bool:
    name = model_name.lower()
    return "qwen3-embedding" in name


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_runtime_options(
    model_name: str,
) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Pick conservative runtime settings for known-problematic combinations.

    Qwen3 embedding models on macOS currently have multiple reported issues when
    the default SDPA path is used. On Apple Silicon we proactively switch to
    eager attention and avoid the model's default BF16 load dtype.
    """
    device = _default_device()
    attn_implementation: Optional[str] = None
    torch_dtype_name: Optional[str] = None
    padding_side: Optional[str] = None

    if _is_qwen3_embedding(model_name):
        # Recommended by the model card for correct last-token pooling behavior.
        padding_side = "left"

        if _is_macos():
            # SentenceTransformers defaults to SDPA when available. That path is
            # currently unstable for Qwen3 embeddings on macOS/MPS, so force the
            # safer eager implementation instead.
            attn_implementation = "eager"

            # The model config advertises BF16. The M3 stack trace shows the crash
            # occurs in a BF16 MPS matmul path, so pin to safer dtypes on macOS.
            if device == "mps":
                torch_dtype_name = "float16"
            else:
                torch_dtype_name = "float32"

    env_device = os.getenv("EMBEDDING_DEVICE")
    if env_device:
        device = env_device

    env_attn = os.getenv("EMBEDDING_ATTN_IMPLEMENTATION")
    if env_attn:
        attn_implementation = env_attn

    env_dtype = os.getenv("EMBEDDING_TORCH_DTYPE")
    if env_dtype:
        torch_dtype_name = env_dtype.strip().lower()

    env_padding_side = os.getenv("EMBEDDING_PADDING_SIDE")
    if env_padding_side:
        padding_side = env_padding_side.strip().lower()

    if torch_dtype_name is not None and torch_dtype_name not in _DTYPE_BY_NAME:
        valid = ", ".join(sorted(_DTYPE_BY_NAME))
        raise ValueError(
            f"Unsupported EMBEDDING_TORCH_DTYPE={torch_dtype_name!r}; expected one of: {valid}"
        )

    return device, attn_implementation, torch_dtype_name, padding_side


@lru_cache(maxsize=4)
def _load_model(
    model_name: str,
    device: str,
    attn_implementation: Optional[str],
    torch_dtype_name: Optional[str],
    padding_side: Optional[str],
) -> SentenceTransformer:
    """Lazy-load and cache the sentence-transformers model."""
    model_kwargs: dict[str, object] = {}
    tokenizer_kwargs: dict[str, object] = {}

    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if torch_dtype_name:
        model_kwargs["torch_dtype"] = _DTYPE_BY_NAME[torch_dtype_name]
    if padding_side:
        tokenizer_kwargs["padding_side"] = padding_side

    if _is_qwen3_embedding(model_name) and _is_macos():
        LOGGER.info(
            (
                "Loading embedding model '%s' on %s with macOS-safe settings "
                "(attn_implementation=%s, torch_dtype=%s, padding_side=%s)"
            ),
            model_name,
            device,
            attn_implementation or "default",
            torch_dtype_name or "default",
            padding_side or "default",
        )

    return SentenceTransformer(
        model_name,
        device=device,
        model_kwargs=model_kwargs or None,
        tokenizer_kwargs=tokenizer_kwargs or None,
    )


@dataclass
class EmbeddingModel:
    """Wrapper around a sentence-transformers embedding model with lazy init."""

    settings: Settings
    _cached_model: Optional[SentenceTransformer] = None

    @property
    def model_name(self) -> str:
        """
        Prefer `embedding_model_name` if present for backward-compat,
        otherwise use `embedding_model` (the current field name).
        """
        name = getattr(self.settings, "embedding_model_name", None)
        if not name:
            name = getattr(self.settings, "embedding_model", None)
        if not name:
            # Sensible default for this project
            name = "thenlper/gte-small"
        return name

    @property
    def model(self) -> SentenceTransformer:
        if self._cached_model is None:
            device, attn_implementation, torch_dtype_name, padding_side = _resolve_runtime_options(
                self.model_name
            )
            self._cached_model = _load_model(
                self.model_name,
                device,
                attn_implementation,
                torch_dtype_name,
                padding_side,
            )
        return self._cached_model

    @property
    def dimension(self) -> int:
        """
        Returns the embedding dimension. If `Settings` declares an explicit
        `embedding_dimension` override, use it; otherwise ask the model.
        """
        dim = getattr(self.settings, "embedding_dimension", None)
        if dim is not None:
            return int(dim)

        # Most SentenceTransformer models expose this method.
        try:
            return int(self.model.get_sentence_embedding_dimension())
        except Exception:
            # Fallback: run a tiny encode to infer dimensionality.
            vec = self.encode(["_probe_"])[0]
            return int(vec.shape[-1])

    def encode(self, texts: Iterable[str]) -> List[np.ndarray]:
        """
        Encode a list/iterable of texts to a list of 1-D numpy arrays
        (L2-normalized) for direct use with cosine-similarity kNN.
        """
        items: List[str] = list(texts)
        if len(items) == 0:
            return []

        arr = self.model.encode(
            items,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Ensure we always return List[np.ndarray] of shape (D,)
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return [arr[i] for i in range(arr.shape[0])]
        if isinstance(arr, np.ndarray) and arr.ndim == 1 and len(items) == 1:
            return [arr]
        return [np.asarray(v, dtype=float) for v in arr]


def to_list(vec: Union[np.ndarray, Sequence[float], List[float]]) -> List[float]:
    """Convert a vector-like object to a Python list of floats."""
    if isinstance(vec, np.ndarray):
        return vec.astype(float).tolist()
    return [float(x) for x in vec]


__all__ = ["EmbeddingModel", "to_list"]