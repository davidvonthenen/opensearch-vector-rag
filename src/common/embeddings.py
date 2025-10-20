"""Embedding utilities using sentence-transformers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Settings


@dataclass
class EmbeddingModel:
    """Wrapper around a sentence-transformers embedding model."""

    settings: Settings

    @property
    def dimension(self) -> int:
        try:
            return self._model.get_sentence_embedding_dimension()
        except AttributeError:
            return self.settings.embedding_dimension

    @property
    def _model(self) -> SentenceTransformer:
        return _load_model(self.settings.embedding_model_name)

    def encode(self, texts: Iterable[str], normalize: bool = True) -> np.ndarray:
        """Encode texts into dense vectors, optionally normalizing for cosine similarity."""

        embeddings = self._model.encode(
            list(texts),
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def to_list(vec: np.ndarray) -> List[float]:
    """Convert a numpy vector to a Python list of floats."""

    return vec.astype(float).tolist()


__all__ = ["EmbeddingModel", "to_list"]
