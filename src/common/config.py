"""Configuration loading utilities for the RAG service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime configuration parameters for the application."""

    opensearch_host: str = "127.0.0.1"
    opensearch_port: int = 9200
    opensearch_index: str = "bbc"

    embedding_model_name: str = "thenlper/gte-small"
    embedding_dimension: int = 384

    llama_model_path: str = "Llama-Pro-8B/llama-pro-8b.Q8_0.gguf"
    llama_ctx: int = 8192
    llama_n_threads: int = max(os.cpu_count() or 1, 1)
    llama_n_gpu_layers: int = -1

    rag_top_k: int = 5
    rag_num_candidates: int = 50

    server_host: str = "0.0.0.0"
    server_port: int = 8000


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc


def load_settings(env_path: Optional[str] = None) -> Settings:
    """Load application settings from environment variables and optional .env file."""

    if env_path:
        load_dotenv(env_path, override=False)
    else:
        load_dotenv(override=False)

    return Settings(
        opensearch_host=os.getenv("OPENSEARCH_HOST", Settings.opensearch_host),
        opensearch_port=_get_int("OPENSEARCH_PORT", Settings.opensearch_port),
        opensearch_index=os.getenv("OPENSEARCH_INDEX", Settings.opensearch_index),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME", Settings.embedding_model_name
        ),
        embedding_dimension=_get_int(
            "EMBEDDING_DIMENSION", Settings.embedding_dimension
        ),
        llama_model_path=os.getenv(
            "LLAMA_MODEL_PATH",
            str(Path.home() / "models" / Settings.llama_model_path),
        ),
        llama_ctx=_get_int("LLAMA_CTX", Settings.llama_ctx),
        llama_n_threads=_get_int("LLAMA_N_THREADS", Settings.llama_n_threads),
        llama_n_gpu_layers=_get_int("LLAMA_N_GPU_LAYERS", Settings.llama_n_gpu_layers),
        rag_top_k=_get_int("RAG_TOP_K", Settings.rag_top_k),
        rag_num_candidates=_get_int(
            "RAG_NUM_CANDIDATES", Settings.rag_num_candidates
        ),
        server_host=os.getenv("SERVER_HOST", Settings.server_host),
        server_port=_get_int("SERVER_PORT", Settings.server_port),
    )


__all__ = ["Settings", "load_settings"]
