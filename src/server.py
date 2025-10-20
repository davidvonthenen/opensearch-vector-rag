"""Flask server exposing an OpenAI-style chat completions endpoint."""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Sequence

from flask import Flask, jsonify, request

from .common.config import Settings, load_settings
from .common.embeddings import EmbeddingModel, to_list
from .common.logging import get_logger
from .common.opensearch_client import create_client, ensure_index, knn_search

LOGGER = get_logger(__name__)
APP = Flask(__name__)


class LLMBackend:
    """Abstract interface for language model backends."""

    def chat(self, messages: Sequence[Dict[str, str]], **gen_kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError


class LlamaCppBackend(LLMBackend):
    """llama-cpp-python implementation of :class:`LLMBackend`."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._llama = None

    def _get_model(self):
        if self._llama is None:
            LOGGER.info("Loading llama model from %s", self.settings.llama_model_path)
            from llama_cpp import Llama

            self._llama = Llama(
                model_path=self.settings.llama_model_path,
                n_ctx=self.settings.llama_ctx,
                n_threads=self.settings.llama_n_threads,
                n_gpu_layers=self.settings.llama_n_gpu_layers,
            )
        return self._llama

    def chat(self, messages: Sequence[Dict[str, str]], **gen_kwargs: Any) -> Dict[str, Any]:
        model = self._get_model()
        return model.create_chat_completion(messages=list(messages), **gen_kwargs)


class MCPBackend(LLMBackend):
    """Placeholder for future MCP integration."""

    def chat(self, messages: Sequence[Dict[str, str]], **gen_kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError("MCP backend not yet implemented")


SETTINGS = load_settings()
EMBEDDER = EmbeddingModel(SETTINGS)
OPENSEARCH_CLIENT = create_client(SETTINGS)
ensure_index(SETTINGS, EMBEDDER.dimension)
LLM = LlamaCppBackend(SETTINGS)


def _extract_user_question(messages: Sequence[Dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "").strip()
            if content:
                return content
    raise ValueError("No user message found in request")


def _trim_snippet(text: str, max_length: int = 1200) -> str:
    if len(text) <= max_length:
        return text
    trimmed = text[:max_length]
    last_space = trimmed.rfind(" ")
    if last_space == -1:
        return trimmed + "..."
    return trimmed[:last_space] + "..."



def _build_context_block(hits: List[Dict[str, Any]]) -> str:
    parts = []
    for idx, hit in enumerate(hits, start=1):
        source = hit.get("_source", {})
        snippet = _trim_snippet(source.get("text", ""))
        parts.append(
            "[DOC {idx} | source: {category}/{title} | path: {path}]\n{snippet}".format(
                idx=idx,
                category=source.get("category", "unknown"),
                title=source.get("title", "unknown"),
                path=source.get("path", ""),
                snippet=snippet,
            )
        )
    return "\n\n".join(parts)


def _rag_hits_from_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    hits = []
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        hits.append(
            {
                "path": source.get("path", ""),
                "title": source.get("title", ""),
                "category": source.get("category", ""),
                "score": float(hit.get("_score", 0.0)),
            }
        )
    return hits


def _compose_messages(question: str, context_block: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a fact-focused assistant. Use only the provided context snippets. "
        "If the answer is not grounded in the snippets, respond with 'I don't know.' "
        "Cite sources inline like [source: ] when drawing from a snippet."
    )
    user_prompt = (
        f"Question:\n{question}\n\nContext:\n{context_block if context_block else 'No context available.'}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _normalize_rag_params(payload: Dict[str, Any], settings: Settings) -> tuple[int, int]:
    rag_config = payload.get("rag", {}) if isinstance(payload.get("rag"), dict) else {}
    k = int(rag_config.get("k", settings.rag_top_k))
    num_candidates = int(rag_config.get("num_candidates", settings.rag_num_candidates))
    return k, num_candidates


@APP.post("/v1/chat/completions")
def chat_completions():
    payload = request.get_json(force=True, silent=False)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    try:
        question = _extract_user_question(payload.get("messages", []))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    k, num_candidates = _normalize_rag_params(payload, SETTINGS)
    embedding = EMBEDDER.encode([question])[0]
    search_response = knn_search(
        OPENSEARCH_CLIENT,
        SETTINGS.opensearch_index,
        to_list(embedding),
        k=k,
        num_candidates=num_candidates,
    )
    hits = search_response.get("hits", {}).get("hits", [])
    context_block = _build_context_block(hits)
    messages = _compose_messages(question, context_block)

    llm_response = LLM.chat(
        messages,
        temperature=float(payload.get("temperature", 0.2)),
        top_p=float(payload.get("top_p", 0.95)),
        max_tokens=int(payload.get("max_tokens", 512)),
    )

    usage = llm_response.get("usage", {})
    choice = llm_response.get("choices", [{}])[0]
    assistant_message = choice.get("message", {})

    response_body = {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", "local-llama"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": assistant_message.get("role", "assistant"),
                    "content": assistant_message.get("content", ""),
                },
                "finish_reason": choice.get("finish_reason", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
        "rag_context": {
            "index": SETTINGS.opensearch_index,
            "k": k,
            "num_candidates": num_candidates,
            "hits": _rag_hits_from_response(search_response),
        },
    }
    return jsonify(response_body)


def main() -> None:
    LOGGER.info(
        "Starting server on %s:%s", SETTINGS.server_host, SETTINGS.server_port
    )
    APP.run(host=SETTINGS.server_host, port=SETTINGS.server_port)


if __name__ == "__main__":
    main()
