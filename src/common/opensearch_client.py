"""OpenSearch client utilities."""
from __future__ import annotations

from typing import List, Optional

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError

from .config import Settings
from .logging import get_logger


LOGGER = get_logger(__name__)


def create_client(settings: Settings) -> OpenSearch:
    """Create an OpenSearch client using provided settings."""
    LOGGER.info(
        "Connecting to OpenSearch at %s:%s", settings.opensearch_host, settings.opensearch_port
    )
    return OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port, "scheme": "http"}],
        http_compress=True,
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
    )


def ensure_index(settings: Settings, dim: int) -> None:
    """Ensure that the target index exists with the correct mapping."""
    client = create_client(settings)
    index_name = settings.opensearch_index
    try:
        if client.indices.exists(index=index_name):
            LOGGER.info("OpenSearch index '%s' already exists", index_name)
            return
    except NotFoundError:
        LOGGER.info("OpenSearch index '%s' does not exist, creating", index_name)

    body = {
        "settings": {
            "index": {
                # Enable k-NN index structures
                "knn": True,
                # Lucene engine ignores ef_search and derives it from k,
                # but keeping this setting is harmless if you switch engines later.
                "knn.algo_param.ef_search": 256,
            }
        },
        "mappings": {
            "properties": {
                "path": {"type": "keyword"},
                "title": {"type": "keyword"},
                "category": {"type": "keyword"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        # You can add "parameters": {"m": 16, "ef_construction": 128} here if desired.
                    },
                },
            }
        },
    }
    LOGGER.info("Creating OpenSearch index '%s'", index_name)
    client.indices.create(index=index_name, body=body)


def knn_search(
    client: OpenSearch,
    index: str,
    query_vec: List[float],
    k: int,
    num_candidates: Optional[int] = None,
):
    """
    Execute a vector k-NN search against the 'embedding' field.

    OpenSearch 3.x k-NN DSL uses the vector field name as the key and
    expects the query vector under 'vector' (not 'query_vector'), and no 'field' key.
    """
    # Core OS 3.x body
    body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vec,
                    "k": k,
                }
            }
        },
        "_source": ["path", "title", "category", "text"],
    }

    # Optional: if callers pass num_candidates, you *could* map it to rescoring oversampling.
    # This is mainly useful for on-disk vectors; Lucene in-memory often doesn't need it.
    if num_candidates and num_candidates > k:
        body["query"]["knn"]["embedding"]["rescore"] = {
            "oversample_factor": float(num_candidates) / float(max(k, 1))
        }

    response = client.search(index=index, body=body)
    LOGGER.info("OpenSearch knn_search took %.2f ms", response.get("took", 0.0))
    return response


__all__ = ["create_client", "ensure_index", "knn_search"]
