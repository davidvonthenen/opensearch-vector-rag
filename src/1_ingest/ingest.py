"""BBC dataset ingestion script."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterator, Sequence

from tqdm import tqdm

from ..common.config import Settings, load_settings
from ..common.embeddings import EmbeddingModel, to_list
from ..common.logging import get_logger
from ..common.opensearch_client import create_client, ensure_index

LOGGER = get_logger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest dataset into OpenSearch")
    parser.add_argument("--data-dir", type=str, default="./bbc", help="Path to dataset root")
    parser.add_argument("--index-name", type=str, default=None, help="Target OpenSearch index")
    parser.add_argument(
        "--vector-chunk-size",
        type=int,
        default=1000,
        help="Maximum characters per vector chunk",
    )
    parser.add_argument(
        "--vector-chunk-overlap",
        type=int,
        default=200,
        help="Character overlap between consecutive vector chunks",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Vector batch size (paragraphs)",
    )
    return parser.parse_args(argv)


def _iter_documents(data_dir: Path) -> Iterator[dict[str, str]]:
    categories = [p for p in data_dir.iterdir() if p.is_dir()]
    if not categories:
        raise FileNotFoundError(f"No category folders found in {data_dir}")

    for category_dir in categories:
        for file_path in category_dir.glob("*.txt"):
            text = file_path.read_text(encoding="utf-8")
            rel_path = file_path.relative_to(data_dir).as_posix()
            yield {
                "path": rel_path,
                "title": file_path.stem,
                "category": category_dir.name,
                "text": text,
            }


def _iter_chunks(text: str, chunk_size: int, chunk_overlap: int) -> Iterator[str]:
    if chunk_size <= 0:
        raise ValueError("vector_chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("vector_chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("vector_chunk_overlap must be smaller than vector_chunk_size")

    if not text:
        yield text
        return

    step = chunk_size - chunk_overlap
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        yield text[start:end]
        if end >= text_length:
            break
        start += step


def _doc_id(path: str, chunk_index: int) -> str:
    return hashlib.sha1(f"{path}:{chunk_index}".encode("utf-8")).hexdigest()


def ingest(
    data_dir: Path,
    settings: Settings,
    vector_chunk_size: int = 1000,
    vector_chunk_overlap: int = 200,
    batch_size: int = 32,
) -> None:
    client = create_client(settings)
    embedder = EmbeddingModel(settings)
    ensure_index(settings, embedder.dimension)

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    total_indexed = 0
    progress = tqdm(desc="Indexing", unit="chunks")
    try:
        for doc in _iter_documents(data_dir):
            client.delete_by_query(
                index=settings.opensearch_index,
                body={"query": {"term": {"path": doc["path"]}}},
                conflicts="proceed",
            )

            chunk_texts = list(_iter_chunks(doc["text"], vector_chunk_size, vector_chunk_overlap))
            for batch_start in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[batch_start : batch_start + batch_size]
                embeddings = embedder.encode(batch_texts)
                for offset, embedding in enumerate(embeddings):
                    chunk_index = batch_start + offset
                    chunk_doc = {**doc, "text": batch_texts[offset]}
                    embedding_list = to_list(embedding)
                    doc_id = _doc_id(doc["path"], chunk_index)
                    response = client.index(
                        index=settings.opensearch_index,
                        id=doc_id,
                        body={**chunk_doc, "embedding": embedding_list},
                    )
                    status = response.get("result", "unknown")
                    LOGGER.info(
                        "Indexed %s chunk %d with status '%s'",
                        doc["path"],
                        chunk_index,
                        status,
                    )
                    progress.update(1)
                    total_indexed += 1
    finally:
        progress.close()

    if total_indexed == 0:
        LOGGER.warning("No documents were ingested.")
    else:
        client.indices.refresh(index=settings.opensearch_index)
        LOGGER.info("Ingested %d chunks into index '%s'", total_indexed, settings.opensearch_index)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")

    settings = load_settings()
    if args.index_name:
        settings.opensearch_index = args.index_name

    ingest(
        data_dir=data_dir,
        settings=settings,
        vector_chunk_size=args.vector_chunk_size,
        vector_chunk_overlap=args.vector_chunk_overlap,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
