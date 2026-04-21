"""Embedding adapter — Voyage AI or OpenAI, selected via EMBEDDING_PROVIDER."""
from __future__ import annotations

from typing import List

from voice.config import (
    EMBEDDING_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    VOYAGE_API_KEY,
    VOYAGE_MODEL,
)

_voyage_client = None
_openai_client = None


def _get_voyage():
    global _voyage_client
    if _voyage_client is None:
        if not VOYAGE_API_KEY:
            raise RuntimeError(
                "VOYAGE_API_KEY is not set. Either set it in .env or switch "
                "EMBEDDING_PROVIDER to openai."
            )
        import voyageai  # lazy import
        _voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    return _voyage_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Either set it in .env or switch "
                "EMBEDDING_PROVIDER to voyage."
            )
        from openai import OpenAI  # lazy import
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed a batch of documents (used at ingest time)."""
    if not texts:
        return []
    if EMBEDDING_PROVIDER == "voyage":
        client = _get_voyage()
        result = client.embed(texts, model=VOYAGE_MODEL, input_type="document")
        return result.embeddings
    if EMBEDDING_PROVIDER == "openai":
        client = _get_openai()
        result = client.embeddings.create(input=texts, model=OPENAI_EMBEDDING_MODEL)
        return [d.embedding for d in result.data]
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER!r}. Use 'voyage' or 'openai'."
    )


def embed_query(text: str) -> List[float]:
    """Embed a single query string (used at retrieval time)."""
    if EMBEDDING_PROVIDER == "voyage":
        client = _get_voyage()
        result = client.embed([text], model=VOYAGE_MODEL, input_type="query")
        return result.embeddings[0]
    if EMBEDDING_PROVIDER == "openai":
        client = _get_openai()
        result = client.embeddings.create(input=[text], model=OPENAI_EMBEDDING_MODEL)
        return result.data[0].embedding
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER!r}. Use 'voyage' or 'openai'."
    )
