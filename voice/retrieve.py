"""Retrieval layer — queries corpus and dialogues collections in one call,
merges the results, and hands back typed RetrievedChunk objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import chromadb

from voice.config import (
    CHROMA_PATH,
    CORPUS_COLLECTION,
    DIALOGUES_COLLECTION,
)
from voice.embeddings import embed_query


@dataclass
class RetrievedChunk:
    text: str
    source_type: str           # novel / essay / review / dialogue
    source_file: str           # filename for corpus; "(dialogue)" for dialogue
    confidence: Optional[str]  # only set for dialogue rows
    distance: float            # lower is closer
    metadata: Dict = field(default_factory=dict)


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _client


def _query_one(
    collection_name: str,
    query_embedding: List[float],
    n_results: int,
    where: Optional[Dict],
) -> List[Dict]:
    """Return a list of {id, text, metadata, distance}. Empty if collection missing."""
    client = _get_client()
    try:
        col = client.get_collection(name=collection_name)
    except Exception:
        # Collection hasn't been created yet — treat as empty.
        return []

    kwargs: Dict = {
        "query_embeddings": [query_embedding],
        "n_results": max(1, n_results),
    }
    if where:
        kwargs["where"] = where

    try:
        result = col.query(**kwargs)
    except Exception:
        return []

    ids = (result.get("ids") or [[]])[0]
    if not ids:
        return []
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[0.0] * len(ids)])[0]

    out: List[Dict] = []
    for i, doc_id in enumerate(ids):
        out.append(
            {
                "id": doc_id,
                "text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": float(dists[i]) if i < len(dists) else 0.0,
            }
        )
    return out


def retrieve(
    query: str,
    n_results: int = 5,
    collections: Optional[List[str]] = None,
    filter_by: Optional[Dict] = None,
) -> List[RetrievedChunk]:
    """Retrieve up to ``n_results`` chunks, merged across the listed collections.

    ``filter_by`` is passed to each collection as a Chroma ``where`` clause.
    Use it for simple equality filters like ``{"source_type": "essay"}`` or
    ``{"confidence": "exact"}``. Fields that don't exist on a given collection
    will simply match nothing there, which is fine.
    """
    collections = collections or [CORPUS_COLLECTION, DIALOGUES_COLLECTION]
    filter_by = filter_by or {}

    query_emb = embed_query(query)

    # Over-fetch a bit from each collection so we can merge+rerank cleanly.
    per_collection_n = max(n_results, 3)

    merged: List[Dict] = []
    for col_name in collections:
        items = _query_one(col_name, query_emb, per_collection_n, filter_by or None)
        merged.extend(items)

    merged.sort(key=lambda x: x["distance"])
    merged = merged[:n_results]

    out: List[RetrievedChunk] = []
    for item in merged:
        meta = item["metadata"] or {}
        src_type = meta.get("source_type", "unknown")
        if src_type == "dialogue":
            # For dialogues, use the quote itself as a display label.
            src_file = meta.get("source_file") or (meta.get("quote", "")[:40] or "(dialogue)")
        else:
            src_file = meta.get("source_file", "")
        out.append(
            RetrievedChunk(
                text=item["text"],
                source_type=src_type,
                source_file=src_file,
                confidence=meta.get("confidence"),
                distance=item["distance"],
                metadata=meta,
            )
        )
    return out
