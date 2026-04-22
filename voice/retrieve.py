"""Retrieval layer.

Does two things:

  1. Vector-retrieves from the corpus and dialogues collections, merges by
     distance, and returns up to ``n_results`` chunks.

  2. **Bidirectional retrieval.** For every corpus chunk that ends up in the
     top-n, we look up the dialogue fragments whose ``linked_sources`` list
     names that corpus file, and append them to the result set. This lets the
     user "bind" an out-of-band quote ("she told me X while discussing the
     Tolkien review") to a specific published piece — so that whenever the
     published piece surfaces, the linked quote rides along.

Linked dialogues appear after the pure vector results in the returned list,
tagged in metadata (``_retrieval_reason = "linked_via:<stem>"``) so callers
can render them differently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

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
    source_type: str           # novel / essay / review / diary / dialogue
    source_file: str           # filename for corpus; "(dialogue)" for dialogue
    confidence: Optional[str]  # only set for dialogue rows
    stance: Optional[str]      # only set for corpus rows (affirm/resist/ambivalent/expository)
    distance: float            # lower is closer; float('inf') for linked items
    metadata: Dict = field(default_factory=dict)


_client = None

_KNOWN_EXTS = (".md", ".txt", ".markdown")


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _client


def _stem_of(filename: Optional[str]) -> str:
    """Return the bare stem of a filename (no extension)."""
    if not filename:
        return ""
    low = filename.lower()
    for ext in _KNOWN_EXTS:
        if low.endswith(ext):
            return filename[: -len(ext)]
    return filename


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


def _fetch_linked_dialogues(stems: Set[str], exclude_ids: Set[str]) -> List[Dict]:
    """Return dialogue rows whose ``linked_sources`` intersects ``stems``.

    Chroma's metadata filter doesn't support substring match on strings, so
    we fetch the full dialogues collection and filter in Python. That's fine
    for a handwritten corpus of quotes (expected size: tens to low hundreds).
    """
    if not stems:
        return []
    client = _get_client()
    try:
        col = client.get_collection(name=DIALOGUES_COLLECTION)
    except Exception:
        return []
    try:
        result = col.get()
    except Exception:
        return []

    ids = result.get("ids") or []
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []

    out: List[Dict] = []
    for i, doc_id in enumerate(ids):
        if doc_id in exclude_ids:
            continue
        meta = metas[i] if i < len(metas) else {}
        if not meta:
            continue
        link_str = meta.get("linked_sources")
        if not link_str:
            continue
        my_stems = {s.strip() for s in str(link_str).split(",") if s.strip()}
        hit = my_stems & stems
        if not hit:
            continue
        # Annotate why this came back.
        annotated_meta = dict(meta)
        annotated_meta["_retrieval_reason"] = "linked_via:" + ",".join(sorted(hit))
        out.append(
            {
                "id": doc_id,
                "text": docs[i] if i < len(docs) else "",
                "metadata": annotated_meta,
                "distance": float("inf"),  # marker: not a vector match
            }
        )
    return out


def _to_chunk(item: Dict) -> RetrievedChunk:
    meta = item["metadata"] or {}
    src_type = meta.get("source_type", "unknown")
    if src_type == "dialogue":
        src_file = (
            meta.get("source_file")
            or (meta.get("quote", "")[:40] or "(dialogue)")
        )
    else:
        src_file = meta.get("source_file", "")
    return RetrievedChunk(
        text=item["text"],
        source_type=src_type,
        source_file=src_file,
        confidence=meta.get("confidence"),
        stance=meta.get("stance") if src_type != "dialogue" else None,
        distance=item["distance"],
        metadata=meta,
    )


def retrieve(
    query: str,
    n_results: int = 5,
    collections: Optional[List[str]] = None,
    filter_by: Optional[Dict] = None,
) -> List[RetrievedChunk]:
    """Retrieve up to ``n_results`` vector matches, plus every dialogue whose
    ``linked_sources`` names one of the corpus files that came back.

    The returned list has vector-matched chunks first (sorted by distance),
    then linked dialogues appended (in no particular order). Linked items have
    ``distance == float('inf')`` and ``metadata['_retrieval_reason']`` set.
    """
    collections = collections or [CORPUS_COLLECTION, DIALOGUES_COLLECTION]
    filter_by = filter_by or {}

    query_emb = embed_query(query)
    per_collection_n = max(n_results, 3)

    # 1. Pure vector retrieval across the requested collections.
    vector_items: List[Dict] = []
    for col_name in collections:
        vector_items.extend(
            _query_one(col_name, query_emb, per_collection_n, filter_by or None)
        )
    vector_items.sort(key=lambda x: x["distance"])
    vector_items = vector_items[:n_results]

    # 2. Linked dialogues — only when the dialogues collection is in play and
    #    at least one corpus chunk came back.
    linked_items: List[Dict] = []
    if DIALOGUES_COLLECTION in collections:
        stems: Set[str] = set()
        for item in vector_items:
            meta = item.get("metadata") or {}
            if meta.get("source_type") == "dialogue":
                continue
            # Prefer source_stem if ingest.py wrote it; fall back to filename.
            stem = meta.get("source_stem") or _stem_of(meta.get("source_file"))
            if stem:
                stems.add(stem)
        if stems:
            seen_ids = {item["id"] for item in vector_items}
            linked_items = _fetch_linked_dialogues(stems, seen_ids)

    return [_to_chunk(it) for it in vector_items] + [_to_chunk(it) for it in linked_items]
