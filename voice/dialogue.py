"""Ad-hoc dialogue fragments — quick capture of things she said or that you
inferred she would say. Lives in its own ChromaDB collection so retrieval can
distinguish a quoted fragment from a paragraph of her published prose.

Every add also appends to data/dialogues.jsonl as a human-readable backup.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

import chromadb

from voice.config import (
    CHROMA_PATH,
    DIALOGUES_COLLECTION,
    DIALOGUES_JSONL,
)
from voice.embeddings import embed_documents

Confidence = Literal["exact", "paraphrase", "inference"]
_VALID_CONFIDENCE = ("exact", "paraphrase", "inference")

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    if _collection is None:
        _collection = _client.get_or_create_collection(
            name=DIALOGUES_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_dialogue(
    quote: str,
    context: Optional[str] = None,
    your_note: Optional[str] = None,
    confidence: Confidence = "paraphrase",
) -> Dict:
    """Add a dialogue fragment. Writes to ChromaDB AND appends to JSONL."""
    if confidence not in _VALID_CONFIDENCE:
        raise ValueError(
            f"confidence must be one of {_VALID_CONFIDENCE}, got {confidence!r}"
        )
    quote = quote.strip()
    if not quote:
        raise ValueError("quote must be non-empty")

    record_id = f"dialogue:{uuid.uuid4().hex}"
    created_at = datetime.now(timezone.utc).isoformat()

    # Build the doc text used for embedding. Include context/note so the
    # fragment retrieves well even if the quote alone is too sparse.
    doc_text = quote
    if context:
        doc_text = f"{doc_text}\n\n[语境] {context}"
    if your_note:
        doc_text = f"{doc_text}\n\n[笔记] {your_note}"

    metadata: Dict = {
        "source_type": "dialogue",
        "source_file": "(dialogue)",  # uniform schema with corpus
        "quote": quote,
        "confidence": confidence,
        "created_at": created_at,
    }
    if context:
        metadata["context"] = context
    if your_note:
        metadata["your_note"] = your_note

    embeddings = embed_documents([doc_text])
    collection = _get_collection()
    collection.add(
        ids=[record_id],
        documents=[doc_text],
        metadatas=[metadata],
        embeddings=embeddings,
    )

    record = {
        "id": record_id,
        "quote": quote,
        "context": context,
        "your_note": your_note,
        "confidence": confidence,
        "created_at": created_at,
    }
    DIALOGUES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with DIALOGUES_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def list_dialogues() -> List[Dict]:
    """Return every dialogue ever added, oldest first, from the JSONL backup."""
    if not DIALOGUES_JSONL.exists():
        return []
    out: List[Dict] = []
    with DIALOGUES_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip a corrupt line rather than crash; user can fix it manually.
                continue
    return out
