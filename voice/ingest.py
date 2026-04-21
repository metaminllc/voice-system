"""Ingest a corpus file (novel / essay / review) into ChromaDB.

Chunking strategies differ by type:

  * ``novel``  — build semantic units of 200..600 Chinese characters by
                 concatenating adjacent sentences inside paragraphs, then
                 attach the immediately preceding and following sentence
                 as a context window (stored in metadata).
  * ``essay``  — split on paragraph boundaries (blank line, or a single
                 newline followed by an indent). No hard length cap.
  * ``review`` — the entire file becomes a single chunk, tagged with the
                 book and her judgment direction.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional

import chromadb

from voice.config import (
    CHROMA_PATH,
    CORPUS_COLLECTION,
    NOVEL_MAX_CHARS,
    NOVEL_MIN_CHARS,
)
from voice.embeddings import embed_documents

CorpusType = Literal["novel", "essay", "review"]

# Sentence boundary across Chinese and Western punctuation.
_SENTENCE_RE = re.compile(r"(?<=[。！？!?.…])\s*")

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    if _collection is None:
        _collection = _client.get_or_create_collection(
            name=CORPUS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---- chunking ------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _chunk_novel(text: str) -> List[Dict]:
    """Return chunks with keys: text, context_prev, context_next."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # (paragraph_index, sentence) tuples in reading order.
    tagged: List[tuple[int, str]] = []
    for p_idx, para in enumerate(paragraphs):
        sents = _split_sentences(para)
        # If a paragraph has no sentence punctuation, treat it as one sentence.
        if not sents:
            sents = [para]
        for s in sents:
            tagged.append((p_idx, s))

    chunks: List[Dict] = []
    i = 0
    n = len(tagged)
    while i < n:
        chunk_sents: List[str] = []
        chunk_len = 0
        j = i
        while j < n:
            s = tagged[j][1]
            # Hard cap: if adding this sentence would break the max AND we
            # already have enough content, stop.
            if chunk_len + len(s) > NOVEL_MAX_CHARS and chunk_len >= NOVEL_MIN_CHARS:
                break
            chunk_sents.append(s)
            chunk_len += len(s)
            j += 1
            # Prefer to stop at a paragraph boundary once we're past the floor.
            if chunk_len >= NOVEL_MIN_CHARS:
                if j < n and tagged[j][0] != tagged[j - 1][0]:
                    break
            # Safety: if a single sentence blows through max on its own, stop.
            if chunk_len >= NOVEL_MAX_CHARS:
                break

        prev_ctx = tagged[i - 1][1] if i > 0 else ""
        next_ctx = tagged[j][1] if j < n else ""
        chunks.append(
            {
                "text": "".join(chunk_sents),
                "context_prev": prev_ctx,
                "context_next": next_ctx,
            }
        )
        if j == i:  # defensive: never make zero-progress
            j = i + 1
        i = j
    return chunks


def _chunk_essay(text: str) -> List[Dict]:
    """Split on blank-line OR single-newline-plus-indent paragraph breaks."""
    # Split on a blank line, or on a newline immediately followed by whitespace + non-whitespace.
    parts = re.split(r"\n\s*\n|\n(?=[ \t\u3000]+\S)", text)
    paragraphs = [p.strip() for p in parts if p.strip()]
    return [{"text": p} for p in paragraphs]


def _chunk_review(text: str) -> List[Dict]:
    stripped = text.strip()
    return [{"text": stripped}] if stripped else []


# ---- public API ----------------------------------------------------------

def ingest(
    file_path: str,
    corpus_type: CorpusType,
    metadata: Optional[Dict] = None,
) -> int:
    """Ingest a single corpus file into ChromaDB. Returns the number of chunks added."""
    metadata = dict(metadata or {})
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding="utf-8")

    if corpus_type == "novel":
        raw_chunks = _chunk_novel(text)
    elif corpus_type == "essay":
        raw_chunks = _chunk_essay(text)
    elif corpus_type == "review":
        raw_chunks = _chunk_review(text)
    else:
        raise ValueError(
            f"Unknown corpus_type: {corpus_type!r}. Use 'novel', 'essay', or 'review'."
        )

    if not raw_chunks:
        return 0

    documents: List[str] = []
    metadatas: List[Dict] = []
    ids: List[str] = []

    for idx, rc in enumerate(raw_chunks):
        doc_text = rc["text"]
        base_meta: Dict = {
            "source_type": corpus_type,
            "source_file": path.name,
            "chunk_index": idx,
        }
        if corpus_type == "novel":
            if rc.get("context_prev"):
                base_meta["context_prev"] = rc["context_prev"]
            if rc.get("context_next"):
                base_meta["context_next"] = rc["context_next"]

        # Merge user-provided metadata. Chroma only accepts scalar values, so
        # lists are flattened to comma-separated strings.
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple, set)):
                base_meta[k] = ",".join(str(x) for x in v)
            elif isinstance(v, (str, int, float, bool)):
                base_meta[k] = v
            else:
                base_meta[k] = str(v)

        documents.append(doc_text)
        metadatas.append(base_meta)
        ids.append(f"{corpus_type}:{path.stem}:{idx}:{uuid.uuid4().hex[:8]}")

    embeddings = embed_documents(documents)

    collection = _get_collection()
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(ids)
