"""Configuration: paths, model selection, chunking constants."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---- Paths ---------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CORPUS_DIR: Path = PROJECT_ROOT / "corpus"
DATA_DIR: Path = PROJECT_ROOT / "data"

CHROMA_PATH: Path = Path(
    os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma"))
).expanduser().resolve()

PERSONA_PATH: Path = Path(
    os.getenv("PERSONA_PATH", str(PROJECT_ROOT / "persona" / "base.md"))
).expanduser().resolve()

# Additional persona layers — handwritten, always live as system-prompt context,
# never vectorized.
PERSONA_INTERIOR_PATH: Path = Path(
    os.getenv("PERSONA_INTERIOR_PATH", str(PROJECT_ROOT / "persona" / "interior.md"))
).expanduser().resolve()
PERSONA_ARC_PATH: Path = Path(
    os.getenv("PERSONA_ARC_PATH", str(PROJECT_ROOT / "persona" / "arc.md"))
).expanduser().resolve()

DIALOGUES_JSONL: Path = DATA_DIR / "dialogues.jsonl"

# ---- API keys ------------------------------------------------------------

ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
VOYAGE_API_KEY: str | None = os.getenv("VOYAGE_API_KEY")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

# ---- Models --------------------------------------------------------------

EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "voyage").lower()
VOYAGE_MODEL: str = os.getenv("VOYAGE_MODEL", "voyage-3")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "claude-sonnet-4-20250514")

# ---- Chunking ------------------------------------------------------------

# Novel chunks try to land between [NOVEL_MIN_CHARS, NOVEL_MAX_CHARS] Chinese characters.
NOVEL_MIN_CHARS: int = 200
NOVEL_MAX_CHARS: int = 600

# ---- Collections ---------------------------------------------------------

CORPUS_COLLECTION: str = "corpus"
DIALOGUES_COLLECTION: str = "dialogues"


def ensure_dirs() -> None:
    """Create the directories the system needs. Safe to call repeatedly."""
    for d in (
        DATA_DIR,
        CHROMA_PATH,
        CORPUS_DIR,
        CORPUS_DIR / "novels",
        CORPUS_DIR / "essays",
        CORPUS_DIR / "reviews",
        CORPUS_DIR / "diaries",
        CORPUS_DIR / "dialogues",
        PERSONA_PATH.parent,
    ):
        d.mkdir(parents=True, exist_ok=True)


# Ensure directories exist on import — cheap, idempotent.
ensure_dirs()
