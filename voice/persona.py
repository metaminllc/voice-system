"""Load the persona file. This file is manually maintained and NOT vectorized —
it's the tonal keynote, not a retrieval target."""
from __future__ import annotations

from voice.config import PERSONA_PATH

_PLACEHOLDER = (
    "(persona/base.md is empty or missing. Fill it in before expecting strong rebuttals.)"
)


def load_persona() -> str:
    """Return the raw contents of persona/base.md.

    Returns a placeholder string if the file is missing so generation still works
    instead of crashing on the first run.
    """
    if not PERSONA_PATH.exists():
        return _PLACEHOLDER
    text = PERSONA_PATH.read_text(encoding="utf-8").strip()
    return text or _PLACEHOLDER
