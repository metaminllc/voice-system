"""Load the persona layers — handwritten, never vectorized. They live as the
tonal keynote in every system prompt.

Three layers:

  * ``base.md``     — the declarative persona sketch: stances, voice features,
                       rebuttal rhythm, charged vocabulary. Stable.
  * ``interior.md`` — the layer *under* the declarative voice: tensions she
                       wouldn't state flatly, situational differences, what
                       actually triggers her. Nuance.
  * ``arc.md``      — evolution tracking. How she would sound now vs. five
                       years ago; per-chapter or per-period state you want the
                       generator to respect.

Missing files degrade gracefully: an absent layer simply contributes nothing.
The assembled string ships into the system prompt as three labeled sections so
the model can tell them apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from voice.config import (
    PERSONA_ARC_PATH,
    PERSONA_INTERIOR_PATH,
    PERSONA_PATH,
)

_BASE_PLACEHOLDER = (
    "(persona/base.md is empty or missing. Fill it in before expecting strong rebuttals.)"
)


@dataclass(frozen=True)
class PersonaLayers:
    """Struct holding the three handwritten layers.

    ``base`` is always a non-empty string (placeholder if the file is missing
    or blank). ``interior`` and ``arc`` are ``None`` when absent — callers can
    decide whether to render a section header for them.
    """
    base: str
    interior: Optional[str]
    arc: Optional[str]


def _read_if_exists(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def load_persona_layers() -> PersonaLayers:
    """Read all three layers from disk. Missing/empty layers come back as ``None``
    (base is coerced to a placeholder so generation never crashes)."""
    base = _read_if_exists(PERSONA_PATH) or _BASE_PLACEHOLDER
    interior = _read_if_exists(PERSONA_INTERIOR_PATH)
    arc = _read_if_exists(PERSONA_ARC_PATH)
    return PersonaLayers(base=base, interior=interior, arc=arc)


def load_persona() -> str:
    """Return the three layers concatenated with labeled section headers,
    ready to drop into a system prompt. Missing layers are simply omitted.

    Kept as a plain string (not structured) so callers that just want "the
    persona text" don't have to know about the layer split. Callers that want
    the split — e.g. to inject only the arc layer differently — should use
    :func:`load_persona_layers` instead.
    """
    layers = load_persona_layers()

    sections = [
        "===== 人格 · 基底（base.md — 立场、声音、反驳节奏）=====\n" + layers.base
    ]
    if layers.interior:
        sections.append(
            "===== 人格 · 内里（interior.md — 表层声音之下的张力、情境差异、触发点）=====\n"
            + layers.interior
        )
    if layers.arc:
        sections.append(
            "===== 人格 · 演化（arc.md — 当下阶段 / 章节对应的声音状态）=====\n"
            + layers.arc
        )
    return "\n\n".join(sections)
