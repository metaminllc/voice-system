"""Generation layer — take retrieved chunks, the persona, a draft, and a mode,
and produce her rebuttal via Claude."""
from __future__ import annotations

from typing import List, Literal

import anthropic

from voice.config import ANTHROPIC_API_KEY, GENERATION_MODEL
from voice.persona import load_persona
from voice.retrieve import RetrievedChunk

Mode = Literal["pressure", "dialogue", "annotation"]

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


MODE_INSTRUCTIONS: dict[str, str] = {
    "pressure": (
        "【模式：施压 pressure】\n"
        "对下面这段草稿发动一次她最典型的反驳。严格按人格文件里的反驳节奏——"
        "通常从部分认同起步，然后在论证依赖的共同前提里找到一个被当作理所当然、"
        "实际上可疑的预设，把它撬开。目标不是击倒，是使对方的论证变得更复杂；"
        "经常以「这个问题比你说的更难」或一个悬而未决的张力收尾。\n"
        "每一个反驳点请在句末用括号注明来源，例如：（来自《XX》小说 chunk "
        "/ 来自你记录的对话 paraphrase）。"
    ),
    "dialogue": (
        "【模式：对话 dialogue】\n"
        "模拟一段短对话。我先说一句（就是下面的草稿摘要），她回一句；"
        "我再追问，她再回——循环 3 到 4 轮。她每一回合的节奏、温度、距离感、"
        "反驳起手（部分认同 → 撬动预设 → 不给整洁结论）都要吻合人格文件。"
        "对话收尾时，另起一段列出她在这轮对话里呼应的语料来源。"
    ),
    "annotation": (
        "【模式：逐句批注 annotation】\n"
        "把草稿拆成一句一行，每一句下面给出她的批注。批注要具体——"
        "指出她在这句里承认了什么、同时质疑什么；哪个词在她这里是带电的、"
        "被你误用；哪个推理动作（例如把历史归结为个别人物意志、把技术进步叙述"
        "为线性必然）踩到了她的禁区。每条批注末尾用括号注明来源。"
    ),
}


def _format_retrieved(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "（未检索到相关语料——仅依据人格文件回应）"
    blocks: List[str] = []
    for i, c in enumerate(chunks, 1):
        header_parts: List[str] = [f"[片段{i}]", f"来源类型：{c.source_type}"]
        if c.source_file:
            header_parts.append(f"文件：{c.source_file}")
        if c.confidence:
            header_parts.append(f"置信度：{c.confidence}")
        # Surface useful optional metadata if present.
        for key in ("topic_tags", "sentiment", "date", "book", "judgment"):
            val = c.metadata.get(key)
            if val:
                header_parts.append(f"{key}：{val}")
        # Context window for novel chunks.
        ctx_parts: List[str] = []
        if c.metadata.get("context_prev"):
            ctx_parts.append(f"（前文）{c.metadata['context_prev']}")
        if c.metadata.get("context_next"):
            ctx_parts.append(f"（后文）{c.metadata['context_next']}")
        header = " | ".join(header_parts)
        body = c.text
        if ctx_parts:
            body = body + "\n" + " ".join(ctx_parts)
        blocks.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(blocks)


def _build_system_prompt(persona: str, corpus_section: str, mode_instruction: str) -> str:
    return (
        "你现在要扮演下面这个写作者本人，用她的声音对一段草稿进行反驳。\n"
        "她不是你的顾问，她是站在你对面的读者。\n\n"
        "【人格文件（她的立场、推理惯式、声音特征——以此为准）】\n"
        f"{persona}\n\n"
        "【从她的语料库中检索到的相关片段（仅供你对齐立场与措辞，不要逐句复述）】\n"
        f"{corpus_section}\n\n"
        f"{mode_instruction}\n\n"
        "纪律：\n"
        "1. 声音、温度、反驳节奏以人格文件为准——不要被礼貌带偏，也不要假装比她"
        "更尖锐。她的锋利来自撬动预设和不给整洁结论，不来自气势。\n"
        "2. 引用她的原话或观点时，在括号里注明出处（例如：来自她的小说"
        "《某某》第一章 / 来自你记录的对话，paraphrase 级别）。对于 inference "
        "级别的对话碎片，要明确标注「根据她的一贯立场推断」。\n"
        "3. 只用中文输出（人格文件里自然嵌入的英文概念词可以保留）。\n"
    )


def generate_rebuttal(
    draft: str,
    retrieved: List[RetrievedChunk],
    mode: Mode = "pressure",
) -> str:
    """Produce her rebuttal to ``draft`` in the requested mode."""
    if mode not in MODE_INSTRUCTIONS:
        raise ValueError(
            f"Unknown mode: {mode!r}. Use one of: {list(MODE_INSTRUCTIONS)}"
        )

    persona = load_persona()
    corpus_section = _format_retrieved(retrieved)
    system_prompt = _build_system_prompt(persona, corpus_section, MODE_INSTRUCTIONS[mode])

    user_message = f"【我的草稿】\n{draft.strip()}"

    client = _get_client()
    response = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    parts: List[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()
