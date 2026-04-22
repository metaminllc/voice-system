"""Generation layer — take retrieved chunks, the persona layers, a draft, and
a mode, and produce the response via Claude.

Three modes:

  * ``pressure``   — **analytical**, not voiced. Produces a four-segment
                      diagnostic wrapped in ━ dividers:
                        1. 她写过的   — retrieved corpus evidence
                        2. 张力所在   — where draft and her texts collide
                        3. 压力方向   — the angle she'd push from
                        4. Subtext    — what the rebuttal is protecting or
                                         avoiding, cited against interior.md /
                                         arc.md with a 高/低 confidence marker
                      Subtext is reference material for the user — not her
                      voice. First-person ventriloquism is explicitly banned.
  * ``dialogue``   — voiced short exchange. *Not yet tuned — experimental.*
  * ``annotation`` — voiced per-sentence margin notes. *Not yet tuned —
                      experimental.*
"""
from __future__ import annotations

from typing import List, Literal

import anthropic

from voice.config import ANTHROPIC_API_KEY, GENERATION_MODEL
from voice.persona import load_persona
from voice.retrieve import RetrievedChunk

Mode = Literal["pressure", "dialogue", "annotation"]

_TUNED_MODES: set[str] = {"pressure"}
_EXPERIMENTAL_MODES: set[str] = {"dialogue", "annotation"}

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
        "【模式：施压 pressure（四段诊断，不是代笔）】\n"
        "你的任务不是替她说话，而是以三层人格文件（base / interior / arc）"
        "为参照，冷静地诊断这段草稿。严格按下面的四段结构输出，段名和分隔线"
        "都照抄，不要改写：\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "她写过的：\n"
        "从检索到的片段中挑出 2 到 4 处和这段草稿直接相关的原文。每一处先引一两行"
        "原文（不要概括成抽象），再另起一行用破折号标注来源，格式：\n"
        "  — 来源文件（置信度：exact / paraphrase / inference）\n"
        "对于 inference 级别的对话片段，明确标注「根据她一贯立场推断」。"
        "带 `_retrieval_reason=linked_via:<stem>` 的片段属于跟随出场的语境证据，"
        "不是直接命中——写出来时要在来源行注明这一点。\n\n"
        "张力所在：\n"
        "一到两句话，讲清楚草稿和她的文本在哪里冲突。必须贴具体位置——"
        "引草稿里的一个短语或短句，再点出它踩到了她的哪个带电词汇、哪个禁区、"
        "或者哪个可疑预设。不要铺陈。\n\n"
        "压力方向：\n"
        "一句话，写她的反驳可能从哪个角度进入。从部分认同起步，撬动一个"
        "可疑预设，不给整洁结论。只写一句，不要展开。\n\n"
        "Subtext：\n"
        "一到两句分析语言，写出这个反驳背后她可能在保护或回避什么——"
        "是 interior.md 里已经写下的某种张力？还是 arc.md 当下阶段里暴露出的"
        "东西？还是你从语料整体推断出的一个新维度？\n"
        "Subtext 是给用户（我）看的参考框架，不是她本人的话——**绝对不要**用"
        "第一人称或引用她的口吻来写。写完另起一行用破折号标注来源和置信度：\n"
        "  — 来源：interior.md / arc.md / 语料推断（选一个或多个）\n"
        "    置信度：高 / 低（高 = 已在 interior.md 或 arc.md 中明示；"
        "低 = 新发现的维度，待验证）\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "通用纪律：\n"
        "- 四段都只写分析语言，不要进入第一人称代笔；\n"
        "- arc.md 指示了当下阶段的声音状态时，以 arc 为准（它覆盖 base）；\n"
        "- 不要软化，也不要假装比她更尖锐。"
    ),
    "dialogue": (
        "【模式：对话 dialogue ·（实验，尚未调校）】\n"
        "模拟一段短对话。我先说一句（就是下面的草稿摘要），她回一句；"
        "我再追问，她再回——循环 3 到 4 轮。她每一回合的节奏、温度、距离感、"
        "反驳起手（部分认同 → 撬动预设 → 不给整洁结论）都要吻合人格文件。"
        "对话收尾时，另起一段列出她在这轮对话里呼应的语料来源。\n\n"
        "注意：这个模式还没有针对三层人格（base/interior/arc）调校，"
        "输出仅供参考。"
    ),
    "annotation": (
        "【模式：逐句批注 annotation ·（实验，尚未调校）】\n"
        "把草稿拆成一句一行，每一句下面给出她的批注。批注要具体——"
        "指出她在这句里承认了什么、同时质疑什么；哪个词在她这里是带电的、"
        "被你误用；哪个推理动作踩到了她的禁区。每条批注末尾用括号注明来源。\n\n"
        "注意：这个模式还没有针对三层人格（base/interior/arc）调校，"
        "输出仅供参考。"
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
        for key in (
            "topic_tags",
            "stance",
            "stance_note",
            "sentiment",
            "date",
            "book",
            "judgment",
            "_retrieval_reason",
        ):
            val = c.metadata.get(key)
            if val:
                header_parts.append(f"{key}：{val}")
        # Context window for semantic chunks.
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


def _build_system_prompt(
    persona: str,
    corpus_section: str,
    mode_instruction: str,
    mode: str,
) -> str:
    if mode in _TUNED_MODES:
        framing = (
            "你现在要以下面这个写作者的人格文件为参照，对一段草稿做诊断性分析——"
            "不是代笔、不是替她说话，而是把她会看到的压力点讲清楚。"
        )
    else:
        framing = (
            "你现在要扮演下面这个写作者本人，用她的声音对一段草稿进行反驳。\n"
            "她不是你的顾问，她是站在你对面的读者。"
        )

    return (
        f"{framing}\n\n"
        "【人格文件（三层：base 基底 / interior 内里 / arc 当下演化——"
        "interior 和 arc 覆盖 base）】\n"
        f"{persona}\n\n"
        "【从她的语料库中检索到的相关片段】\n"
        f"{corpus_section}\n\n"
        f"{mode_instruction}\n\n"
        "通用纪律：\n"
        "1. 声音、温度、反驳节奏以人格文件为准——不要被礼貌带偏，也不要假装"
        "比她更尖锐。她的锋利来自撬动预设和不给整洁结论，不来自气势。\n"
        "2. 引用原话或观点时注明出处：例如（来自她的小说《某某》第一章 / "
        "来自对话 paraphrase）。对 inference 级别的对话碎片，要标注"
        "「根据她的一贯立场推断」。\n"
        "3. 带 `_retrieval_reason=linked_via:<stem>` 的片段是因为对应的语料"
        "被命中才带出，属于背景证据，不要当作直接命中。\n"
        "4. 只用中文输出（人格文件里自然嵌入的英文概念词可以保留）。\n"
    )


def generate_rebuttal(
    draft: str,
    retrieved: List[RetrievedChunk],
    mode: Mode = "pressure",
) -> str:
    """Produce her response to ``draft`` in the requested mode.

    ``pressure`` is the only mode currently tuned against the three-layer
    persona; it returns analytical output, not a first-person rebuttal.
    ``dialogue`` and ``annotation`` are retained but experimental.
    """
    if mode not in MODE_INSTRUCTIONS:
        raise ValueError(
            f"Unknown mode: {mode!r}. Use one of: {list(MODE_INSTRUCTIONS)}"
        )

    persona = load_persona()
    corpus_section = _format_retrieved(retrieved)
    system_prompt = _build_system_prompt(
        persona, corpus_section, MODE_INSTRUCTIONS[mode], mode
    )

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
