"""Generation layer — take retrieved chunks, the persona layers, a draft, and
a mode, and produce the response via Claude.

Three modes:

  * ``pressure``   — **analytical**, not voiced. Produces a three-segment
                      diagnostic: 她写过的 / 张力所在 / 压力方向. Useful during
                      revision when you want to see where her voice would push,
                      with the pressure points named and grounded in retrieval.
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
        "【模式：施压 pressure（分析，不是代笔）】\n"
        "你的任务不是替她说话，而是以上述人格文件为参照，冷静地诊断这段草稿。"
        "严格按下面三段结构输出，段名就用中文标题，不要改写：\n\n"
        "## 她写过的\n"
        "从检索到的片段中挑出 2 到 4 处和这段草稿直接相关的材料。每一处给一两行"
        "要点式概括，并在末尾用括号注明来源与置信度——例如：（来自《魔山》评论 "
        "essay / affirm）、（来自对话 paraphrase / linked_via:8303）。"
        "对于 inference 级别的对话片段，明确标注「根据她一贯立场推断」。"
        "检索结果里若有 `_retrieval_reason=linked_via:<stem>`，视为该 stem 出现时"
        "主动带出的语境证据，不要当作直接命中。\n\n"
        "## 张力所在\n"
        "具体指出草稿里她会注意到的地方。不是泛泛的「可以更深入」——而是：\n"
        "  - 草稿的哪一句 / 哪个词，踩到了人格文件里她的带电词汇或禁区；\n"
        "  - 论证依赖了哪个被当作理所当然、但她会视为可疑的预设；\n"
        "  - 有没有把个人意志叙述成了结构性问题的替代、把技术进步叙述成线性必然、"
        "或者把复杂历史归结为几个关键人物——这类她本能抵触的推理动作；\n"
        "  - interior 层里提到的情境差异或触发点，草稿有没有不自觉地触碰。\n"
        "每条都要贴到草稿原文的具体位置（引一个短语或一个短句）。\n\n"
        "## 压力方向\n"
        "写出她会怎样施压——语气仍是分析性的，但内容要像她的反驳：从部分认同"
        "开始，然后撬动预设，以「这个问题比你说的更难」类的开放张力收束。"
        "这一段不超过 4 句，不要铺陈，不要给整洁结论。\n\n"
        "纪律：三段都只写分析语言，不要进入第一人称代笔；不要软化、也不要"
        "假装比她更尖锐；arc.md 指示了当下阶段的声音状态时，以 arc 为准。"
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
