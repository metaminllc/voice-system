
# Voice Distillation System — 完整设计文档

> 唯一真相来源。所有改动直接更新此文件，用 git 记录历史。

---

## 概述

构建一个本地运行的 RAG 系统，用于"蒸馏"一个写作者的声音，供写作时召唤她的立场对草稿进行压力分析。

**工作方式：** 用户写完一段草稿，系统检索语料库中与该论点最相关或最冲突的片段，生成四段式分析（原文 + 张力 + 压力方向 + subtext），由用户自己用自己的语言把她的声音写进文章。系统不生成她的声音，只提供方向和支撑材料。

---

## 技术栈

- Python 3.11+
- ChromaDB（本地向量库，持久化）
- Voyage AI `voyage-3`（embedding，中文支持更好）或 OpenAI `text-embedding-3-small`
- Claude API `claude-sonnet-4-20250514`（生成）
- FastAPI（本地 web 接口）
- Rich（CLI 美化）

---

## 项目结构

```
voice-system/
├── SPEC.md                    # 本文件
├── HANDOFF.md                 # 当前实现进度（每次 session 开始前更新）
├── pyproject.toml
├── .env.example
│
├── corpus/                    # 原始文本存放
│   ├── novels/
│   ├── essays/
│   ├── reviews/
│   └── diaries/
│
├── data/
│   ├── chroma/                # ChromaDB 持久化目录
│   └── dialogues.jsonl        # 对话碎片的人类可读备份
│
├── persona/
│   ├── base.md                # 从文本归纳的立场、推理结构、声音特征
│   ├── interior.md            # 初始人物建模：心理底色（ground truth，当前留空）
│   └── arc.md                 # 叙事曲线：自由文本日志（当前留空）
│
├── voice/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   ├── dialogue.py
│   ├── retrieve.py
│   ├── generate.py
│   └── persona.py
│
├── cli.py
└── app.py
```

---

## Persona 文件系统

三个文件，职责分离，在每次调用 `generate.py` 时实时读取，修改任何一个后下次调用自动生效。

### persona/base.md
从语料文本中归纳的内容：核心立场、推理惯式、带电词汇、声音特征（句法节奏、温度、距离感）。**已有初稿，见项目根目录。**

### persona/interior.md
初始人物建模。她在这本书开始之前是谁——心理底色、表层行为与内在结构的张力、情感触发结构。由用户作为 ground truth 手动写入，不入向量库。**当前留空，待填写。**

模板：
```markdown
# 心理结构（ground truth）

## 表层行为与内在结构的张力
-

## 她在不同情境下的声音差异
- 写作时 vs 私聊时：
- 被认同时 vs 被质疑时：

## 情感触发结构
- 什么会让她真正生气（而不是表演的生气）：
- 什么会让她沉默而不是反驳：
- 什么是她表面退让但内心没有移动的情况：
```

### persona/arc.md
叙事曲线。自由文本日志，不是结构化数据。记录每章写作之后你和她之间发生了什么——包括但不限于：她暴露的内层、两个声音的互渗痕迹、关键论题在 rebuttal 过程中的演进、对 interior.md 假设的验证或推翻。

**系统读取方式：** 整个文件作为自由文本直接进入 system prompt，不做解析。

**当前留空，随写作推进手动更新。**

### system prompt 构成顺序
```python
system_prompt = (
    read("persona/base.md")
    + read("persona/interior.md")
    + read("persona/arc.md")
)
```

---

## 核心模块规格

### ingest.py — 语料入库

```python
ingest(
    file_path: str,
    corpus_type: Literal["novel", "essay", "review", "diary"],
    metadata: dict = {}
)
```

**分块策略：**
- `novel`：按段落分块，200-600 字，保留前后各一句作为上下文窗口
- `essay`：按论点段落分块，识别换行+缩进作为段落边界，不强制截断
- `review`：整篇作为单个 chunk
- `diary`：按自然段落分块

**Metadata schema：**
```python
{
    "source_type": "novel" | "essay" | "review" | "diary",
    "source_file": str,                          # 文件名，用于 linked_sources 关联
    "date": str | None,                          # "YYYY-MM-DD" 或 "YYYY"
    "stance": "affirm" | "resist" | "ambivalent" | "expository" | None,
    "stance_note": str | None                    # 自由文本，说明立场指向什么
}
```

**`stance` 字段说明：**
防止把她"介绍"的观点误认为她"认同"的观点，在哲学书评中尤其重要。`expository` 表示"大量篇幅在介绍他人观点，生成时需二次判断哪些是她自己的立场"。

**CLI：**
```bash
python cli.py ingest corpus/essays/essay1.md \
  --type essay \
  --stance ambivalent \
  --stance-note "对卡西尔更同情，但认为海德格尔提出了真正的问题"
```

---

### dialogue.py — 对话碎片模块

存储私聊中产生的观点碎片，支持增量更新。

**Schema：**
```python
{
    "quote": str,                                # 原话或概括
    "context": str | None,                       # 这是在讨论什么的时候
    "your_note": str | None,                     # 用户对这段话的诠释
    "confidence": "exact" | "paraphrase" | "inference",
    "linked_sources": list[str] | None           # 关联的语料文件名
}
```

**Confidence 权重：**
- `exact`：原话，生成时可直接援引
- `paraphrase`：转述，生成时标注"她曾表达过类似立场"
- `inference`：推断，生成时标注"根据她的一贯立场推断"

**存储：** 入 ChromaDB 的独立 collection `dialogues`，同时写入 `data/dialogues.jsonl`（人类可读备份）。

**CLI：**
```bash
python cli.py add "理性叙事本身就是一种暴力" \
  --context "讨论启蒙理性的时候" \
  --note "她认为任何系统性叙事都有排斥性" \
  --confidence exact \
  --link magic_mountain_divide_essay_20121111

python cli.py list-dialogues
```

**Web 接口：**
```
POST /add-dialogue
{
  "quote": str,
  "context": str | null,
  "your_note": str | null,
  "confidence": "exact" | "paraphrase" | "inference",
  "linked_sources": list[str] | null
}
```

---

### retrieve.py — 检索层

```python
retrieve(
    query: str,
    n_results: int = 5,
    collections: list = ["corpus", "dialogues"],
    filter_by: dict = {}
) -> list[RetrievedChunk]
```

```python
@dataclass
class RetrievedChunk:
    text: str
    source_type: str           # novel / essay / review / diary / dialogue
    source_file: str
    confidence: str | None     # 仅 dialogue 有
    stance: str | None         # 仅 corpus 有
    distance: float
    metadata: dict
```

**双向检索逻辑：**
检索到某篇语料时，自动拉取所有 `linked_sources` 包含该文件名的对话碎片，合并进结果集，一并返回。没有 `linked_sources` 的碎片纯靠向量检索，行为不变。

---

### generate.py — 生成层

**仅实现 `pressure` 模式，** `dialogue` 和 `annotation` 模式暂不实现。

```python
generate_rebuttal(
    draft: str,
    retrieved: list[RetrievedChunk]
) -> str
```

**输出格式（四段式）：**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
她写过的：
[检索到的原文片段]
— 来源文件（置信度：exact / paraphrase / inference）

张力所在：
[你的论述和她的文本在哪里产生冲突，一到两句话]

压力方向：
[她的反驳可能从哪个角度进入，一句话]

Subtext：
[这个反驳背后，她可能在保护或回避什么]
— 来源：interior.md / arc.md
  置信度：高 / 低（低 = 新发现的维度，待验证）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Subtext 生成逻辑：**
- 读取 interior.md 当前版本
- 读取 arc.md，只调用其中已标注为"已暴露"的 interior 层面
- 如果某个 subtext 来自 arc.md 中尚未暴露的部分，置信度标注为"低，待验证"
- Subtext 不进入她的声音，只作为用户写她的声音时的参考框架

---

### persona.py — 读取人格文件

同时读取三个文件，按顺序拼接为 system prompt：

```python
def load_persona() -> str:
    return (
        read_file("persona/base.md")
        + "\n\n"
        + read_file("persona/interior.md")
        + "\n\n"
        + read_file("persona/arc.md")
    )
```

三个文件在每次调用时实时读取，无需重启服务。

---

## CLI 命令总览

```bash
# 入库
python cli.py ingest corpus/essays/essay1.md --type essay
python cli.py ingest corpus/novels/novel1.txt --type novel
python cli.py ingest corpus/reviews/review1.md --type review --stance resist --stance-note "反对书中的技术决定论"
python cli.py ingest corpus/diaries/diary1.md --type diary

# 添加对话碎片
python cli.py add "原话或概括" --context "语境" --confidence exact --link source_filename

# 查看对话碎片
python cli.py list-dialogues

# 生成压力分析
python cli.py rebut "你的草稿段落"

# 启动 web 界面
python app.py
```

---

## Web 界面（app.py）

极简单页，三个区域：
1. **左栏**：输入草稿段落
2. **右栏**：四段式压力分析输出
3. **底部**：快速添加对话碎片的表单（手机也能用）

---

## 环境变量（.env）

```
ANTHROPIC_API_KEY=
EMBEDDING_PROVIDER=voyage       # or openai
VOYAGE_API_KEY=                 # if using voyage
OPENAI_API_KEY=                 # if using openai
CHROMA_PATH=./data/chroma
PERSONA_BASE_PATH=./persona/base.md
PERSONA_INTERIOR_PATH=./persona/interior.md
PERSONA_ARC_PATH=./persona/arc.md
```

---

## 注意事项

- 所有中文文本确保 UTF-8 编码处理
- ChromaDB 使用持久化模式，不用每次重建
- `dialogues.jsonl` 在每次 `add` 后同步写入，作为人类可读备份
- 生成输出里标注每个论点的来源和置信度
- `stance` 字段为 `expository` 的文本，生成 subtext 时需提示 LLM 二次判断哪些是她自己的立场
- Persona 三个文件变动后无需重新入库，下次调用自动生效

---

## 实现优先级

1. `ingest.py`（四种语料类型 + stance 字段）
2. `dialogue.py`（含 linked_sources + JSONL 备份）
3. `retrieve.py`（双向检索逻辑）
4. `persona.py`（读取三个文件，拼接 system prompt）
5. `generate.py`（四段式输出，含 subtext 和置信度）
6. `cli.py`（完整命令覆盖）
7. `app.py`（FastAPI + 极简前端）
8. 创建 `persona/interior.md` 和 `persona/arc.md` 空白模板
