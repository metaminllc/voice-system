# Voice Distillation

本地运行的 RAG 系统：**蒸馏一个写作者的声音，供你写作时召唤她对草稿进行反驳。**

三种反驳模式：

- `pressure` — 对你的论点施压，找最脆弱的地方
- `dialogue` — 模拟一段短对话，你一句她一句
- `annotation` — 她逐句批注你的草稿

---

## 安装

需要 Python 3.11+。

```bash
cd voice-system
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

复制并填写环境变量：

```bash
cp .env.example .env
# 然后编辑 .env
```

`.env` 里需要的字段：

```
ANTHROPIC_API_KEY=sk-ant-...
EMBEDDING_PROVIDER=voyage           # 或 openai
VOYAGE_API_KEY=pa-...               # 若 provider=voyage
OPENAI_API_KEY=sk-...               # 若 provider=openai
CHROMA_PATH=./data/chroma
PERSONA_PATH=./persona/base.md
```

---

## 准备工作（只做一次）

### 1. 写人格文件

打开 `persona/base.md`，手动填写她的核心立场、推理惯式、带电词汇、声音特征、禁区。

**这个文件不进向量库**——它是声音的基调，不是检索对象。改动后下一次调用自动生效，不用重入库。

### 2. 放原始语料

按类型放进 `corpus/`：

```
corpus/
├── novels/    # 小说 .txt 或 .md
├── essays/    # 评论文章
└── reviews/   # 书评（持续增长）
```

### 3. 入库

```bash
# 小说——按语义单元分块（200-600 字），保留前后各一句作上下文
python cli.py ingest corpus/novels/novel1.txt --type novel

# 评论——按论点段落分块
python cli.py ingest corpus/essays/essay1.md --type essay \
    --date 2023 --tags 启蒙理性,女性 --sentiment resist

# 书评——整篇作为一个 chunk
python cli.py ingest corpus/reviews/review1.md --type review \
    --book "某本书" --judgment resist
```

---

## 日常使用

### 召唤反驳

```bash
python cli.py rebut "这里是你的草稿段落..." --mode pressure
python cli.py rebut "..." --mode dialogue
python cli.py rebut "..." --mode annotation
```

输出会在末尾标注来源（哪段语料、哪条对话、confidence 级别）。

### 添加对话碎片

随手记下她说过的话（或者你推断她会说的）：

```bash
python cli.py add \
  "理性叙事本身就是一种暴力" \
  --context "讨论启蒙理性的时候" \
  --note "她认为任何系统性叙事都有排斥性" \
  --confidence exact
```

`confidence` 三档：

- `exact` — 原话引用，生成时可直接援引
- `paraphrase` — 转述，生成时标注"她曾表达过类似立场"
- `inference` — 推断，生成时标注"根据她的一贯立场推断"

碎片同时写进 ChromaDB 的 `dialogues` collection 和 `data/dialogues.jsonl`（人类可读备份，方便手动校对）。

### 查看已存的碎片

```bash
python cli.py list-dialogues
```

### 调试：看检索返回了什么

```bash
python cli.py retrieve "你想测试的 query"
```

---

## Web 界面

```bash
python app.py
# 浏览器打开 http://127.0.0.1:8765
```

极简单页，三个区域：

1. 左栏：输入你的草稿段落
2. 右栏：她的反驳（可选三种模式）
3. 底部：快速添加对话碎片的表单

HTTP API：

- `POST /rebut` — body: `{draft, mode, n_results}`
- `POST /add-dialogue` — body: `{quote, context?, your_note?, confidence}`
- `GET /dialogues` — 返回所有碎片

---

## 项目结构

```
voice-system/
├── README.md
├── pyproject.toml
├── .env.example
│
├── corpus/
│   ├── novels/
│   ├── essays/
│   ├── reviews/
│   └── dialogues/     # 占位，暂不使用
│
├── data/
│   ├── chroma/        # ChromaDB 持久化目录（自动生成）
│   └── dialogues.jsonl # 对话碎片备份（自动生成）
│
├── voice/
│   ├── __init__.py
│   ├── config.py       # 路径、模型、常量
│   ├── embeddings.py   # Voyage / OpenAI 适配
│   ├── ingest.py       # 三种分块策略
│   ├── dialogue.py     # 对话碎片存取
│   ├── retrieve.py     # 统一检索两个 collection
│   ├── generate.py     # 三种反驳模式
│   └── persona.py      # 读取人格文件
│
├── persona/
│   └── base.md         # 手写人格文件
│
├── cli.py              # 命令行入口
└── app.py              # FastAPI + 极简前端
```

---

## 一些注意事项

- **编码**：所有文件按 UTF-8 读写，中文语料直接贴进 `.txt` / `.md` 即可。
- **持久化**：ChromaDB 用 `PersistentClient`，入库一次即可，不用每次重建。
- **人格文件**：改完之后下一次生成自动生效，不需要重新入库。
- **来源标注**：生成的反驳中，每个论点应该带括号标注来源（例如"来自她的小说《XX》第一章"、"来自你记录的对话，paraphrase 级别"）。如果模型偶尔忘记，可以在 `voice/generate.py` 的 `MODE_INSTRUCTIONS` 里把这条纪律写得更严。
- **切换 embedding provider**：只改 `.env` 里的 `EMBEDDING_PROVIDER`。注意切换后之前入的库会用不同维度的向量——要么清空 `data/chroma/` 重入，要么保持一致。

---

## 快速验证

装完之后、还没配 API key 之前，可以先跑一下这个不触网的冒烟测试确认结构没坏：

```bash
python -c "
from voice.ingest import _chunk_novel, _chunk_essay, _chunk_review
print('novel chunks:', len(_chunk_novel('这是第一段。有两句话。\\n\\n这是第二段。' * 10)))
print('essay chunks:', len(_chunk_essay('第一段。\\n\\n第二段。\\n\\n第三段。')))
print('review chunks:', len(_chunk_review('整篇书评内容。')))
"
```
