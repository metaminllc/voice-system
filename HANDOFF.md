# HANDOFF — 当前实现进度

> 每次 session 开始前更新此文件。

---

## 状态：基础实现完成，待首次真实运行验证

最后更新：2026-04-22

---

## 已实现

- `voice/config.py` — 路径、模型、常量；支持三层 persona 路径（base/interior/arc）
- `voice/embeddings.py` — Voyage / OpenAI 双 provider 适配
- `voice/ingest.py` — 四种语料类型（novel/essay/review/diary）+ stance 字段 + chunk 策略
- `voice/dialogue.py` — add/list、linked_sources 双向绑定、JSONL 备份、confidence 三档
- `voice/retrieve.py` — 双向检索逻辑、_retrieval_reason 标注、RetrievedChunk dataclass
- `voice/persona.py` — 三层文件实时读取、缺失文件降级处理
- `voice/generate.py` — pressure 模式四段式输出（含 subtext 置信度）；dialogue/annotation 保留但标注为实验性
- `cli.py` — ingest / add / list-dialogues / rebut / retrieve / web 全部命令
- `app.py` — FastAPI + 三端点（/rebut /add-dialogue /dialogues）+ 极简前端

---

## 已知偏差（vs SPEC）

- `diary` 分块走的是和 `novel` 相同的语义分块（含上下文窗口），SPEC 原意是"按自然段落"。当前行为更好，暂不改。
- Web 前端表单没有 `linked_sources` 输入框，只能通过 CLI 绑定。
- `persona/base.example.md` 未创建。
- `persona/interior.md` 和 `persona/arc.md` 空白模板待确认是否存在。

---

## 待办

- [ ] 充值 Anthropic API，完成首次端到端 rebut 测试
- [ ] 填写 `persona/base.md`（核心，没有它 pressure 模式无效）
- [ ] 创建 `persona/interior.md` 和 `persona/arc.md`（空白模板即可，随写作推进填写）
- [ ] 向量库入第一批语料

---

## 环境

- Python 3.11
- Embedding provider：voyage（`.env` 里配置）
- Generation model：claude-sonnet-4-20250514
- ChromaDB 持久化路径：`./data/chroma`
