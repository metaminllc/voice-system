"""FastAPI web entry point. Run with:

    python app.py
    # or
    uvicorn app:app --reload --port 8765
"""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from voice.dialogue import add_dialogue, list_dialogues
from voice.generate import generate_rebuttal
from voice.retrieve import retrieve

app = FastAPI(title="Voice Distillation", version="0.1.0")


# ---- schemas -------------------------------------------------------------

class DialogueIn(BaseModel):
    quote: str = Field(..., description="The quote or paraphrase.")
    context: Optional[str] = None
    your_note: Optional[str] = None
    confidence: Literal["exact", "paraphrase", "inference"] = "paraphrase"
    linked_sources: Optional[list[str]] = Field(
        default=None,
        description="Corpus file stems this quote is semantically bound to.",
    )


class RebutIn(BaseModel):
    draft: str = Field(..., description="Your draft passage.")
    mode: Literal["pressure", "dialogue", "annotation"] = "pressure"
    n_results: int = 6


class SourceOut(BaseModel):
    source_type: str
    source_file: str
    confidence: Optional[str] = None
    distance: float
    text_preview: str


class RebutOut(BaseModel):
    rebuttal: str
    sources: list[SourceOut]


# ---- endpoints -----------------------------------------------------------

@app.post("/add-dialogue")
def api_add_dialogue(payload: DialogueIn):
    try:
        return add_dialogue(
            quote=payload.quote,
            context=payload.context,
            your_note=payload.your_note,
            confidence=payload.confidence,
            linked_sources=payload.linked_sources,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/dialogues")
def api_list_dialogues():
    return list_dialogues()


@app.post("/rebut", response_model=RebutOut)
def api_rebut(payload: RebutIn):
    chunks = retrieve(payload.draft, n_results=payload.n_results)
    try:
        rebuttal = generate_rebuttal(payload.draft, chunks, mode=payload.mode)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RebutOut(
        rebuttal=rebuttal,
        sources=[
            SourceOut(
                source_type=c.source_type,
                source_file=c.source_file,
                confidence=c.confidence,
                distance=c.distance,
                text_preview=c.text[:160],
            )
            for c in chunks
        ],
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


# ---- frontend ------------------------------------------------------------

INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Voice Distillation</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, "Helvetica Neue", "PingFang SC",
                 "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
    margin: 0; padding: 24px; background: #faf7f2; color: #222;
  }
  h1 { font-size: 16px; letter-spacing: 2px; margin: 0 0 18px; color: #2d2a26; }
  .wrap { max-width: 1240px; margin: 0 auto; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .panel {
    background: #fff; border: 1px solid #e6dfd2; border-radius: 6px;
    padding: 16px; box-shadow: 0 1px 2px rgba(60,40,20,0.04);
  }
  .panel h2 {
    font-size: 12px; margin: 0 0 10px; color: #7a6a4a;
    letter-spacing: 2px; text-transform: uppercase;
  }
  textarea {
    width: 100%; min-height: 320px; border: 1px solid #e0d8c6; border-radius: 4px;
    padding: 12px; font-family: inherit; font-size: 14px; line-height: 1.7;
    background: #fdfcf9; resize: vertical; color: #222;
  }
  .controls { display: flex; gap: 10px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
  select, input[type=text] {
    padding: 7px 10px; border: 1px solid #e0d8c6; border-radius: 4px;
    background: #fff; font-family: inherit; font-size: 13px;
  }
  button {
    padding: 8px 16px; background: #2d2a26; color: #faf7f2; border: none;
    border-radius: 4px; cursor: pointer; letter-spacing: 1px; font-size: 13px;
  }
  button:hover { background: #55483a; }
  button.secondary { background: #efe7d6; color: #2d2a26; }
  .rebuttal {
    white-space: pre-wrap; line-height: 1.85; font-size: 15px; min-height: 320px;
    color: #1a1a1a;
  }
  .rebuttal.placeholder { color: #bba; font-style: italic; }
  .sources {
    font-size: 12px; color: #6a5d48; margin-top: 14px;
    border-top: 1px dashed #dcd2bb; padding-top: 10px; line-height: 1.6;
  }
  .sources strong { color: #7a6a4a; letter-spacing: 1px; }
  .sources div { margin: 3px 0; }
  .dialogue-form {
    display: grid; grid-template-columns: 2fr 1fr 1fr 140px 110px;
    gap: 8px; align-items: start;
  }
  @media (max-width: 900px) {
    .row { grid-template-columns: 1fr; }
    .dialogue-form { grid-template-columns: 1fr; }
  }
  .status { color: #7a6a4a; font-size: 13px; margin-left: 8px; }
  .status.err { color: #a63c2b; }
</style>
</head>
<body>
<div class="wrap">
  <h1>VOICE DISTILLATION</h1>

  <div class="row">
    <div class="panel">
      <h2>草稿</h2>
      <textarea id="draft" placeholder="把你要被她反驳的段落贴进来..."></textarea>
      <div class="controls">
        <select id="mode">
          <option value="pressure">pressure — 施压</option>
          <option value="dialogue">dialogue — 对话</option>
          <option value="annotation">annotation — 逐句批注</option>
        </select>
        <button onclick="doRebut()">召唤她</button>
        <span class="status" id="rebut-status"></span>
      </div>
    </div>
    <div class="panel">
      <h2>她的反驳</h2>
      <div class="rebuttal placeholder" id="rebuttal">（此处显示她的回应）</div>
      <div class="sources" id="sources"></div>
    </div>
  </div>

  <div class="panel">
    <h2>添加对话碎片</h2>
    <div class="dialogue-form">
      <input type="text" id="d-quote" placeholder="原话或转述" />
      <input type="text" id="d-context" placeholder="语境（可选）" />
      <input type="text" id="d-note" placeholder="你的笔记（可选）" />
      <select id="d-conf">
        <option value="exact">exact — 原话</option>
        <option value="paraphrase" selected>paraphrase — 转述</option>
        <option value="inference">inference — 推断</option>
      </select>
      <button class="secondary" onclick="addDialogue()">保存</button>
    </div>
    <div class="status" id="dialogue-status"></div>
  </div>
</div>

<script>
async function doRebut() {
  const draft = document.getElementById('draft').value.trim();
  const mode = document.getElementById('mode').value;
  if (!draft) { alert('先写点什么吧'); return; }
  const status = document.getElementById('rebut-status');
  const rebuttalEl = document.getElementById('rebuttal');
  status.className = 'status';
  status.textContent = '检索 + 生成中...';
  try {
    const r = await fetch('/rebut', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({draft, mode, n_results: 6}),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({detail: r.statusText}));
      throw new Error(err.detail || r.statusText);
    }
    const data = await r.json();
    rebuttalEl.classList.remove('placeholder');
    rebuttalEl.textContent = data.rebuttal;
    const srcDiv = document.getElementById('sources');
    srcDiv.innerHTML = '';
    if (data.sources && data.sources.length) {
      const title = document.createElement('strong');
      title.textContent = '来源：';
      srcDiv.appendChild(title);
      data.sources.forEach(s => {
        const d = document.createElement('div');
        const conf = s.confidence ? ` (${s.confidence})` : '';
        d.textContent = `• ${s.source_type} / ${s.source_file}${conf} — ${s.text_preview}`;
        srcDiv.appendChild(d);
      });
    }
    status.textContent = '完成';
  } catch (e) {
    status.className = 'status err';
    status.textContent = '错误：' + e.message;
  }
}

async function addDialogue() {
  const quote = document.getElementById('d-quote').value.trim();
  if (!quote) { alert('quote 必填'); return; }
  const context = document.getElementById('d-context').value.trim() || null;
  const your_note = document.getElementById('d-note').value.trim() || null;
  const confidence = document.getElementById('d-conf').value;
  const status = document.getElementById('dialogue-status');
  status.className = 'status';
  status.textContent = '保存中...';
  try {
    const r = await fetch('/add-dialogue', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({quote, context, your_note, confidence}),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({detail: r.statusText}));
      throw new Error(err.detail || r.statusText);
    }
    const data = await r.json();
    status.textContent = '已保存：' + data.id;
    document.getElementById('d-quote').value = '';
    document.getElementById('d-context').value = '';
    document.getElementById('d-note').value = '';
  } catch (e) {
    status.className = 'status err';
    status.textContent = '错误：' + e.message;
  }
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8765, reload=False)
