"""原文检索与读原文。

- 原文检索走 P2 固定 RAG 那条链路（BM25 + 向量 → RRF → rerank），返回 chunk 全文；可选限定卷、章节范围、
  提到某实体的段落。固定流程 RAG 直接调用 run_search（同一实现），Agent 通过 search_text 工具调用。
- 读原文按 chunk_id 读取，可以向前后扩展若干块（不跨章）。
阅读进度与检索范围来自 ToolContext（运行环境注入），模型看不到也改不了。
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from novel_rag.retrieval.base import Hit
from novel_rag.retrieval.expand import _stitch
from novel_rag.retrieval.filters import SearchFilter
from novel_rag.tools.backends import ToolBackends
from novel_rag.tools.base import ToolContext, ToolInputError, ToolNotFound, ToolResult

CHUNK_ID = re.compile(r"^(?P<book>[^:\s]+):(?P<chap>\d{1,4}):(?P<seq>\d{1,4})$")
CHAPTER_KEY = re.compile(r"^(?P<book>[^:\s]+):(?P<chap>\d{1,4})$")


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


def chunk_header(b: ToolBackends, c: dict[str, Any]) -> str:
    return f"【{c['chunk_id']}】{b.where(c['book_id'], c['chapter_idx'])}"


def parse_chapter(key: str) -> tuple[str, int]:
    m = CHAPTER_KEY.match(key.strip()) or CHUNK_ID.match(key.strip())
    if not m:
        raise ToolInputError(f"章节编号格式应为 卷ID:章号（如 {key.split(':')[0] or 'book'}:0012），收到 {key!r}")
    return m.group("book"), int(m.group("chap"))


def chapter_ro(b: ToolBackends, book_id: str, chapter_idx: int) -> int | None:
    c = b.chunks.get(book_id, chapter_idx, 0)
    if c is None:
        c = next((x for x in b.chunks.chunks if x["book_id"] == book_id and x["chapter_idx"] == chapter_idx), None)
    return None if c is None else int(c["reading_order"])


def make_filter(
    b: ToolBackends,
    ctx: ToolContext,
    *,
    volume: str | None = None,
    chapter_from: str | None = None,
    chapter_to: str | None = None,
    entity_id: str | None = None,
) -> tuple[SearchFilter, dict[str, Any]]:
    """把模型给的范围参数和运行环境的阅读进度合成一个过滤条件。返回 (过滤条件, 说明)。"""
    books = ctx.book_ids
    if volume:
        if volume not in books:
            raise ToolInputError(f"没有卷 {volume!r}；可选：{', '.join(books)}")
        books = (volume,)
    flt = SearchFilter(book_ids=books, max_reading_order=ctx.max_reading_order)
    note: dict[str, Any] = {}
    if chapter_from or chapter_to:
        ros = []
        for key in (chapter_from, chapter_to):
            if key is None:
                ros.append(None)
                continue
            bk, ch = parse_chapter(key)
            ro = chapter_ro(b, bk, ch)
            if ro is None:
                raise ToolNotFound(f"没有章节 {key}")
            ros.append(ro)
        lo = ros[0] if ros[0] is not None else 0
        hi = ros[1] if ros[1] is not None else 10**9
        if lo > hi:
            lo, hi = hi, lo
        if ctx.max_reading_order is not None:
            hi = min(hi, ctx.max_reading_order)
        allowed = sorted({int(c["reading_order"]) for c in b.chunks.chunks if lo <= int(c["reading_order"]) <= hi})
        if not allowed:
            raise ToolNotFound("该章节范围内没有可读的内容（可能超出了当前阅读进度）")
        flt = flt.with_reading_orders(tuple(allowed))
        note["chapters"] = [allowed[0], allowed[-1]]
    if entity_id:
        if b.entity_chunks is None:
            raise ToolInputError("当前环境不支持按实体限定检索，请去掉 entity_id")
        from novel_rag.config import get_settings

        ids = b.entity_chunks(entity_id, flt)
        if not ids:
            raise ToolNotFound(f"在当前范围和阅读进度内没有提到实体 {entity_id} 的段落")
        if len(ids) > get_settings().tools.entity_filter_max_chunks:
            note["entity_filter"] = "ignored_too_broad"
        else:
            flt = flt.with_chunk_ids(tuple(ids))
            note["entity_chunks"] = len(ids)
    return flt, note


def run_search(b: ToolBackends, query: str, top_k: int, flt: SearchFilter) -> tuple[list[Hit], dict[str, float]]:
    """检索链路本身（固定 RAG 与 search_text 工具共用）。GPU 模型由共享实例串行处理。"""
    if b.retriever is None:
        raise ToolInputError("当前环境没有原文检索后端")
    with b.gpu_lock:
        hits = b.retriever.search(query, top_k, flt)
        lat = dict(getattr(b.retriever, "last_latency_ms", {}) or {})
    return hits, lat


class SearchTextArgs(_Args):
    query: str = Field(min_length=1, max_length=200, description="检索语句：写成陈述句或关键词组合，带上人名、地名")
    top_k: int = Field(5, ge=1, le=8, description="返回段落数，默认 5")
    volume: str | None = Field(None, description="只在这一卷里检索（卷 ID）")
    chapter_from: str | None = Field(None, description="章节范围起点（章节编号 卷ID:章号，含）")
    chapter_to: str | None = Field(None, description="章节范围终点（含）")
    entity_id: str | None = Field(None, description="只在提到该实体的段落里检索（实体 ID 来自 find_entity）")


class SearchTextArgsNoEntity(_Args):
    # 去掉全部图谱的消融：原文检索不再支持按实体限定。这里不能写类 docstring——它会进参数 schema，被模型看到
    query: str = Field(min_length=1, max_length=200, description="检索语句：写成陈述句或关键词组合，带上人名、地名")
    top_k: int = Field(5, ge=1, le=8, description="返回段落数，默认 5")
    volume: str | None = Field(None, description="只在这一卷里检索（卷 ID）")
    chapter_from: str | None = Field(None, description="章节范围起点（章节编号 卷ID:章号，含）")
    chapter_to: str | None = Field(None, description="章节范围终点（含）")


SEARCH_TEXT_DESC_NO_ENTITY = (
    "原文检索（BM25 + 向量 + rerank）：按语义找相关原文段落，返回段落全文及 chunk_id、卷名和章节。"
    "什么时候用：找某个情节、细节、对话的原文证据；已知章节时可以用 volume / 章节范围缩小范围。"
    "多跳问题先查出桥接实体，再针对桥接实体单独检索。"
)

SEARCH_TEXT_DESC = (
    "原文检索（BM25 + 向量 + rerank）：按语义找相关原文段落，返回段落全文及 chunk_id、卷名和章节。"
    "什么时候用：找某个情节、细节、对话的原文证据；已知人物或章节时可以用 entity_id / volume / 章节范围缩小范围。"
    "多跳问题先查出桥接实体，再针对桥接实体单独检索。"
)


def search_text(args: SearchTextArgs | SearchTextArgsNoEntity, ctx: ToolContext, b: ToolBackends) -> ToolResult:
    from novel_rag.config import get_settings

    cfg = get_settings().tools
    flt, note = make_filter(
        b, ctx, volume=args.volume, chapter_from=args.chapter_from, chapter_to=args.chapter_to,
        entity_id=getattr(args, "entity_id", None),
    )  # fmt: skip
    hits, lat = run_search(b, args.query, min(args.top_k, cfg.search_max_top_k), flt)
    if not hits:
        raise ToolNotFound("没有检索到相关段落，换个说法或放宽范围")
    blocks, shown, omitted, ros = [], [], [], []
    used = 0
    for h in hits:
        c = b.chunks.by_id(h.chunk_id)
        block = f"{chunk_header(b, c)}\n{c['text']}"
        ros.append(int(c["reading_order"]))
        if used + len(block) > cfg.search_max_chars and shown:
            omitted.append(h.chunk_id)
            continue
        blocks.append(block)
        shown.append(h.chunk_id)
        used += len(block) + 2
    text = "\n\n".join(blocks)
    if omitted:
        text += f"\n\n（另有 {len(omitted)} 段因长度上限未展示：{'、'.join(omitted)}；需要时用 read_text 读取）"
    return ToolResult(
        "search_text", True, text, shown_chunk_ids=shown, ref_chunk_ids=omitted, reading_orders=ros,
        truncated=bool(omitted),
        meta={"hits": [[h.chunk_id, round(h.score, 4)] for h in hits],
              "latency": {k: round(v, 1) for k, v in lat.items()}, **note},
    )  # fmt: skip


class ReadTextArgs(_Args):
    chunk_id: str = Field(description="要读的段落编号（卷ID:章号:段号）；读一章开头用 卷ID:章号:0000")
    before: int = Field(1, ge=0, le=3, description="向前扩展几段（不跨章）")
    after: int = Field(1, ge=0, le=3, description="向后扩展几段（不跨章）")


READ_TEXT_DESC = (
    "读原文：按 chunk_id 读取一段原文，可以向前后扩展几段（不跨章）。"
    "什么时候用：检索结果被截断、需要上下文确认指代，或从图谱 / 摘要给出的证据编号回到原文核实。"
)


def read_text(args: ReadTextArgs, ctx: ToolContext, b: ToolBackends) -> ToolResult:
    from novel_rag.config import get_settings

    cfg = get_settings().tools
    m = CHUNK_ID.match(args.chunk_id.strip())
    if not m:
        raise ToolInputError(f"chunk_id 格式应为 卷ID:章号:段号，收到 {args.chunk_id!r}")
    book, chap, seq = m.group("book"), int(m.group("chap")), int(m.group("seq"))
    center = b.chunks.get(book, chap, seq)
    flt = SearchFilter(book_ids=ctx.book_ids, max_reading_order=ctx.max_reading_order)
    if center is None or not flt.accepts(center):
        raise ToolNotFound(f"段落 {args.chunk_id} 不存在或不可读")
    w_before, w_after = min(args.before, cfg.read_max_window), min(args.after, cfg.read_max_window)
    chunks = []
    for s in range(seq - w_before, seq + w_after + 1):
        c = b.chunks.get(book, chap, s)
        if c is not None and c["chunk_type"] == center["chunk_type"] and flt.accepts(c):
            chunks.append(c)
    _, _, body = _stitch(chunks)
    ids = [c["chunk_id"] for c in chunks]
    head = f"【{ids[0]} … {ids[-1]}】" if len(ids) > 1 else f"【{ids[0]}】"
    text = f"{head}{b.where(book, chap)}\n{body}"
    truncated = False
    if len(text) > cfg.read_max_chars:
        # 超长时缩回中心段落，保证引用编号与展示的原文一致
        chunks, ids = [center], [center["chunk_id"]]
        text = f"【{ids[0]}】{b.where(book, chap)}\n{center['text']}（扩展范围过长，只展示中心段落）"
        truncated = True
    return ToolResult(
        "read_text", True, text, shown_chunk_ids=ids, reading_orders=[int(c["reading_order"]) for c in chunks],
        truncated=truncated,
    )  # fmt: skip
