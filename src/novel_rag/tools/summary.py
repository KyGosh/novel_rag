"""章节摘要工具：摘要检索（定位"某事发生在哪几章"）与读摘要（摘要 + 事件列表，事件带证据 chunk_id）。

摘要是模型生成的概括（P3 忠实度抽查 95.9%），只用于定位；结论要回原文核实。
"""

from __future__ import annotations

from pydantic import Field

from novel_rag.retrieval.filters import SearchFilter
from novel_rag.tools.backends import ToolBackends
from novel_rag.tools.base import ToolContext, ToolInputError, ToolNotFound, ToolResult, truncate
from novel_rag.tools.text import _Args, parse_chapter


def _pov(rec: dict) -> str:
    return f"（视角：{rec['pov']}）" if rec.get("pov") else ""


class SearchSummariesArgs(_Args):
    query: str = Field(min_length=1, max_length=200, description="要定位的情节或事件")
    top_k: int = Field(5, ge=1, le=8, description="返回章节数，默认 5")
    volume: str | None = Field(None, description="只在这一卷里找（卷 ID）")


SEARCH_SUMMARIES_DESC = (
    "摘要检索：在各章的情节摘要里检索，返回最相关的章节编号、标题和摘要片段。"
    "什么时候用：先定位某件事发生在哪几章（跨章节情节、事件先后、某人经历），再用 read_summary 或 "
    "search_text（限定章节范围）、read_text 读原文。摘要是概括，不能直接作为证据引用。"
)


def search_summaries(args: SearchSummariesArgs, ctx: ToolContext, b: ToolBackends) -> ToolResult:
    from novel_rag.config import get_settings

    cfg = get_settings().tools
    if b.summary_retriever is None or b.summaries is None:
        raise ToolInputError("当前环境没有章节摘要")
    books = ctx.book_ids
    if args.volume:
        if args.volume not in books:
            raise ToolInputError(f"没有卷 {args.volume!r}；可选：{', '.join(books)}")
        books = (args.volume,)
    flt = SearchFilter(book_ids=books, max_reading_order=ctx.max_reading_order)
    with b.gpu_lock:
        hits = b.summary_retriever.search(args.query, args.top_k, flt)
    if not hits:
        raise ToolNotFound("没有找到相关章节")
    lines, ros = [], []
    for h in hits:
        rec = b.summaries.get(h.chunk_id)
        if rec is None:
            continue
        ros.append(int(rec["reading_order"]))
        snippet = rec["summary"][: cfg.summary_snippet_chars]
        more = "……" if len(rec["summary"]) > cfg.summary_snippet_chars else ""
        lines.append(f"- {h.chunk_id} {b.where(rec['book_id'], rec['chapter_idx'])}{_pov(rec)}：{snippet}{more}")
    text, cut = truncate("\n".join(lines), cfg.summary_search_max_chars)
    return ToolResult(
        "search_summaries", True, text, reading_orders=ros, truncated=cut,
        meta={"chapters": [h.chunk_id for h in hits]},
    )  # fmt: skip


class ReadSummaryArgs(_Args):
    chapter: str = Field(description="章节编号（卷ID:章号），来自 search_summaries 或 chunk_id 的前两段")


READ_SUMMARY_DESC = (
    "读摘要：读取一章的情节摘要和关键事件列表，每个事件带证据 chunk_id。"
    "什么时候用：确认一章里发生了什么、事件的参与者，并拿到可以用 read_text 核实的证据编号。"
)


def read_summary(args: ReadSummaryArgs, ctx: ToolContext, b: ToolBackends) -> ToolResult:
    from novel_rag.config import get_settings

    cfg = get_settings().tools
    if b.summaries is None:
        raise ToolInputError("当前环境没有章节摘要")
    book, chap = parse_chapter(args.chapter)
    key = f"{book}:{chap:04d}"
    rec = b.summaries.get(key)
    visible = (
        rec is not None
        and book in ctx.book_ids
        and (ctx.max_reading_order is None or int(rec["reading_order"]) <= ctx.max_reading_order)
    )
    if not visible:
        raise ToolNotFound(f"章节 {key} 不存在或不可读")
    lines = [f"{key} {b.where(book, chap)}{_pov(rec)}", f"摘要：{rec['summary']}", "事件："]
    refs: list[str] = []
    for e in rec.get("events", []):
        ids = list(e.get("chunk_ids", []))
        refs += ids
        who = "、".join(e.get("participants", []))
        lines.append(f"- {e['description']}（参与：{who or '不详'}；证据：{'、'.join(ids) or '无'}）")
    text, cut = truncate("\n".join(lines), cfg.read_summary_max_chars)
    ros = [int(rec["reading_order"])] + [r for r in (b.chunk_ro(c) for c in refs) if r is not None]
    return ToolResult(
        "read_summary", True, text, ref_chunk_ids=list(dict.fromkeys(refs)), reading_orders=ros, truncated=cut
    )
