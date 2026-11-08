"""章节摘要：分段去重叠、引用校验（越界编号丢弃、无引用事件作废）、prompt 渲染、摘要检索的防剧透复核；
SearchFilter 的章节集合条件。样例全部自拟。"""

import asyncio
import json

import pytest

from novel_rag.corpus import BookText
from novel_rag.ingestion.summary import ChapterSummaryOut, chapter_segments, summarize_chapter, validate_summary
from novel_rag.llm.fake import FakeChatModel
from novel_rag.prompts import load_prompt
from novel_rag.retrieval.base import Hit
from novel_rag.retrieval.filters import SearchFilter
from novel_rag.retrieval.hybrid import FilterViolation
from novel_rag.retrieval.summary_store import SummaryRetriever
from tests.unit.book_factory import make_book, para


def _chapter(tmp_path):
    make_book(tmp_path, "demo-1", [("米拉", [para("米拉", 300), para("阿岚", 300)])])
    book = BookText.load(tmp_path / "demo-1")
    ch = book.chapter(1)
    b = ch.body_start
    chunks = [
        {"chunk_id": "demo-1:0001:0000", "char_start": b, "char_end": b + 320},
        {"chunk_id": "demo-1:0001:0001", "char_start": b + 280, "char_end": ch.char_end},  # 与前块重叠 40 字
    ]
    return book, ch, chunks


def test_segments_drop_overlap(tmp_path):
    book, ch, chunks = _chapter(tmp_path)
    segs = chapter_segments(book, chunks)
    assert [s.label for s in segs] == [1, 2]
    joined = segs[0].text + segs[1].text
    assert len(joined.replace("\n", "")) == len(book.text[ch.body_start : ch.char_end].replace("\n", ""))


def test_validate_summary_refs(tmp_path):
    book, ch, chunks = _chapter(tmp_path)
    segs = chapter_segments(book, chunks)
    out = ChapterSummaryOut.model_validate(
        {
            "summary": "米拉在港口等船。",
            "events": [
                {"description": "米拉等船", "participants": ["米拉"], "segments": [1, 9]},
                {"description": "凭空出现的事件", "participants": [], "segments": [7]},
                {"description": "阿岚出场", "participants": ["阿岚"], "location": "港口", "segments": [2, 2]},
            ],
            "characters": ["米拉", "阿岚", "米拉"],
        }
    )
    events, stats = validate_summary(out, ch, segs)
    assert [e["chunk_ids"] for e in events] == [["demo-1:0001:0000"], ["demo-1:0001:0001"]]
    assert stats == {"invalid_refs": 2, "dropped_events": 1}
    assert events[0]["event_id"] == "demo-1:0001:e01" and events[1]["location"] == "港口"


def test_summarize_chapter_with_fake_llm(tmp_path):
    book, ch, chunks = _chapter(tmp_path)
    reply = json.dumps(
        {"summary": "摘要", "events": [{"description": "事件", "participants": ["米拉"], "segments": [1]}],
         "characters": ["米拉", "米拉", "阿岚"]}, ensure_ascii=False
    )  # fmt: skip
    chat = FakeChatModel(lambda msgs: reply)
    rec = asyncio.run(summarize_chapter(book, ch, chunks, chat, load_prompt("chapter_summary"), chunk_version="v"))
    assert rec["chapter_key"] == "demo-1:0001" and rec["characters"] == ["米拉", "阿岚"]
    assert rec["prompt_version"] == "chapter_summary.v1" and rec["chunk_version"] == "v"


def test_prompt_render_keeps_braces_in_text():
    msgs = load_prompt("chapter_summary").render(
        book_title="卷", chapter_title="章", pov="无", section_type="narrative", section_note="", segments="原文{x}"
    )
    assert "原文{x}" in msgs[1]["content"] and "{segments}" not in msgs[1]["content"]


def test_filter_reading_orders():
    f = SearchFilter(max_reading_order=10).with_reading_orders((7, 3, 3))
    assert f.reading_orders == (3, 7)
    assert "reading_order in [3, 7]" in f.to_milvus_expr() and "reading_order <= 10" in f.to_milvus_expr()
    assert {"terms": {"reading_order": [3, 7]}} in f.to_es_filter()
    assert f.accepts({"reading_order": 3}) and not f.accepts({"reading_order": 4})
    assert f.with_max_reading_order(2).reading_orders == (3, 7)  # 改阅读进度不丢章节条件


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.name = "fake"
        self.seen = None

    def search(self, q, k, flt):
        self.seen = flt
        return self.hits


def test_summary_retriever_rechecks_spoiler_filter():
    good = [Hit(1, "demo-1:0001", 1.0, {"reading_order": 1, "book_id": "demo-1"})]
    es = FakeStore(good)
    r = SummaryRetriever(es, None, None, bm25_top_k=5, vector_top_k=5, rrf_k=60)
    out = r.search("米拉", 5, SearchFilter(max_reading_order=3, chunk_types=("text",)))
    assert [h.chunk_id for h in out] == ["demo-1:0001"]
    assert es.seen.chunk_types is None and es.seen.max_reading_order == 3  # 摘要索引没有 chunk_type 字段
    leak = FakeStore([Hit(1, "demo-1:0009", 1.0, {"reading_order": 9, "book_id": "demo-1"})])
    with pytest.raises(FilterViolation):
        SummaryRetriever(leak, None, None, bm25_top_k=5, vector_top_k=5, rrf_k=60).search(
            "x", 5, SearchFilter(max_reading_order=3)
        )
