"""切句与分块：句末标点、引号、省略号、超长句；分块不变量、重叠、尾块合并、版本号、token 预算。样例均为自拟。"""

import pytest

from novel_rag.config import ChunkConfig
from novel_rag.corpus import BookText
from novel_rag.ingestion.chunking import (
    ChunkingError,
    build_chunks,
    check_invariants,
    chunk_spans,
    chunk_version,
    load_chunks,
    write_chunks,
)
from novel_rag.ingestion.sentences import split_sentences
from novel_rag.llm.fake import FakeTokenCounter
from tests.unit.book_factory import make_book

CFG = ChunkConfig()


def sents(text: str, cfg: ChunkConfig = CFG) -> list[str]:
    return [text[s:e] for s, e in split_sentences(text, 0, len(text), cfg)]


def test_basic_sentence_end_and_closers():
    assert sents("他走了。她留下！真的吗？") == ["他走了。", "她留下！", "真的吗？"]
    assert sents("“你去哪儿？”他问。“北方。”") == ["“你去哪儿？”他问。", "“北方。”"]
    assert sents("他说完了（真的。）然后离开。") == ["他说完了（真的。）", "然后离开。"]
    assert sents("什么？！不可能。") == ["什么？！", "不可能。"]


def test_no_split_inside_quotes():
    assert sents("他说：“我们走吧。天快黑了。”大家点头。") == ["他说：“我们走吧。天快黑了。”大家点头。"]


def test_long_quote_can_split():
    cfg = ChunkConfig(long_quote_chars=10)
    out = sents("他说：“第一句话说得很长很长很长。第二句。”众人沉默。", cfg)
    assert out[0].endswith("很长很长很长。") and len(out) >= 2
    assert "".join(out) == "他说：“第一句话说得很长很长很长。第二句。”众人沉默。"


def test_ellipsis():
    assert sents("他犹豫了……然后开口。") == ["他犹豫了……然后开口。"]
    assert sents("“我不知道……”她说。") == ["“我不知道……”她说。"]
    assert sents("夜色渐浓……") == ["夜色渐浓……"]


def test_hard_cut_prefers_soft_breaks():
    cfg = ChunkConfig(sentence_hard_max=10)
    text = "一二三四五六，七八九十一二三四五六七八。"
    out = sents(text, cfg)
    assert "".join(out) == text and all(len(x) <= 10 for x in out)
    assert out[0] == "一二三四五六，"


def _book(tmp_path, chapters):
    make_book(tmp_path / "books", "demo-1", chapters)
    return BookText.load(tmp_path / "books" / "demo-1")


def _para(n_sent: int, seed: str) -> str:
    return "".join(f"{seed}第{i}句话写在这里，内容平淡无奇。" for i in range(n_sent))


SMALL = ChunkConfig(target_chars=60, max_chars=100, min_chars=25, overlap_chars=20)


def test_chunk_spans_invariants_and_overlap(tmp_path):
    book = _book(tmp_path, [("甲", [_para(6, "甲"), _para(5, "乙")]), ("乙", [_para(3, "丙")])])
    chunks, manifest = build_chunks(book, SMALL)
    check_invariants(book, chunks)  # 切片一致、不跨章、全覆盖、顺序单调
    ch1 = [c for c in chunks if c["chapter_idx"] == 1]
    assert len(ch1) >= 2
    for a, b in zip(ch1, ch1[1:], strict=False):
        assert b["char_start"] < a["char_end"]  # 有重叠
        assert a["char_end"] - b["char_start"] <= SMALL.overlap_chars  # 重叠不超过上限
        assert book.text[b["char_start"] - 1] in "。\n"  # 重叠从整句开始
    assert all(c["char_end"] - c["char_start"] <= SMALL.max_chars or c["text"].count("。") == 1 for c in chunks)
    assert chunks[0]["chunk_id"] == "demo-1:0001:0000" and manifest["n_chunks"] == len(chunks)


def test_short_tail_merged_into_previous(tmp_path):
    cfg = ChunkConfig(target_chars=40, max_chars=60, min_chars=30, overlap_chars=0)
    book = _book(tmp_path, [("甲", ["一" * 20 + "。" + "二" * 20 + "。" + "三" * 10 + "。"])])
    spans = chunk_spans(book, book.chapter(1), cfg)
    assert book.text[spans[-1][0] : spans[-1][1]].endswith("三" * 10 + "。")
    assert all(e - s >= cfg.min_chars for s, e in spans)


def test_chunk_never_crosses_chapter_and_skips_title(tmp_path):
    book = _book(tmp_path, [("甲章", ["短句。"]), ("乙章", ["另一句。"])])
    chunks, _ = build_chunks(book, SMALL)
    assert [c["text"] for c in chunks] == ["短句。", "另一句。"]
    assert [c["chapter_title"] for c in chunks] == ["甲章", "乙章"]


def test_deterministic_and_versioned(tmp_path):
    book = _book(tmp_path, [("甲", [_para(8, "甲")])])
    a, ma = build_chunks(book, SMALL)
    b, mb = build_chunks(book, SMALL)
    assert a == b and ma == mb
    assert chunk_version(SMALL) != chunk_version(SMALL.model_copy(update={"target_chars": 61}))
    assert chunk_version(SMALL) == chunk_version(SMALL.model_copy(update={"query_max_tokens": 1}))


def test_token_budget_and_roundtrip(tmp_path):
    book = _book(tmp_path, [("甲", [_para(8, "甲")])])
    chunks, manifest = build_chunks(book, SMALL, token_counter=FakeTokenCounter(), token_limit=1024)
    assert manifest["token_budget"]["ok"] and all(c["token_count"] == len(c["text"]) for c in chunks)
    _, tight = build_chunks(book, SMALL, token_counter=FakeTokenCounter(), token_limit=10)
    assert not tight["token_budget"]["ok"]
    write_chunks(tmp_path / "out", chunks, manifest)
    assert load_chunks(tmp_path / "out") == (chunks, manifest)


def test_invariant_violation_detected(tmp_path):
    book = _book(tmp_path, [("甲", [_para(3, "甲")])])
    chunks, _ = build_chunks(book, SMALL)
    chunks[0] = {**chunks[0], "text": "被改过的文本"}
    with pytest.raises(ChunkingError, match="切片不一致"):
        check_invariants(book, chunks)
