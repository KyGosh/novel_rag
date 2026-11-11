"""P4 工具层：参数 schema、输出格式（chunk_id + 卷名 · 章节）、注入式阅读进度、出错处理与重试、防剧透断言。"""

import itertools
import json

import pytest

from novel_rag.tools import SpoilerLeak, Toolbox, ToolContext
from novel_rag.tools.base import TransientToolError, is_transient
from tests.unit.tool_fakes import make_backends

ALL_BOOKS = ("demo-1", "demo-2")


def _tb(**kw) -> Toolbox:
    return Toolbox(make_backends(**kw), sleep=lambda s: None)


def ctx(max_ro):
    return ToolContext(max_reading_order=max_ro, book_ids=ALL_BOOKS)


def test_schemas_hide_injected_context():
    tb = _tb()
    names = [s["name"] for s in tb.schemas()]
    assert names == ["search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query"]
    for s in tb.schemas():
        props = json.dumps(s["parameters"], ensure_ascii=False)
        assert "reading_order" not in props and "book_ids" not in props, s["name"]
        assert "什么时候用" in s["description"]


def test_search_text_output_has_chunk_id_and_location():
    r = _tb().call("search_text", {"query": "白狐"}, ctx(None))
    assert r.ok and r.shown_chunk_ids
    assert "【demo-1:0000:0000】北岭记 · 序章" in r.text
    assert r.meta["hits"][0][0] == r.shown_chunk_ids[0]


def test_reading_order_is_injected_and_cannot_be_overridden_by_model():
    tb = _tb()
    # 模型试图传入阅读进度：schema 不允许额外字段
    r = tb.call("search_text", {"query": "石峰", "max_reading_order": 99}, ctx(1))
    assert not r.ok and r.error_kind == "invalid_args"
    r = tb.call("search_text", {"query": "石峰战死", "top_k": 8}, ctx(1))
    assert all(tb.backends.chunks.by_id(c)["reading_order"] <= 1 for c in r.chunk_ids)
    assert tb.backends.retriever.calls[-1].max_reading_order == 1


def test_search_text_ranges_and_entity_filter():
    tb = _tb()
    r = tb.call("search_text", {"query": "柳月", "volume": "demo-2"}, ctx(None))
    assert r.shown_chunk_ids and all(c.startswith("demo-2") for c in r.shown_chunk_ids)
    r = tb.call("search_text", {"query": "白狐", "chapter_from": "demo-1:0001", "chapter_to": "demo-1:0002"}, ctx(None))
    assert {c.split(":")[1] for c in r.shown_chunk_ids} <= {"0001", "0002"}
    r = tb.call("search_text", {"query": "白狐", "entity_id": "demo:person:柳月"}, ctx(None))
    assert all("柳月" in tb.backends.chunks.by_id(c)["text"] for c in r.shown_chunk_ids)
    # 章节范围整体超出阅读进度 → 找不到，而不是悄悄返回后文
    r = tb.call("search_text", {"query": "石岩", "chapter_from": "demo-2:0000"}, ctx(2))
    assert not r.ok and r.error_kind == "not_found"
    r = tb.call("search_text", {"query": "x", "volume": "demo-9"}, ctx(None))
    assert r.error_kind == "invalid_args"


def test_read_text_window_stays_in_chapter_and_respects_progress():
    tb = _tb()
    r = tb.call("read_text", {"chunk_id": "demo-1:0001:0001", "before": 3, "after": 3}, ctx(None))
    assert r.shown_chunk_ids == ["demo-1:0001:0000", "demo-1:0001:0001", "demo-1:0001:0002"]
    assert "北岭记 · 阿青" in r.text and "柳伯的女儿柳月" in r.text
    r = tb.call("read_text", {"chunk_id": "demo-2:0001:0001"}, ctx(3))
    assert not r.ok and r.error_kind == "not_found" and "真相" not in r.text
    assert tb.call("read_text", {"chunk_id": "bad"}, ctx(None)).error_kind == "invalid_args"
    assert tb.call("read_text", {"chunk_id": "demo-1:0001:0001", "after": 9}, ctx(None)).error_kind == "invalid_args"


def test_summary_tools():
    tb = _tb()
    r = tb.call("search_summaries", {"query": "白狐 山神"}, ctx(1))
    assert r.ok and "demo-1:0001 北岭记 · 阿青（视角：阿青）" in r.text and "demo-2" not in r.text
    r = tb.call("read_summary", {"chapter": "demo-1:0002"}, ctx(None))
    assert r.ok and r.ref_chunk_ids == ["demo-1:0002:0000"] and "证据：demo-1:0002:0000" in r.text
    assert tb.call("read_summary", {"chapter": "demo-2:0001"}, ctx(2)).error_kind == "not_found"


def test_graph_tools_render_evidence_and_single_evidence_note():
    tb = _tb()
    r = tb.call("find_entity", {"name": "柳月"}, ctx(None))
    assert r.ok and "demo:person:柳月" in r.text and r.meta["candidates"] == ["demo:person:柳月"]
    r = tb.call("graph_query", {"op": "profile", "entity": "demo:person:阿青"}, ctx(None))
    assert r.ok and "证据 1 条，单条证据，建议回原文核实" in r.text
    assert set(r.ref_chunk_ids) == {"demo-1:0001:0002", "demo-2:0001:0001"}
    assert "北岭记 · 阿青" in r.text  # 证据带"卷名 · 章节标题"
    r = tb.call("graph_query", {"op": "between", "entity": "demo:person:阿青"}, ctx(None))
    assert r.error_kind == "invalid_args" and "other" in r.error
    r = tb.call("graph_query", {"op": "neighbors", "entity": "demo:person:阿青", "relation_types": ["FOO"]}, ctx(None))
    assert r.error_kind == "invalid_args" and "FOO" in r.error
    r = tb.call("graph_query", {"op": "paths", "entity": "demo:person:阿青", "other": "demo:person:柳月"}, ctx(None))
    assert r.error_kind == "not_found"
    # 石峰在 ro 3 才出现：ro 2 时查不到，也看不到他与阿青的亲属关系
    assert tb.call("find_entity", {"name": "石峰"}, ctx(2)).error_kind == "not_found"
    r = tb.call("graph_query", {"op": "profile", "entity": "demo:person:阿青"}, ctx(2))
    assert "石峰" not in r.text and "demo-2" not in r.text


def test_unknown_tool_and_bad_json_are_structured_errors():
    tb = _tb()
    r = tb.call("delete_everything", {}, ctx(None))
    assert not r.ok and r.error_kind == "unknown_tool" and "search_text" in r.text
    r = tb.call("search_text", None, ctx(None), args_error="参数不是合法的 JSON：x")
    assert r.error_kind == "invalid_args" and r.text.startswith("[错误]")
    r = tb.call("search_text", {"query": ""}, ctx(None))
    assert r.error_kind == "invalid_args" and "query" in r.error


def test_transient_errors_are_retried_then_returned():
    tb = _tb()
    calls = {"n": 0}
    real = tb.backends.retriever.search

    def flaky(query, top_k, flt=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientToolError("timeout")
        return real(query, top_k, flt)

    tb.backends.retriever.search = flaky
    r = tb.call("search_text", {"query": "白狐"}, ctx(None))
    assert r.ok and r.attempts == 3

    class ConnectionTimeout(Exception):  # 模拟某家客户端的超时异常（按类名识别）
        pass

    assert is_transient(ConnectionTimeout())

    def always_down(query, top_k, flt=None):
        raise ConnectionTimeout("es down")

    tb.backends.retriever.search = always_down
    r = tb.call("search_text", {"query": "白狐"}, ctx(None))
    assert not r.ok and r.error_kind == "transient" and r.attempts == tb.max_retries + 1

    def broken(query, top_k, flt=None):
        raise KeyError("boom")

    tb.backends.retriever.search = broken
    r = tb.call("search_text", {"query": "白狐"}, ctx(None))
    assert not r.ok and r.error_kind == "backend" and r.attempts == 1


@pytest.mark.parametrize("leak", ["text", "graph"])
def test_spoiler_backstop_catches_leaking_backend(leak):
    tb = _tb(leak_text=leak == "text", leak_graph=leak == "graph")
    with pytest.raises(SpoilerLeak):
        if leak == "text":
            tb.call("search_text", {"query": "真相 石峰 私生子", "top_k": 8}, ctx(1))
        else:
            tb.call("graph_query", {"op": "profile", "entity": "demo:person:阿青"}, ctx(1))


CALLS = [
    ("search_text", {"query": "白狐 山神 石峰", "top_k": 8}),
    ("search_text", {"query": "柳月", "entity_id": "demo:person:柳月", "top_k": 8}),
    ("read_text", {"chunk_id": "demo-1:0002:0000", "after": 3}),
    ("search_summaries", {"query": "真相 山神", "top_k": 8}),
    ("read_summary", {"chapter": "demo-1:0001"}),
    ("find_entity", {"name": "阿"}),
    ("graph_query", {"op": "profile", "entity": "demo:person:阿青"}),
    ("graph_query", {"op": "neighbors", "entity": "demo:person:阿青"}),
    ("graph_query", {"op": "mentions", "entity": "demo:person:柳月"}),
]


@pytest.mark.parametrize(("call", "max_ro"), list(itertools.product(CALLS, [0, 1, 2, 3, 4])))
def test_no_observation_exceeds_reading_progress(call, max_ro):
    """防剧透断言：任何工具、任何阅读进度下，观察中都不出现超过阅读进度的 chunk 或位置。"""
    tb = _tb()
    name, args = call
    r = tb.call(name, args, ctx(max_ro))
    for cid in r.chunk_ids:
        assert tb.backends.chunks.by_id(cid)["reading_order"] <= max_ro, (name, cid)
        assert cid not in r.text or tb.backends.chunks.by_id(cid)["reading_order"] <= max_ro
    assert all(x <= max_ro for x in r.reading_orders)
    # 观察文本中出现的任何 chunk 编号、原文也不超过阅读进度（错误信息复述模型自己给的编号除外）
    echoed = json.dumps(args, ensure_ascii=False)
    for c in tb.backends.chunks.chunks:
        if c["reading_order"] > max_ro and c["chunk_id"] not in echoed:
            assert c["chunk_id"] not in r.text and c["text"] not in r.text, (name, c["chunk_id"])
