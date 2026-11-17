"""P6 交互式问答服务：阅读进度换算、模式选择、返回结果不含原文、事件转发。用注入的假系统，不连真实服务。"""

import json

import pytest

from novel_rag.api.service import Chapter, QAService
from novel_rag.config import get_settings
from novel_rag.llm.fake import Call, Say
from tests.unit.test_agent import ACT, ANSWER, PLAN, REFLECT, answer, reflect, system
from tests.unit.tool_fakes import make_backends

GOOD = "demo-1:0002:0000"


def service(script=None):
    b = make_backends()
    chapters = sorted(
        {(c["book_id"], c["chapter_idx"]): Chapter(c["book_id"], c["chapter_idx"], c["chapter_title"],
                                                   c["reading_order"]) for c in b.chunks.chunks}.values(),
        key=lambda ch: ch.reading_order,
    )  # fmt: skip
    script = script or {
        PLAN: [Say("先查白狐")],
        ACT: [Call.one("search_text", {"query": "白狐 取名"}), Call.one("finish", {"reason": "够了"})],
        REFLECT: [reflect([{"claim": "白狐叫雪团", "chunk_id": GOOD, "quote": "取名雪团"}])],
        ANSWER: [answer("雪团", [GOOD])],
    }  # fmt: skip
    agent, _ = system(script)
    return QAService(get_settings(), ("demo-1", "demo-2"), b.book_titles, chapters, b.where_chunk,
                     {"agent": lambda: agent})  # fmt: skip


def test_reading_order_of_volume_and_chapter():
    svc = service()
    last1 = max(ch.reading_order for ch in svc.chapters if ch.book_id == "demo-1")
    assert svc.reading_order_of(1) == svc.reading_order_of("demo-1") == last1
    ch = svc.chapters[1]
    assert svc.reading_order_of(ch.book_id, ch.chapter_idx) == ch.reading_order
    with pytest.raises(ValueError):
        svc.reading_order_of(3)
    with pytest.raises(ValueError):
        svc.reading_order_of("demo-1", 999)
    assert svc.describe_progress(None).startswith("不限")
    assert "北岭记" in svc.describe_progress(ch.reading_order)
    vols = svc.volumes()
    assert [v["book_id"] for v in vols] == ["demo-1", "demo-2"] and vols[0]["chapters"][0]["idx"] == 0


def test_ask_returns_answer_notes_and_citations_without_raw_text():
    events = []
    res = service().ask("白狐后来被取了什么名字？", "agent", None, on_event=events.append)
    assert res["answer"] == "雪团" and res["mode_name"] == "Agent"
    assert res["citations"] == [{"chunk_id": GOOD, "where": "北岭记 · 柳月"}]
    assert res["notes"][0]["quote"] == "取名雪团" and res["plan"] == "先查白狐"
    assert res["cost"]["llm_calls"] == 5 and res["route"] is None
    for c in make_backends().chunks.chunks:  # 不返回整段原文
        assert c["text"] not in json.dumps(res, ensure_ascii=False)
    assert [e["type"] for e in events][0] == "plan" and events[-1]["type"] == "answer"


def test_ask_validates_input():
    svc = service()
    with pytest.raises(ValueError):
        svc.ask("  ", "agent")
    with pytest.raises(ValueError):
        svc.ask("问" * 201, "agent")
    with pytest.raises(ValueError):
        svc.ask("问题", "closed")


def test_spoiler_progress_is_passed_to_tools():
    # 阅读进度 0：只能看到第 0 章，白狐取名那段（更靠后）检索不到
    acts = [Call.one("search_text", {"query": "白狐 取名"}), Call.one("finish", {"reason": "x"})]
    script = {PLAN: [Say("p")], ACT: acts, REFLECT: [reflect([])], ANSWER: [answer("原文未提及", [])]}
    events = []
    res = service(script).ask("白狐后来被取了什么名字？", "agent", 0, on_event=events.append)
    shown = [c for e in events if e["type"] == "tool" for c in e["shown"]]
    chunks = make_backends().chunks
    assert all(chunks.by_id(c)["reading_order"] <= 0 for c in shown)
    assert res["abstained"] and res["max_reading_order"] == 0
