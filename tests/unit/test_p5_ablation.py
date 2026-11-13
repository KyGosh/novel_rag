"""P5 消融开关与分析函数：去掉全部图谱、图谱范围（运行环境注入）、行动环节看原始历史、题目分组与证据轨迹。

全部用脚本化 Fake 模型 + 假后端和自造数据，不调用真实模型和服务。
"""

import json

from novel_rag.agent.loop import format_history
from novel_rag.agent.system import AgentSystem
from novel_rag.config import get_settings
from novel_rag.eval.dataset import EvalItem, Evidence
from novel_rag.eval.metrics import Span
from novel_rag.llm.fake import Call, Say, ScriptedChatModel
from novel_rag.qa.trail import evidence_state, evidence_trail, gain_source, question_groups
from novel_rag.tools import Toolbox, ToolContext
from tests.unit.tool_fakes import FakeGraph, make_backends

BOOKS = ("demo-1", "demo-2")
PLAN, ACT, ACT_RAW, REFLECT, ANSWER = (
    "agent_plan.v1", "agent_act.v1", "agent_act_raw.v1", "agent_reflect.v1", "qa_answer.v1",
)  # fmt: skip


def ctx(max_ro=None, scope=None):
    return ToolContext(max_reading_order=max_ro, book_ids=BOOKS, graph_entity_types=scope)


def _tb(**kw) -> Toolbox:
    return Toolbox(make_backends(**kw), sleep=lambda s: None)


# ---------------------------------------------------------------- 去掉全部图谱


def test_without_graph_removes_graph_tools_and_entity_filter():
    tb = _tb().without_graph()
    names = [s["name"] for s in tb.schemas()]
    assert names == ["search_text", "read_text", "search_summaries", "read_summary"]
    st = next(s for s in tb.schemas() if s["name"] == "search_text")
    assert "entity_id" not in json.dumps(st, ensure_ascii=False)  # 参数和说明里都不再出现
    # 模型仍然传 entity_id：按非法参数拒绝；图谱工具不存在
    r = tb.call("search_text", {"query": "白狐", "entity_id": "demo:person:柳月"}, ctx())
    assert not r.ok and r.error_kind == "invalid_args"
    assert tb.call("find_entity", {"name": "柳月"}, ctx()).error_kind == "unknown_tool"
    # 其余参数照常
    r = tb.call("search_text", {"query": "柳月", "volume": "demo-2"}, ctx())
    assert r.ok and all(c.startswith("demo-2") for c in r.shown_chunk_ids)
    # 原工具箱不受影响
    assert "entity_id" in json.dumps(_tb().schemas(), ensure_ascii=False)


# ---------------------------------------------------------------- 图谱范围


class ScopedGraph(FakeGraph):
    ENTITIES = {
        **FakeGraph.ENTITIES,
        "demo:location:青石村": {"name": "青石村", "type": "Location", "first_ro": 0, "aliases": []},
    }
    RELS = [*FakeGraph.RELS, ("demo:person:阿青", "demo:location:青石村", "LIVES_IN", None, [("demo-1:0000:0001", 0)])]


def _scoped_tb() -> Toolbox:
    b = make_backends()
    b.graph = ScopedGraph()
    return Toolbox(b, sleep=lambda s: None)


def test_graph_scope_default_is_full_graph():
    tb = _scoped_tb()
    assert tb.call("find_entity", {"name": "青石村"}, ctx()).ok
    r = tb.call("graph_query", {"op": "profile", "entity": "demo:person:阿青"}, ctx())
    assert r.ok and "青石村" in r.text


def test_graph_scope_person_hides_other_entity_types():
    tb, person = _scoped_tb(), ("Person",)
    # 范围外的实体：查不到；范围内的照常
    r = tb.call("find_entity", {"name": "青石村"}, ctx(scope=person))
    assert not r.ok and r.error_kind == "not_found"
    assert tb.call("find_entity", {"name": "阿青"}, ctx(scope=person)).ok
    # 档案和邻居里不再出现范围外的对方
    r = tb.call("graph_query", {"op": "profile", "entity": "demo:person:阿青"}, ctx(scope=person))
    assert r.ok and "青石村" not in r.text and "柳月" in r.text
    r = tb.call("graph_query", {"op": "neighbors", "entity": "demo:person:阿青"}, ctx(scope=person))
    assert r.ok and "青石村" not in r.text
    # 以范围外实体为查询对象或另一端：找不到
    r = tb.call("graph_query", {"op": "profile", "entity": "demo:location:青石村"}, ctx(scope=person))
    assert not r.ok and r.error_kind == "not_found"
    r = tb.call("graph_query", {"op": "between", "entity": "demo:person:阿青", "other": "demo:location:青石村"},
                ctx(scope=person))  # fmt: skip
    assert not r.ok and r.error_kind == "not_found"
    # 模型点名要范围外的类型：找不到；路径查询带上节点类型限定
    r = tb.call("graph_query", {"op": "neighbors", "entity": "demo:person:阿青", "other_types": ["Location"]},
                ctx(scope=person))  # fmt: skip
    assert not r.ok and r.error_kind == "not_found"
    tb.call(
        "graph_query", {"op": "paths", "entity": "demo:person:阿青", "other": "demo:person:柳月"}, ctx(scope=person)
    )
    assert tb.backends.graph.last_paths_kw["node_types"] == person


def test_graph_scope_is_not_a_model_visible_parameter():
    tb = _scoped_tb()
    for s in tb.schemas():
        assert "graph_entity_types" not in json.dumps(s, ensure_ascii=False)
    r = tb.call("find_entity", {"name": "阿青", "graph_entity_types": ["Location"]}, ctx(scope=("Person",)))
    assert not r.ok and r.error_kind == "invalid_args"


# ---------------------------------------------------------------- 行动环节看原始历史


def test_format_history_keeps_latest_rounds_within_cap():
    assert format_history([], 100) == "（暂无）"
    full = format_history(["甲" * 50, "乙" * 50], 1000)
    assert full.index("第 1 轮") < full.index("第 2 轮") and "省略" not in full
    cut = format_history(["甲" * 500, "乙" * 50], 300)
    assert "乙" * 50 in cut and "截断" in cut and len(cut) < 400  # 最近一轮完整保留，更早的截断
    dropped = format_history(["甲" * 500, "乙" * 280], 300)
    assert "乙" * 280 in dropped and "甲" not in dropped and "第 1 轮及更早的观察因长度上限省略" in dropped


def _item():
    return EvalItem(
        id="q1", type="multihop", question="白狐后来被取了什么名字？", answer="雪团", source="generated", split="dev",
        evidence=[Evidence(book_id="demo-1", chapter_idx=2, char_start=0, char_end=10)], max_reading_order=4,
    )  # fmt: skip


def _reflect(notes):
    return Say(json.dumps({"new_notes": notes, "status": "已确认", "sufficient": False}, ensure_ascii=False))


def _system(script, **agent_over):
    s = get_settings()
    s = s.model_copy(update={"agent": s.agent.model_copy(update=agent_over)})
    m = ScriptedChatModel(script, tokens_per_call=(100, 20))
    tb = Toolbox(make_backends(), sleep=lambda x: None)
    return AgentSystem(tb, s, scope=BOOKS, count_tokens=len, agent_chat=m, reflect_chat=m, answer_chat=m), m


def test_raw_memory_mode_shows_observations_to_act_and_keeps_notes():
    act = [Call.one("search_text", {"query": "白狐 取名"}), Call.one("finish", {"reason": "ok"})]
    sysm, m = _system({
        PLAN: [Say("计划")], ACT_RAW: act,
        REFLECT: [_reflect([{"claim": "白狐名叫雪团", "chunk_id": "demo-1:0002:0000"}])],
        ANSWER: [Say(json.dumps({"answer": "雪团", "rationale": "", "citations": ["demo-1:0002:0000"]}))],
    }, act_memory="raw")  # fmt: skip
    out = sysm.run(_item())
    acts = [h for h in m.history if h["prompt_version"] == ACT_RAW]
    assert len(acts) == 2 and not [h for h in m.history if h["prompt_version"] == ACT]
    first, second = (a["messages"][-1]["content"] for a in acts)
    assert "历次工具返回的原文：\n（暂无）" in first
    assert "【第 1 轮观察】" in second and "柳月给白狐取名雪团" in second  # 原文可见
    assert "白狐名叫雪团 [demo-1:0002:0000]" in second  # 笔记照旧
    assert out.record["answer"]["citations"] == ["demo-1:0002:0000"]  # 作答仍只依据笔记
    assert sysm.describe()["act_memory"] == "raw" and ACT_RAW in sysm.describe()["prompts"]


def test_default_memory_mode_unchanged():
    sysm, m = _system({
        PLAN: [Say("计划")],
        ACT: [Call.one("search_text", {"query": "白狐 取名"}), Call.one("finish", {"reason": "ok"})],
        REFLECT: [_reflect([{"claim": "白狐名叫雪团", "chunk_id": "demo-1:0002:0000"}])],
        ANSWER: [Say(json.dumps({"answer": "雪团", "rationale": "", "citations": []}))],
    })  # fmt: skip
    sysm.run(_item())
    second = [h for h in m.history if h["prompt_version"] == ACT][1]["messages"][-1]["content"]
    assert "历次工具返回的原文" not in second and "柳月给白狐取名雪团" not in second
    d = sysm.describe()
    assert d["act_memory"] == "notes" and d["graph_scope"] is None and ACT in d["prompts"]


# ---------------------------------------------------------------- 分析函数

REL = {"min_ratio": 0.5, "min_chars": 50}
SPANS = {f"b:0001:{i:04d}": Span("b", i * 100, i * 100 + 100) for i in range(6)}
SPANS.update({f"b:0002:{i:04d}": Span("b", 1000 + i * 100, 1100 + i * 100) for i in range(3)})


def _q(evs):
    return EvalItem(
        id="q", type="multihop", question="问", answer="答", source="generated", split="dev",
        evidence=[Evidence(book_id="b", chapter_idx=1, char_start=s, char_end=e) for s, e in evs],
    )  # fmt: skip


def _tool(rnd, tool, shown=(), refs=(), args=None, meta=None):
    return {"step": "tool", "ok": True, "round": rnd, "tool": tool, "shown": list(shown), "refs": list(refs),
            "args": args or {}, "meta": meta or {}}  # fmt: skip


def _row(label, trace, notes=(), cites=(), tool_counts=None):
    return {"label": label, "citations": list(cites), "evidence_sets": {"notes": list(notes)},
            "agent": {"trace": trace, "tool_counts": tool_counts or {}, "rounds": 3}}  # fmt: skip


def test_question_groups():
    r = lambda *labels: {f"q{i}": {"label": x} for i, x in enumerate(labels)}  # noqa: E731
    g = question_groups(
        r("correct", "wrong", "correct", "wrong"), r("correct", "correct", "wrong", "partial"),
        r("correct", "correct", "correct", "wrong"),
    )  # fmt: skip
    assert g["stable_fail"] == ["q3"] and g["unstable"] == ["q2"]
    assert g["regress"] == ["q2"] and g["regress_s1"] == [] and g["gain"] == ["q1"] and g["gain_s1"] == ["q1"]


def test_evidence_trail_modes():
    item = _q([(0, 100), (200, 300), (400, 500), (1000, 1100)])
    trace = [
        _tool(1, "search_text", shown=["b:0001:0000"]),  # 证据 0：不带限定的检索
        _tool(1, "read_summary", refs=["b:0001:0002"]),  # 证据 1 的编号先由摘要给出
        _tool(2, "read_text", shown=["b:0001:0002"]),  # → 读原文（编号来自摘要）
        _tool(2, "read_text", shown=["b:0001:0004"]),  # 证据 2：同章邻近段
        _tool(3, "search_text", shown=["b:0002:0000"], args={"entity_id": "e"}, meta={"entity_chunks": 9}),
    ]
    row = _row("correct", trace, notes=["b:0001:0000", "b:0001:0002"], cites=["b:0001:0000"])
    t = evidence_trail(row, item, SPANS, REL)
    assert [x["shown"]["mode"] for x in t] == ["search_plain", "read_via_ref", "read_adjacent", "search_entity"]
    assert t[1]["ref"] == {"round": 1, "tool": "read_summary", "chunk_id": "b:0001:0002"}
    assert t[1]["shown"]["ref_from"] == "read_summary" and t[1]["shown"]["round"] == 2
    assert [x["noted"] for x in t] == [True, True, False, False] and [x["cited"] for x in t] == [
        True,
        False,
        False,
        False,
    ]
    assert evidence_state(t) == "all_shown" and evidence_state(t[:2]) == "all_noted"
    # 实体限定太宽被忽略时不算实体限定；带卷 / 章节范围算限定范围
    ignored = _tool(1, "search_text", shown=["b:0001:0000"], args={"entity_id": "e"}, meta={"entity_filter": "x"})
    scoped = _tool(1, "search_text", shown=["b:0001:0000"], args={"volume": "b"})
    one = _q([(0, 100)])
    assert evidence_trail(_row("wrong", [ignored]), one, SPANS, REL)[0]["shown"]["mode"] == "search_plain"
    assert evidence_trail(_row("wrong", [scoped]), one, SPANS, REL)[0]["shown"]["mode"] == "search_scoped"
    assert evidence_state(evidence_trail(_row("wrong", []), one, SPANS, REL)) == "none"


def test_gain_source_kinds():
    item = _q([(0, 100), (1000, 1100)])
    agent = _row("correct", [_tool(1, "search_text", shown=["b:0001:0000"]),
                             _tool(2, "search_text", shown=["b:0002:0000"], args={"chapter_from": "b:0002"})],
                 tool_counts={"search_text": 2, "find_entity": 1})  # fmt: skip
    rag_missing = {"label": "wrong", "hits": ["b:0001:0000", "b:0001:0003", "b:0002:0000"],
                   "evidence_sets": {"context": ["b:0001:0000", "b:0001:0003"]}}  # fmt: skip
    g = gain_source(rag_missing, agent, item, SPANS, REL)
    assert g["kind"] == "retrieval" and g["rag_missed"] == 1 and g["mechanisms"] == ["search_scoped"]
    assert g["last_found_round"] == 2 and g["rag_missed_ranks"] == [3] and g["used_find_entity"]
    rag_full = {"label": "abstain", "hits": [], "evidence_sets": {"context": ["b:0001:0000", "b:0002:0000"]}}
    assert gain_source(rag_full, agent, item, SPANS, REL)["kind"] == "rag_had_evidence"
    blind = _row("correct", [_tool(1, "search_text", shown=["b:0001:0000"])])
    assert gain_source(rag_missing, blind, item, SPANS, REL)["kind"] == "other_evidence"
