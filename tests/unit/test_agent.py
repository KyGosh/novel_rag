"""P4 Agent 循环：五个环节、证据笔记是唯一跨轮证据、行动环节看不到原始观察、三种上限、错误处理、引用校验、trace。

全部用脚本化 Fake 模型 + 假后端，完全确定、不调用真实模型和服务。
"""

import json

import pytest

from novel_rag.agent.system import AgentSystem
from novel_rag.config import get_settings
from novel_rag.eval.dataset import EvalItem, Evidence
from novel_rag.llm.fake import Call, Say, ScriptedChatModel
from novel_rag.qa.runner import _run_one
from novel_rag.tools import Toolbox
from tests.unit.tool_fakes import make_backends

PLAN, ACT, REFLECT, ANSWER = "agent_plan.v1", "agent_act.v1", "agent_reflect.v1", "qa_answer.v1"
BOOKS = ("demo-1", "demo-2")


def item(max_ro=4, q="白狐后来被取了什么名字？"):
    return EvalItem(
        id="q1", type="multihop", question=q, answer="雪团", source="generated", split="dev",
        evidence=[Evidence(book_id="demo-1", chapter_idx=2, char_start=0, char_end=10)], max_reading_order=max_ro,
    )  # fmt: skip


def reflect(notes, status="已确认", sufficient=False):
    return Say(json.dumps({"new_notes": notes, "status": status, "sufficient": sufficient}, ensure_ascii=False))


def answer(ans, cites):
    return Say(json.dumps({"answer": ans, "rationale": "依据笔记", "citations": cites}, ensure_ascii=False))


def system(script, *, leak=False, tokens=(100, 20), **limits):
    s = get_settings()
    s = s.model_copy(update={"agent": s.agent.model_copy(update=limits)})
    m = ScriptedChatModel(script, tokens_per_call=tokens)
    tb = Toolbox(make_backends(leak_text=leak), sleep=lambda x: None)
    return AgentSystem(tb, s, scope=BOOKS, count_tokens=len, agent_chat=m, reflect_chat=m, answer_chat=m), m


def test_happy_path_plan_act_execute_reflect_answer():
    sysm, m = system({
        PLAN: [Say("先检索白狐的名字")],
        ACT: [Call.one("search_text", {"query": "白狐 取名"}), Call.one("finish", {"reason": "够了"})],
        REFLECT: [reflect([{"claim": "白狐叫雪团", "chunk_id": "demo-1:0002:0000", "quote": "柳月给白狐取名雪团"},
                           {"claim": "编造", "chunk_id": "demo-2:0001:0001", "quote": ""}])],
        ANSWER: [answer("雪团", ["demo-1:0002:0000", "demo-1:0000:0000"])],
    })  # fmt: skip
    out = sysm.run(item())
    rec = out.record
    assert rec["termination"] == "finish"
    assert rec["answer"]["answer"] == "雪团"
    # 引用校验：只保留笔记引用过的 chunk
    assert rec["answer"]["citations"] == ["demo-1:0002:0000"]
    assert rec["answer"]["invalid_citations"] == ["demo-1:0000:0000"]
    # 笔记引用未出现在本轮观察中的 chunk → 作废
    assert rec["agent"]["n_notes"] == 1 and rec["agent"]["invalid_notes"] == 1
    assert rec["agent"]["notes"][0] == {"chunk_id": "demo-1:0002:0000", "round": 1, "quote_ok": True, "seen_text": True}
    assert rec["evidence_sets"]["notes"] == ["demo-1:0002:0000"]
    assert "demo-1:0002:0000" in rec["evidence_sets"]["shown"]
    assert rec["cost"]["llm_calls"] == 5 and rec["cost"]["tool_calls"] == 1  # 规划 + 2 次行动 + 反思 + 作答
    assert rec["agent"]["note_sources"] == {"demo-1:0002:0000": ["search_text"]}
    steps = [st["step"] for st in out.trace["trace"]]
    assert steps == ["plan", "act", "tool", "reflect", "act"]
    # trace 摘要不含原文；完整 trace 含观察原文
    assert "observation" not in json.dumps(rec["agent"]["trace"], ensure_ascii=False)
    assert "柳月给白狐取名雪团" in out.trace["trace"][2]["observation"]
    # 作答环节的上下文 = 笔记 + 笔记所引原文
    assert "证据笔记" in out.trace["context"] and "柳月给白狐取名雪团" in out.trace["context"]


def test_act_never_sees_raw_observations_only_notes_and_overview():
    sysm, m = system({
        PLAN: [Say("计划")],
        ACT: [Call.one("search_text", {"query": "白狐 取名"}), Call.one("finish", {"reason": "ok"})],
        REFLECT: [reflect([{"claim": "白狐名叫雪团", "chunk_id": "demo-1:0002:0000"}], status="已确认名字")],
        ANSWER: [answer("雪团", ["demo-1:0002:0000"])],
    })  # fmt: skip
    sysm.run(item())
    acts = [h for h in m.history if h["prompt_version"] == ACT]
    second = acts[1]["messages"][-1]["content"]
    assert "白狐名叫雪团 [demo-1:0002:0000]" in second  # 笔记可见
    assert "已确认名字" in second  # 证据现状可见
    assert "demo-1:0002:0000（北岭记 · 柳月）" in second  # 行动记录的结果概况只有编号与位置
    for c in make_backends().chunks.chunks:
        assert c["text"] not in second  # 原文不可见
    assert "finish" in acts[0]["tools"] and "search_text" in acts[0]["tools"]


def test_round_limit_forces_answer():
    search = [Call.one("search_text", {"query": f"白狐 {i}"}) for i in range(5)]
    sysm, _ = system({PLAN: [Say("p")], ACT: search, REFLECT: [reflect([]) for _ in range(5)],
                      ANSWER: [answer("原文未提及", [])]}, max_rounds=2)  # fmt: skip
    rec = sysm.run(item()).record
    assert rec["termination"] == "limit_rounds" and rec["agent"]["rounds"] == 2
    assert rec["answer"]["abstained"]


def test_tool_call_limit_drops_extra_calls():
    many = Call(tuple(("search_text", {"query": f"q{i}"}) for i in range(4)))
    sysm, _ = system({PLAN: [Say("p")], ACT: [many, many], REFLECT: [reflect([]), reflect([])],
                      ANSWER: [answer("原文未提及", [])]}, max_tool_calls=5, max_calls_per_round=4)  # fmt: skip
    rec = sysm.run(item()).record
    assert rec["cost"]["tool_calls"] == 5 and rec["agent"]["dropped_calls"] == 3
    assert rec["termination"] == "limit_tool_calls"


def test_token_limit():
    sysm, _ = system({PLAN: [Say("p")], ACT: [Call.one("search_text", {"query": "白狐"})], REFLECT: [reflect([])],
                      ANSWER: [answer("原文未提及", [])]}, tokens=(3000, 100), max_tokens=8000)  # fmt: skip
    rec = sysm.run(item()).record
    assert rec["termination"] == "limit_tokens" and rec["cost"]["loop_tokens"] >= 8000


def test_bad_tool_calls_become_observations_then_error_streak():
    bad = [
        Call.one("no_such_tool", {}),
        Call.one("search_text", '{"query": 白'),
        Call.one("read_text", {"chunk_id": "x"}),
    ]
    sysm, m = system({PLAN: [Say("p")], ACT: bad, ANSWER: [answer("原文未提及", [])]}, max_error_rounds=3)
    out = sysm.run(item())
    rec = out.record
    assert rec["termination"] == "error_streak" and rec["agent"]["rounds"] == 3
    assert rec["agent"]["tool_errors"] == {"unknown_tool": 1, "invalid_args": 2}
    # 错误作为观察交还给模型：下一轮行动记录里能看到错误说明
    acts = [h for h in m.history if h["prompt_version"] == ACT]
    assert "错误（unknown_tool）" in acts[1]["messages"][-1]["content"]
    assert "错误（invalid_args）" in acts[2]["messages"][-1]["content"]
    # 只有错误的轮次不调用反思
    assert not [h for h in m.history if h["prompt_version"] == REFLECT]


def test_no_tool_and_no_finish_is_treated_as_finish():
    sysm, _ = system({PLAN: [Say("p")], ACT: [Say("我觉得够了")], ANSWER: [answer("原文未提及", [])]})
    assert sysm.run(item()).record["termination"] == "finish_implicit"


def test_tool_failure_is_retried_then_returned_to_model():
    act = [Call.one("search_text", {"query": "白狐"}), Call.one("finish", {"reason": "x"})]
    sysm, m = system({PLAN: [Say("p")], ACT: act,
                      ANSWER: [answer("原文未提及", [])]})  # fmt: skip

    def down(*a, **k):
        raise TimeoutError("es timeout")

    sysm.tb.backends.retriever.search = down
    rec = sysm.run(item()).record
    tool_step = next(st for st in rec["agent"]["trace"] if st["step"] == "tool")
    assert tool_step["error_kind"] == "transient" and tool_step["attempts"] == sysm.tb.max_retries + 1
    assert rec["termination"] == "finish"
    acts = [h for h in m.history if h["prompt_version"] == ACT]
    assert "错误（transient）" in acts[1]["messages"][-1]["content"]


def test_spoiler_leak_fails_the_question_in_runner():
    sysm, _ = system({PLAN: [Say("p")], ACT: [Call.one("search_text", {"query": "真相 石峰 私生子", "top_k": 8})]},
                     leak=True)  # fmt: skip
    rec, _ = _run_one(sysm, item(max_ro=1), 0, None)
    assert rec["termination"] == "spoiler_violation" and "SpoilerLeak" not in rec["error"]


def test_finish_with_other_calls_executes_them_then_ends():
    both = Call((("search_text", {"query": "白狐 取名"}), ("finish", {"reason": "最后查一次"})))
    sysm, _ = system({PLAN: [Say("p")], ACT: [both],
                      REFLECT: [reflect([{"claim": "白狐叫雪团", "chunk_id": "demo-1:0002:0000", "quote": "取名"}])],
                      ANSWER: [answer("雪团", ["demo-1:0002:0000"])]})  # fmt: skip
    rec = sysm.run(item()).record
    assert rec["termination"] == "finish" and rec["agent"]["n_notes"] == 1 and rec["agent"]["rounds"] == 1


@pytest.mark.parametrize("max_ro", [0, 1, 2])
def test_agent_observations_respect_reading_progress(max_ro):
    sysm, _ = system({PLAN: [Say("p")],
                      ACT: [Call((("search_text", {"query": "白狐 雪团 石峰 真相", "top_k": 8}),
                                  ("read_text", {"chunk_id": "demo-1:0000:0000", "after": 3}))),
                            Call.one("finish", {"reason": "x"})],
                      REFLECT: [reflect([])], ANSWER: [answer("原文未提及", [])]})  # fmt: skip
    out = sysm.run(item(max_ro=max_ro))
    chunks = sysm.tb.backends.chunks
    for cid in out.record["evidence_sets"]["shown"]:
        assert chunks.by_id(cid)["reading_order"] <= max_ro


def test_token_limit_is_predictive_from_last_round_cost():
    # 每次调用 1000 token：规划 1000；每轮行动 + 反思 2000。上限 6000：第 2 轮后累计 5000，再一轮会到 7000 → 停止
    acts = [Call.one("search_text", {"query": f"白狐 {i}"}) for i in range(5)]
    sysm, _ = system({PLAN: [Say("p")], ACT: acts, REFLECT: [reflect([]) for _ in range(5)],
                      ANSWER: [answer("原文未提及", [])]}, tokens=(900, 100), max_tokens=6000)  # fmt: skip
    rec = sysm.run(item()).record
    assert rec["termination"] == "limit_tokens" and rec["agent"]["rounds"] == 2
    assert rec["cost"]["loop_tokens"] <= 6000
