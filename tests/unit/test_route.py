"""P6 按需升级：规则、三种路由信号、升级时 Agent 从头开始、成本合计、dry 模式、默认作答路径不变。

全部用脚本化 Fake 模型 + 假后端，不调用真实模型和服务。
"""

import json

import pytest

from novel_rag.agent.system import AgentSystem
from novel_rag.config import get_settings
from novel_rag.llm.fake import Call, Say, ScriptedChatModel
from novel_rag.prompts import load_prompt
from novel_rag.qa.answer import AnswerOutput
from novel_rag.qa.route import CascadeSystem, SelfCheckAnswer, rule_reason
from novel_rag.qa.systems import FixedRAGSystem
from novel_rag.tools import Toolbox
from tests.unit.test_agent import ACT, ANSWER, BOOKS, PLAN, REFLECT, item, reflect
from tests.unit.tool_fakes import make_backends

SELF, CHECK = "qa_answer_selfcheck.v1", "route_check.v1"
GOOD = "demo-1:0002:0000"  # 白狐取名那段（fake 检索能检到）


def say(**kw):
    return Say(json.dumps(kw, ensure_ascii=False))


def build(script, signal="verify", dry=False, with_agent=True):
    s = get_settings()
    m = ScriptedChatModel(script)
    b = make_backends()
    selfcheck = signal == "selfcheck"
    rag = FixedRAGSystem(
        "rag", b, m, len, scope=BOOKS, top_k=4, pool=8, budget=4000, abstain="原文未提及",
        answer_prompt=load_prompt("qa_answer_selfcheck") if selfcheck else None,
        answer_schema=SelfCheckAnswer if selfcheck else AnswerOutput,
    )  # fmt: skip
    agent = AgentSystem(Toolbox(b, sleep=lambda x: None), s, scope=BOOKS, count_tokens=len, agent_chat=m,
                        reflect_chat=m, answer_chat=m) if with_agent else None  # fmt: skip
    return CascadeSystem(rag, agent, signal=signal, check_chat=m if signal == "verify" else None, dry=dry), m


def agent_script():  # 脚本会被消耗，每个测试取一份新的
    return {
        PLAN: [Say("p")],
        ACT: [Call.one("search_text", {"query": "白狐 取名"}), Call.one("finish", {"reason": "够了"})],
        REFLECT: [reflect([{"claim": "白狐叫雪团", "chunk_id": GOOD, "quote": "取名"}])],
    }


@pytest.mark.parametrize(
    ("ans", "reason"),
    [
        ({"abstained": True, "citations": [], "invalid_citations": []}, "abstain"),
        ({"abstained": False, "citations": [], "invalid_citations": []}, "no_citation"),
        ({"abstained": False, "citations": ["a"], "invalid_citations": ["b"]}, "invalid_citation"),
        ({"abstained": False, "citations": ["a"], "invalid_citations": []}, None),
    ],
)
def test_rule_reason(ans, reason):
    assert rule_reason(ans) == reason


def test_verify_sufficient_returns_stage1_without_agent():
    sysm, m = build({ANSWER: [say(answer="雪团", rationale="原文", citations=[GOOD])],
                     CHECK: [say(parts=[{"part": "名字", "supported": True}], sufficient=True)]})  # fmt: skip
    rec = sysm.run(item()).record
    assert rec["answer"]["answer"] == "雪团" and not rec["route"]["escalate"]
    assert rec["cost"]["llm_calls"] == 2 and rec["cost"]["tool_calls"] == 1  # 作答 + 核对
    assert "agent" not in rec and rec["termination"] == "answered"
    # 核对环节看到的是所引原文，不含原文的结果才写进记录
    check_msg = next(h for h in m.history if h["prompt_version"] == CHECK)["messages"][-1]["content"]
    assert "柳月给白狐取名雪团" in check_msg and "雪团" in check_msg
    assert "柳月" not in json.dumps(rec["route"], ensure_ascii=False)


def test_verify_insufficient_escalates_to_fresh_agent_and_sums_cost():
    script = {ANSWER: [say(answer="石峰", rationale="猜的", citations=[GOOD]), say(answer="雪团", citations=[GOOD])],
              CHECK: [say(parts=[{"part": "名字", "supported": False}], sufficient=False)],
              **agent_script()}  # fmt: skip
    sysm, m = build(script)
    rec = sysm.run(item()).record
    assert rec["route"]["escalate"] and rec["route"]["reason"] == "verify_insufficient"
    assert rec["answer"]["answer"] == "雪团" and rec["stage1"]["answer"] == "石峰"
    # 第一阶段 1 + 核对 1 + Agent（规划、2 次行动、反思、作答）5
    assert rec["cost"]["llm_calls"] == 7 and rec["cost"]["tool_calls"] == 1 + 1
    assert rec["agent"]["rounds"] == 1 and rec["hits"]  # 保留第一阶段的检索结果（检索指标口径不变）
    assert {"stage1", "route", "total", "agent.total"} <= set(rec["latency_ms"])
    # Agent 从头开始：规划环节看不到第一阶段的答案
    plan_msg = next(h for h in m.history if h["prompt_version"] == PLAN)["messages"][-1]["content"]
    assert "石峰" not in plan_msg


def test_rules_escalate_without_check_call():
    script = {ANSWER: [say(answer="原文未提及", citations=[]), say(answer="雪团", citations=[GOOD])], **agent_script()}
    sysm, m = build(script)
    rec = sysm.run(item()).record
    assert rec["route"]["reason"] == "abstain" and "verify" not in rec["route"]
    assert not [h for h in m.history if h["prompt_version"] == CHECK]


def test_selfcheck_signal_uses_selfcheck_prompt():
    sysm, m = build({SELF: [say(answer="雪团", citations=[GOOD], all_supported=False)], **agent_script(),
                     ANSWER: [say(answer="雪团", citations=[GOOD])]}, signal="selfcheck")  # fmt: skip
    rec = sysm.run(item()).record
    assert rec["route"]["reason"] == "self_unsupported" and rec["route"]["self_supported"] is False
    assert rec["stage1"]["cost"]["llm_calls"] == 1
    sysm2, _ = build({SELF: [say(answer="雪团", citations=[GOOD], all_supported=True)]}, signal="selfcheck")
    rec2 = sysm2.run(item()).record
    assert not rec2["route"]["escalate"] and rec2["answer"]["extra"] == {"all_supported": True}
    assert rec2["cost"]["llm_calls"] == 1


def test_dry_mode_never_escalates():
    sysm, _ = build({ANSWER: [say(answer="原文未提及", citations=[])]}, dry=True, with_agent=False)
    rec = sysm.run(item()).record
    assert rec["route"]["escalate"] and rec["answer"]["abstained"] and "agent" not in rec
    assert sysm.name == "cascade_verify_dry"


def test_default_answer_path_has_no_extra_key():
    sysm, _ = build({ANSWER: [say(answer="雪团", citations=[GOOD])], CHECK: [say(sufficient=True)]})
    assert "extra" not in sysm.run(item()).record["answer"]


def test_events_for_cascade():
    events = []
    script = {ANSWER: [say(answer="石峰", citations=[GOOD]), say(answer="雪团", citations=[GOOD])],
              CHECK: [say(sufficient=False)], **agent_script()}  # fmt: skip
    sysm, _ = build(script)
    sysm.run(item(), on_event=events.append)
    types = [e["type"] for e in events]
    assert types[:3] == ["retrieve", "answer", "route"] and types[3] == "plan" and types[-1] == "answer"
    assert events[2]["escalate"] and events[0]["chunks"][0]["where"].startswith("北岭记")
