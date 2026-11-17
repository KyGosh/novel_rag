"""按需升级（P6 B2）：先走固定 RAG，判断证据是否充分；充分就直接返回，不充分才启动 Agent。

路由信号（P6 方案第三节）：
- rules：不花调用的两条规则——弃答必升级；引用为空或有无效引用必升级。另外两种信号都先过这两条规则。
- selfcheck：固定 RAG 的作答环节换成 qa_answer_selfcheck（多输出一项 all_supported：每个子问题是否都有所引原文
  的直接支持），all_supported 为 false 时升级。不增加调用，但第一阶段的作答提示词变了。
- verify：第一阶段与固定 RAG 完全相同；另加一次 LLM 调用（route_check），输入问题、答案和所引原文，逐个子问题
  核对是否有支持，sufficient 为 false 时升级。

升级时 Agent 从头开始，不复用第一阶段的检索结果，所以 Agent 部分的行为与单独运行 Agent 完全相同（评估时可以从
缓存回放已有的 Agent 运行）。成本和延迟 = 第一阶段 + 路由 + （升级时）Agent。
dry=True 时只做第一阶段和路由、从不升级（可行性验证用）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from novel_rag.eval.dataset import EvalItem
from novel_rag.llm.base import ChatModel, StructuredOutputError
from novel_rag.prompts import load_prompt
from novel_rag.qa.answer import AnswerOutput
from novel_rag.qa.systems import FixedRAGSystem, QASystem, SystemOutput, emit

RouteSignal = Literal["rules", "selfcheck", "verify"]
SIGNALS: tuple[str, ...] = ("rules", "selfcheck", "verify")


class SelfCheckAnswer(AnswerOutput):
    all_supported: bool = Field(False, description="每个子问题是否都有所引原文的直接支持")


class PartCheck(BaseModel):
    part: str = ""
    supported: bool = False


class RouteCheckOut(BaseModel):
    parts: list[PartCheck] = Field(default_factory=list)
    sufficient: bool = False


def rule_reason(ans: dict[str, Any]) -> str | None:
    """不花调用的规则：弃答 → abstain；没有有效引用 → no_citation；有无效引用 → invalid_citation。"""
    if ans.get("abstained"):
        return "abstain"
    if not ans.get("citations"):
        return "no_citation"
    if ans.get("invalid_citations"):
        return "invalid_citation"
    return None


def check_support(
    question: str, ans: dict[str, Any], evidence: list[tuple[str, str, str]], chat: ChatModel, sample: int = 0
) -> dict[str, Any]:
    """独立核对（信号 verify）。evidence = [(chunk_id, 位置, 原文)]。返回不含原文的结果；输出不合格按"不充分"处理。"""
    p = load_prompt("route_check")
    ev = "\n\n".join(f"[{cid}] {where}\n{text}" for cid, where, text in evidence) or "（无）"
    msgs = p.render(question=question, answer=ans.get("answer", ""), rationale=ans.get("rationale", ""), evidence=ev)
    t0 = time.perf_counter()
    try:
        res = chat.chat(msgs, schema=RouteCheckOut, prompt_version=p.version_id, sample=sample)
    except StructuredOutputError as e:
        return {"sufficient": False, "parts": [], "error": str(e)[:200], "usage": {}, "cached": False,
                "prompt_version": p.version_id, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}  # fmt: skip
    out: RouteCheckOut = res.parsed  # type: ignore[assignment]
    return {
        "sufficient": out.sufficient,
        "parts": [{"part": x.part[:40], "supported": x.supported} for x in out.parts],
        "usage": {"prompt_tokens": res.usage.prompt_tokens, "completion_tokens": res.usage.completion_tokens},
        "cached": res.cached, "prompt_version": p.version_id,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }  # fmt: skip


def _add_cost(*costs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in costs:
        for k, v in c.items():
            if isinstance(v, int | float):
                out[k] = out.get(k, 0) + v
    return out


class CascadeSystem:
    """按需升级：固定 RAG → 路由 →（不充分时）Agent。agent 为 None 或 dry=True 时只做第一阶段和路由。"""

    def __init__(
        self,
        rag: FixedRAGSystem,
        agent: QASystem | None,
        *,
        signal: RouteSignal,
        check_chat: ChatModel | None = None,
        dry: bool = False,
    ):
        if signal == "verify" and check_chat is None:
            raise ValueError("verify 信号需要核对用的模型")
        if signal == "selfcheck" and rag.answer_schema is not SelfCheckAnswer:
            raise ValueError("selfcheck 信号要求第一阶段用 qa_answer_selfcheck 作答")
        self.rag, self.agent, self.signal, self.check_chat = rag, agent, signal, check_chat
        self.dry = dry or agent is None
        self.name = f"cascade_{signal}" + ("_dry" if self.dry else "")

    def route(self, item: EvalItem, ans: dict[str, Any], sample: int) -> dict[str, Any]:
        reason = rule_reason(ans)
        route: dict[str, Any] = {"signal": self.signal, "rule": reason}
        if self.signal == "selfcheck":
            sup = (ans.get("extra") or {}).get("all_supported")
            route["self_supported"] = sup
            if reason is None and not sup:
                reason = "self_unsupported"
        elif self.signal == "verify" and reason is None:
            b = self.rag.b
            ev = []
            for cid in ans.get("citations", []):
                c = b.chunks.by_id(cid)
                ev.append((cid, b.where(c["book_id"], c["chapter_idx"]), c["text"]))
            v = check_support(item.question, ans, ev, self.check_chat, sample)  # type: ignore[arg-type]
            route["verify"] = v
            if not v["sufficient"]:
                reason = "verify_insufficient"
        route["reason"] = reason
        route["escalate"] = reason is not None
        return route

    def run(
        self, item: EvalItem, sample: int = 0, on_event: Callable[[dict[str, Any]], None] | None = None
    ) -> SystemOutput:
        t0 = time.perf_counter()
        out1 = self.rag.run(item, sample, on_event=on_event)
        rec1 = out1.record
        t1 = time.perf_counter()
        route = self.route(item, rec1["answer"], sample)
        t2 = time.perf_counter()
        v = route.get("verify") or {}
        route_cost = {"llm_calls": 1, **v.get("usage", {})} if v else {}
        emit(on_event, {"type": "route", **{k: route[k] for k in ("signal", "reason", "escalate")},
                        "parts": v.get("parts"), "latency_ms": round((t2 - t1) * 1000, 1)})  # fmt: skip
        a1 = rec1["answer"]
        stage1 = {"answer": a1["answer"], "abstained": a1["abstained"], "citations": a1["citations"],
                  "cost": rec1["cost"], "latency_ms": rec1["latency_ms"]}  # fmt: skip
        if not route["escalate"] or self.dry:
            rec = {**rec1, "route": route, "cost": _add_cost(rec1["cost"], route_cost)}
            rec["latency_ms"] = {**rec1["latency_ms"], "route": round((t2 - t1) * 1000, 1),
                                 "total": round((t2 - t0) * 1000, 1)}  # fmt: skip
            return SystemOutput(rec, {**out1.trace, "route": route})
        run_agent: Any = self.agent.run  # type: ignore[union-attr]
        out2 = run_agent(item, sample, on_event=on_event) if on_event is not None else run_agent(item, sample)
        t3 = time.perf_counter()
        rec2 = out2.record
        rec = {**rec2, "hits": rec1.get("hits"), "route": route, "stage1": stage1,
               "cost": _add_cost(rec1["cost"], route_cost, rec2.get("cost", {}))}  # fmt: skip
        rec["latency_ms"] = {
            **{f"agent.{k}": v for k, v in rec2.get("latency_ms", {}).items()},
            "stage1": rec1["latency_ms"].get("total"), "route": round((t2 - t1) * 1000, 1),
            "total": round((t3 - t0) * 1000, 1),
        }  # fmt: skip
        return SystemOutput(rec, {"stage1": out1.trace, "route": route, **out2.trace})

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "signal": self.signal, "dry": self.dry, "stage1": self.rag.describe()}
        if self.check_chat is not None:
            out["check_model"] = self.check_chat.model_name
            out["check_prompt"] = load_prompt("route_check").version_id
        if self.agent is not None and not self.dry:
            out["agent"] = self.agent.describe()
        return out
