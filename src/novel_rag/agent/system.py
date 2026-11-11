"""把 LangGraph Agent 适配成评测用的 QASystem：运行循环 → 统一作答环节 → 不含原文的记录 + 含原文的 trace。"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from typing import Any

from novel_rag.agent.loop import AgentDeps, format_notes, run_agent
from novel_rag.config import Settings
from novel_rag.eval.dataset import EvalItem
from novel_rag.llm import get_chat_model
from novel_rag.llm.base import ChatModel
from novel_rag.prompts import load_prompt
from novel_rag.qa.answer import EvidenceBlock, answer_question, notes_context
from novel_rag.qa.systems import SystemOutput
from novel_rag.tools.base import ToolContext
from novel_rag.tools.registry import Toolbox

TOOL_GROUPS = {
    "search_text": "retrieval", "search_summaries": "retrieval", "read_text": "read", "read_summary": "read",
    "find_entity": "graph", "graph_query": "graph",
}  # fmt: skip
_TEXT_KEYS = ("observation", "output", "text")


def trace_summary(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """去掉原文和模型长输出的 trace 摘要（写进 eval/results）。"""
    out = []
    for st in trace:
        s = {k: v for k, v in st.items() if k not in _TEXT_KEYS}
        if "added" in s:
            s["added"] = [{k: n[k] for k in ("chunk_id", "quote_ok", "seen_text")} for n in s["added"]]
        out.append(s)
    return out


class AgentSystem:
    name = "agent"

    def __init__(
        self,
        toolbox: Toolbox,
        settings: Settings,
        *,
        scope: tuple[str, ...],
        count_tokens: Callable[[str], int],
        agent_chat: ChatModel | None = None,
        reflect_chat: ChatModel | None = None,
        answer_chat: ChatModel | None = None,
    ):
        self.tb, self.s, self.scope, self.count = toolbox, settings, scope, count_tokens
        cfg = settings.agent
        self.agent_chat = agent_chat or get_chat_model("agent_thinking" if cfg.thinking else "agent")
        self.reflect_chat = reflect_chat or get_chat_model("reflect")
        self.answer_chat = answer_chat or get_chat_model("answer")
        titles = toolbox.backends.book_titles
        self.volumes = "、".join(f"{b}《{titles.get(b, b)}》" for b in scope)

    def run(self, item: EvalItem, sample: int = 0) -> SystemOutput:
        t0 = time.perf_counter()
        deps = AgentDeps(
            toolbox=self.tb, agent_chat=self.agent_chat, reflect_chat=self.reflect_chat, cfg=self.s.agent,
            ctx=ToolContext(max_reading_order=item.max_reading_order, book_ids=self.scope), volumes=self.volumes,
            sample=sample,
        )  # fmt: skip
        state = run_agent(item.question, deps)
        t1 = time.perf_counter()
        notes = state.get("notes", [])
        cited = list(dict.fromkeys(n["chunk_id"] for n in notes))
        blocks = []
        for cid in cited:
            try:
                c = self.tb.backends.chunks.by_id(cid)
            except KeyError:
                continue
            blocks.append(EvidenceBlock(cid, self.tb.backends.where(c["book_id"], c["chapter_idx"]), c["text"]))
        ctx = notes_context(format_notes(notes) if notes else "", blocks, self.s.answer.context_token_budget,
                            self.count)  # fmt: skip
        ans = answer_question(
            item.question, ctx, self.answer_chat, allowed_citations=set(cited), abstain=self.s.answer.abstain_text,
            sample=sample,
        )  # fmt: skip
        t2 = time.perf_counter()
        trace = state.get("trace", [])
        lat: dict[str, float] = {"total": round((t2 - t0) * 1000, 1), "loop": round((t1 - t0) * 1000, 1)}
        llm_ms = sum(st.get("latency_ms", 0.0) for st in trace if st["step"] in ("plan", "act", "reflect"))
        lat["llm"] = round(llm_ms + (t2 - t1) * 1000, 1)
        for st in trace:
            if st["step"] == "tool":
                g = TOOL_GROUPS.get(st["tool"], "other")
                lat[g] = round(lat.get(g, 0.0) + st.get("latency_ms", 0.0), 1)
                rr = (st.get("meta") or {}).get("latency", {}).get("rerank")
                if rr:
                    lat["rerank"] = round(lat.get("rerank", 0.0) + rr, 1)
        tool_steps = [st for st in trace if st["step"] == "tool"]
        u = ans.get("usage") or {}
        rec = {
            "answer": ans,
            "evidence_sets": {
                "shown": state.get("shown", []),
                "notes": cited,
                "context": [b.chunk_id for b in ctx.blocks],
            },
            "context_tokens": ctx.tokens,
            "cost": {
                "llm_calls": state.get("llm_calls", 0) + 1,
                "tool_calls": state.get("tool_calls", 0),
                "prompt_tokens": state.get("prompt_tokens", 0) + int(u.get("prompt_tokens", 0)),
                "completion_tokens": state.get("completion_tokens", 0) + int(u.get("completion_tokens", 0)),
                "loop_tokens": state.get("prompt_tokens", 0) + state.get("completion_tokens", 0),
            },
            "latency_ms": lat,
            "termination": state.get("termination"),
            "agent": {
                "rounds": state.get("rounds", 0),
                "tool_counts": dict(Counter(st["tool"] for st in tool_steps)),
                "tool_errors": dict(Counter(st["error_kind"] for st in tool_steps if not st.get("ok"))),
                "n_notes": len(notes),
                "notes": [
                    {
                        "chunk_id": n["chunk_id"],
                        "round": n["round"],
                        "quote_ok": n["quote_ok"],
                        "seen_text": n["seen_text"],
                    }
                    for n in notes
                ],  # fmt: skip
                "note_sources": self._note_sources(trace, notes),
                "invalid_notes": state.get("invalid_notes", 0),
                "dropped_calls": state.get("dropped_calls", 0),
                "context_dropped": ctx.dropped,
                "trace": trace_summary(trace),
            },
        }
        full = {"plan": state.get("plan"), "notes": notes, "status": state.get("status"), "trace": trace,
                "context": ctx.text, "answer": ans, "finish_reason": state.get("finish_reason")}  # fmt: skip
        return SystemOutput(rec, full)

    @staticmethod
    def _note_sources(trace: list[dict[str, Any]], notes: list[dict[str, Any]]) -> dict[str, list[str]]:
        """每个笔记 chunk 最早出现在哪个工具的观察里（工具贡献统计）。"""
        first: dict[str, list[str]] = {}
        for st in trace:
            if st["step"] != "tool":
                continue
            for c in [*st.get("shown", []), *st.get("refs", [])]:
                first.setdefault(c, [])
                if st["tool"] not in first[c]:
                    first[c].append(st["tool"])
        return {n["chunk_id"]: first.get(n["chunk_id"], []) for n in notes}

    def describe(self) -> dict[str, Any]:
        a = self.s.agent
        return {
            "name": self.name,
            "tools": list(self.tb.tools),
            "limits": {"max_rounds": a.max_rounds, "max_tool_calls": a.max_tool_calls, "max_tokens": a.max_tokens,
                       "max_calls_per_round": a.max_calls_per_round, "max_error_rounds": a.max_error_rounds},
            "thinking": a.thinking,
            "models": {"agent": self.agent_chat.model_name, "reflect": self.reflect_chat.model_name,
                       "answer": self.answer_chat.model_name},
            "prompts": [load_prompt(n).version_id for n in ("agent_plan", "agent_act", "agent_reflect", "qa_answer")],
            "retriever": self.tb.backends.retriever.describe() if self.tb.backends.retriever else None,
        }  # fmt: skip
