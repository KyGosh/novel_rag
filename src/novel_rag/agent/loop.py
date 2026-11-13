"""Agent 循环（LangGraph）：规划 → 行动 → 执行 → 反思 →（行动 … | 作答）。

- 规划（第一轮前）：模型写一份简短计划，之后每轮可见。
- 行动：模型看到问题、计划、证据笔记、证据现状和行动记录（**看不到原始观察**），用 function calling 决定调用哪些
  工具，或调用 finish 结束。
- 执行：经 Toolbox 调用工具（参数校验、重试、出错返回、防剧透断言）；超出上限的调用不执行并告知模型。
- 反思：单独一次 LLM 调用，输入问题、当前笔记和本轮原始观察，输出新增证据笔记（结论 + chunk_id + 短引文）
  和一句证据现状。笔记引用的 chunk 必须出现在本轮观察中，否则作废。原始观察用完即丢：证据笔记是唯一跨轮
  保留的证据。
- 作答：统一的作答环节（qa/answer.py），上下文 = 证据笔记 + 笔记所引 chunk 的原文（4000 token 预算）。

上限（config.agent）：轮数、工具调用次数、token；触及上限时用现有笔记强制作答，终止原因记为 limit_*。
模型不调用工具也不调用 finish → 按 finish 处理（finish_implicit）；连续若干轮只产生错误 → 强制作答（error_streak）。
LangGraph 只在 agent/ 中 import；节点是闭包，外部依赖（模型、工具箱）注入。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from novel_rag.config import AgentConfig
from novel_rag.llm.base import ChatModel, ChatResult, StructuredOutputError, ToolCall
from novel_rag.prompts import Prompt, load_prompt
from novel_rag.tools.base import ToolContext, ToolResult
from novel_rag.tools.registry import Toolbox

FINISH = "finish"
FINISH_SCHEMA = {
    "name": FINISH,
    "description": "证据笔记已经足以回答问题，或已确认原文中找不到答案时调用，结束检索、进入作答。",
    "parameters": {
        "type": "object",
        "properties": {"reason": {"type": "string", "description": "为什么可以结束"}},
        "required": ["reason"],
    },
}


class NoteIn(BaseModel):
    claim: str
    chunk_id: str
    quote: str = ""


class ReflectOut(BaseModel):
    new_notes: list[NoteIn] = Field(default_factory=list)
    status: str = ""
    sufficient: bool = False


class AgentState(TypedDict, total=False):
    question: str
    plan: str
    notes: list[dict[str, Any]]  # {claim, chunk_id, quote, quote_ok, seen_text, round}
    status: str
    actions: list[dict[str, Any]]  # 行动记录：{round, tool, args, overview, ok}
    observations: list[dict[str, Any]]  # 本轮原始观察（只交给反思，用完即丢）
    history: list[str]  # 历次观察的原文（只在 act_memory=raw 的消融里保留并交给行动环节）
    pending_calls: list[ToolCall]
    finish_requested: bool
    finish_reason: str
    rounds: int
    tool_calls: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    error_rounds: int
    round_start_tokens: int  # 本轮开始时的累计 token
    last_round_tokens: int  # 上一轮（行动 + 反思）的花费，用于预估 token 上限
    termination: str | None
    shown: list[str]  # 所有展示过原文的 chunk（过程覆盖率）
    seen_refs: list[str]  # 只给出编号的 chunk
    invalid_notes: int
    dropped_calls: int
    trace: list[dict[str, Any]]


@dataclass
class AgentDeps:
    toolbox: Toolbox
    agent_chat: ChatModel  # 规划与行动
    reflect_chat: ChatModel
    cfg: AgentConfig
    ctx: ToolContext
    volumes: str  # "卷ID《卷名》、……"
    sample: int = 0
    prompts: dict[str, Prompt] = field(default_factory=dict)

    def prompt(self, name: str) -> Prompt:
        if name not in self.prompts:
            self.prompts[name] = load_prompt(name)
        return self.prompts[name]


def _tok(state: AgentState) -> int:
    return state.get("prompt_tokens", 0) + state.get("completion_tokens", 0)


def _usage_update(state: AgentState, res: ChatResult) -> dict[str, int]:
    return {
        "llm_calls": state.get("llm_calls", 0) + 1,
        "prompt_tokens": state.get("prompt_tokens", 0) + res.usage.prompt_tokens,
        "completion_tokens": state.get("completion_tokens", 0) + res.usage.completion_tokens,
    }


def format_notes(notes: list[dict[str, Any]]) -> str:
    if not notes:
        return "（暂无）"
    lines = []
    for i, n in enumerate(notes, 1):
        q = f"（引：“{n['quote']}”）" if n.get("quote") else ""
        lines.append(f"{i}. {n['claim']}{q} [{n['chunk_id']}]")
    return "\n".join(lines)


def _args_text(args: dict[str, Any] | None) -> str:
    if args is None:
        return "（参数无效）"
    return ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())


def overview(r: ToolResult, where: Callable[[str], str]) -> str:
    """行动记录里的结果概况：只有编号、位置和计数，不含原文。"""
    if not r.ok:
        return f"错误（{r.error_kind}）：{r.error}"
    if r.tool == "search_text":
        parts = [f"{c}（{where(c)}）" for c in r.shown_chunk_ids]
        more = f"；另有 {len(r.ref_chunk_ids)} 段未展示" if r.ref_chunk_ids else ""
        return f"{len(r.shown_chunk_ids)} 段：" + "、".join(parts) + more
    if r.tool == "read_text":
        ids = r.shown_chunk_ids
        return f"读了 {ids[0]} … {ids[-1]}（{where(ids[0])}）" if len(ids) > 1 else f"读了 {ids[0]}（{where(ids[0])}）"
    if r.tool == "search_summaries":
        return "章节：" + "、".join(r.meta.get("chapters", []))
    if r.tool == "read_summary":
        return f"摘要与事件，证据编号 {len(r.ref_chunk_ids)} 个"
    if r.tool == "find_entity":
        return "候选：" + "；".join(r.meta.get("names", r.meta.get("candidates", [])))
    if r.tool == "graph_query":
        return f"{r.meta.get('op')}：{r.meta.get('rows')} 条结果，证据编号 {len(r.ref_chunk_ids)} 个"
    return f"成功，{len(r.text)} 字"


def format_actions(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "（暂无）"
    return "\n".join(f"第 {a['round']} 轮 {a['tool']}({a['args']}) → {a['overview']}" for a in actions)


def format_history(history: list[str], max_chars: int) -> str:
    """raw 模式下交给行动环节的历次观察：从最近一轮往前保留，超出总长上限的更早内容截断或省略。"""
    if not history:
        return "（暂无）"
    kept: list[str] = []
    left = max_chars
    for k in range(len(history), 0, -1):
        block = f"【第 {k} 轮观察】\n{history[k - 1]}"
        if len(block) > left:
            if left > 200:  # 还放得下一部分：截断这一轮，更早的省略
                kept.append(block[:left] + "……（截断）")
                if k > 1:
                    kept.append(f"（第 {k - 1} 轮及更早的观察因长度上限省略）")
            else:
                kept.append(f"（第 {k} 轮及更早的观察因长度上限省略）")
            break
        kept.append(block)
        left -= len(block)
    return "\n\n".join(reversed(kept))


_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub("", s).replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")


def build_graph(deps: AgentDeps):  # noqa: ANN201 —— 返回编译后的 LangGraph 图
    cfg = deps.cfg
    tb = deps.toolbox
    where = tb.backends.where_chunk
    tool_schemas = [*tb.schemas(), FINISH_SCHEMA]
    tools_brief = "\n".join(f"- {t['name']}：{t['description']}" for t in tb.schemas())

    def plan_node(state: AgentState) -> dict[str, Any]:
        p = deps.prompt("agent_plan")
        msgs = p.render(volumes=deps.volumes, tools=tools_brief, question=state["question"])
        t0 = time.perf_counter()
        res = deps.agent_chat.chat(msgs, prompt_version=p.version_id, sample=deps.sample)
        plan = (res.text or "").strip()[:600] or "（无）"
        step = {"step": "plan", "prompt": p.version_id, "tokens": res.usage.total_tokens, "cached": res.cached,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "output": plan}  # fmt: skip
        return {"plan": plan, **_usage_update(state, res), "trace": [*state.get("trace", []), step]}

    def act_node(state: AgentState) -> dict[str, Any]:
        rounds, calls_used = state.get("rounds", 0), state.get("tool_calls", 0)
        # 上限检查（执行到这里说明上一轮已经反思完毕）
        reason = None
        if rounds >= cfg.max_rounds:
            reason = "limit_rounds"
        elif calls_used >= cfg.max_tool_calls:
            reason = "limit_tool_calls"
        elif _tok(state) + state.get("last_round_tokens", 0) > cfg.max_tokens:
            # 预估：再进行一轮（按上一轮的花费估计）就会超过上限时不再开始新的一轮，避免超出一整轮的量
            reason = "limit_tokens"
        elif state.get("error_rounds", 0) >= cfg.max_error_rounds:
            reason = "error_streak"
        if reason:
            return {"termination": reason, "pending_calls": []}
        raw = cfg.act_memory == "raw"
        p = deps.prompt("agent_act_raw" if raw else "agent_act")
        values = dict(
            volumes=deps.volumes, max_calls=str(cfg.max_calls_per_round), question=state["question"],
            plan=state.get("plan", ""), notes=format_notes(state.get("notes", [])),
            status=state.get("status") or "（尚未开始检索）", actions=format_actions(state.get("actions", [])),
            rounds_left=str(cfg.max_rounds - rounds), calls_left=str(cfg.max_tool_calls - calls_used),
        )  # fmt: skip
        if raw:  # 消融：行动环节另外看到历次工具返回的原文（按总长上限截断）
            values["observations"] = format_history(state.get("history", []), cfg.raw_history_max_chars)
        msgs = p.render(**values)
        t0 = time.perf_counter()
        res = deps.agent_chat.chat(msgs, prompt_version=p.version_id, tools=tool_schemas, sample=deps.sample)
        calls = list(res.tool_calls)
        step = {
            "step": "act", "round": rounds + 1, "prompt": p.version_id, "tokens": res.usage.total_tokens,
            "cached": res.cached, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "calls": [c.to_dict() for c in calls], "text": res.text[:500],
        }  # fmt: skip
        upd: dict[str, Any] = {**_usage_update(state, res), "trace": [*state.get("trace", []), step]}
        finish = [c for c in calls if c.name == FINISH]
        others = [c for c in calls if c.name != FINISH]
        if not calls:  # 既不调用工具也不调用 finish：按 finish 处理
            return {**upd, "termination": "finish_implicit", "pending_calls": [], "finish_reason": res.text[:200]}
        if finish and not others:
            reason = (finish[0].args or {}).get("reason", "") if finish[0].args else ""
            return {**upd, "termination": "finish", "pending_calls": [], "finish_reason": str(reason)[:200]}
        upd.update({"pending_calls": others, "finish_requested": bool(finish), "rounds": rounds + 1,
                    "round_start_tokens": _tok(state)})  # fmt: skip
        if finish:
            upd["finish_reason"] = str((finish[0].args or {}).get("reason", ""))[:200]
        return upd

    def execute_node(state: AgentState) -> dict[str, Any]:
        rnd = state["rounds"]
        used = state.get("tool_calls", 0)
        budget = min(cfg.max_calls_per_round, cfg.max_tool_calls - used)
        observations, actions, trace = [], list(state.get("actions", [])), list(state.get("trace", []))
        shown, refs = list(state.get("shown", [])), list(state.get("seen_refs", []))
        dropped = state.get("dropped_calls", 0)
        n_ok = n_exec = 0
        for i, call in enumerate(state.get("pending_calls", [])):
            if i >= budget:
                dropped += 1
                actions.append({"round": rnd, "tool": call.name, "args": _args_text(call.args), "ok": False,
                                "overview": "未执行：超出本轮或本题的工具调用上限"})  # fmt: skip
                continue
            r = tb.call(call.name, call.args, deps.ctx, args_error=call.args_error)  # SpoilerLeak 向外抛
            n_exec += 1
            n_ok += int(r.ok)
            shown += [c for c in r.shown_chunk_ids if c not in shown]
            refs += [c for c in r.ref_chunk_ids if c not in refs]
            ov = overview(r, where)
            actions.append({"round": rnd, "tool": call.name, "args": _args_text(call.args), "ok": r.ok, "overview": ov})
            observations.append({"tool": call.name, "args": _args_text(call.args), "result": r})
            trace.append({"step": "tool", "round": rnd, "tool": call.name, "args": call.args,
                          "args_error": call.args_error, **r.summary(), "observation": r.text})  # fmt: skip
        error_rounds = 0 if n_ok else state.get("error_rounds", 0) + 1
        return {
            "observations": observations, "actions": actions, "trace": trace, "tool_calls": used + n_exec,
            "shown": shown, "seen_refs": refs, "dropped_calls": dropped, "error_rounds": error_rounds,
            "pending_calls": [],
        }  # fmt: skip

    def reflect_node(state: AgentState) -> dict[str, Any]:
        obs = [o for o in state.get("observations", []) if o["result"].ok]
        notes = list(state.get("notes", []))
        if not obs:  # 本轮只有错误：不调用反思，直接把错误告诉下一轮
            errs = "；".join(f"{o['tool']}：{o['result'].error}" for o in state.get("observations", []))
            spent = _tok(state) - state.get("round_start_tokens", 0)
            return {"status": f"上一轮工具调用都出错了：{errs[:200]}", "observations": [], "last_round_tokens": spent}
        allowed: dict[str, bool] = {}  # chunk_id → 是否展示了原文
        for o in obs:
            for c in o["result"].shown_chunk_ids:
                allowed[c] = True
            for c in o["result"].ref_chunk_ids:
                allowed.setdefault(c, False)
        share = max(500, cfg.observation_max_chars // len(obs))
        blocks = []
        for k, o in enumerate(obs, 1):
            text = o["result"].text
            if len(text) > share:
                text = text[:share] + "……（截断）"
            blocks.append(f"[观察 {k}] {o['tool']}({o['args']})\n{text}")
        p = deps.prompt("agent_reflect")
        msgs = p.render(question=state["question"], notes=format_notes(notes), observations="\n\n".join(blocks))
        history = [*state.get("history", []), "\n\n".join(blocks)] if cfg.act_memory == "raw" else []
        t0 = time.perf_counter()
        try:
            res = deps.reflect_chat.chat(msgs, schema=ReflectOut, prompt_version=p.version_id, sample=deps.sample)
        except StructuredOutputError as e:
            step = {"step": "reflect", "round": state["rounds"], "error": str(e)}
            return {"observations": [], "status": "（上一轮反思失败）", "trace": [*state.get("trace", []), step],
                    "llm_calls": state.get("llm_calls", 0) + 1, "history": history}  # fmt: skip
        out: ReflectOut = res.parsed  # type: ignore[assignment]
        invalid = state.get("invalid_notes", 0)
        added, rejected = [], []
        texts = {}
        for c in allowed:
            try:
                texts[c] = tb.backends.chunks.by_id(c)["text"]
            except KeyError:
                continue
        existing = {(n["chunk_id"], n["claim"]) for n in notes}
        for n in out.new_notes:
            cid = n.chunk_id.strip()
            if cid not in allowed or (cid, n.claim) in existing:
                invalid += cid not in allowed
                rejected.append({"chunk_id": cid, "claim": n.claim[:80], "why": "not_observed" if cid not in allowed
                                 else "duplicate"})  # fmt: skip
                continue
            if len(notes) + len(added) >= cfg.max_notes:
                invalid += 1
                rejected.append({"chunk_id": cid, "why": "max_notes"})
                continue
            quote = n.quote.strip()[:80]
            quote_ok = bool(quote) and cid in texts and _norm(quote) in _norm(texts[cid])
            added.append({"claim": n.claim.strip()[:200], "chunk_id": cid, "quote": quote, "quote_ok": quote_ok,
                          "seen_text": allowed[cid], "round": state["rounds"]})  # fmt: skip
        step = {
            "step": "reflect", "round": state["rounds"], "prompt": p.version_id, "tokens": res.usage.total_tokens,
            "cached": res.cached, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "added": added, "rejected": rejected, "status": out.status[:300], "sufficient": out.sufficient,
        }  # fmt: skip
        upd: dict[str, Any] = {
            **_usage_update(state, res), "notes": notes + added, "status": out.status.strip()[:300] or "（无）",
            "observations": [], "invalid_notes": invalid, "trace": [*state.get("trace", []), step],
            "history": history,
        }  # fmt: skip
        upd["last_round_tokens"] = upd["prompt_tokens"] + upd["completion_tokens"] - state.get("round_start_tokens", 0)
        if state.get("finish_requested"):
            upd["termination"] = "finish"
        return upd

    def after_act(state: AgentState) -> str:
        return "answer" if state.get("termination") else "execute"

    def after_reflect(state: AgentState) -> str:
        return "answer" if state.get("termination") else "act"

    def answer_node(state: AgentState) -> dict[str, Any]:
        return {}  # 作答由统一的作答环节完成（AgentSystem），这里只作为图的终点

    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("act", act_node)
    g.add_node("execute", execute_node)
    g.add_node("reflect", reflect_node)
    g.add_node("answer", answer_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "act")
    g.add_conditional_edges("act", after_act, {"execute": "execute", "answer": "answer"})
    g.add_edge("execute", "reflect")
    g.add_conditional_edges("reflect", after_reflect, {"act": "act", "answer": "answer"})
    g.add_edge("answer", END)
    return g.compile()


def run_agent(question: str, deps: AgentDeps) -> AgentState:
    graph = build_graph(deps)
    init: AgentState = {"question": question, "notes": [], "actions": [], "trace": [], "rounds": 0, "tool_calls": 0}
    # 每轮 = act + execute + reflect 三步，外加 plan / answer 和最后一次 act（上限检查）
    limit = 3 * deps.cfg.max_rounds + 6
    return graph.invoke(init, {"recursion_limit": limit})
