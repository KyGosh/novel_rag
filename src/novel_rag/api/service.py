"""交互式问答服务（P6 演示）：命令行入口 scripts/ask.py 与本地 Web 演示（api/）共用。

- 模式：rag = 固定 RAG；agent = Agent（默认配置，即 test 上确认过的那套）。两种模式共用同一套工具后端（同一个
  GPU 锁）。按需升级（qa/route.py）的路由信号没有通过可行性门槛（P6，eval/results/p6_route_feasibility.md），
  不作为演示模式。
- 阅读进度：max_reading_order（全局阅读顺序）；reading_order_of(卷, 章) 把"读到第几卷第几章"换算过来。
  None 表示不限（读完全部）。
- on_event：各环节的事件（演示用的过程时间线），见 agent/loop.py 与 qa/route.py。
- 返回不含大段原文的结果：答案、依据、引用（带"卷名 · 章节标题"）、证据笔记（引文本身不超过 80 字）、成本、延迟。
系统在第一次用到时才组装（加载 GPU 模型、连接各服务），之后常驻。测试可以直接注入系统。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from novel_rag.config import Settings
from novel_rag.eval.dataset import EvalItem
from novel_rag.qa.systems import QASystem

MODES = ("rag", "agent")
MODE_NAMES = {"rag": "固定 RAG", "agent": "Agent"}
EventFn = Callable[[dict[str, Any]], None]


@dataclass
class Chapter:
    book_id: str
    chapter_idx: int
    title: str
    reading_order: int


@dataclass
class QAService:
    settings: Settings
    scope: tuple[str, ...]
    book_titles: dict[str, str]
    chapters: list[Chapter]
    where_chunk: Callable[[str], str]
    factories: dict[str, Callable[[], QASystem]] = field(default_factory=dict)
    _systems: dict[str, QASystem] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------ 组装

    @classmethod
    def build(cls, settings: Settings) -> QAService:
        """真实环境：检索范围 = 当前分块版本下有 chunk 的全部卷。"""
        from novel_rag.llm import get_chat_model, get_token_counter
        from novel_rag.qa.env import load_env
        from novel_rag.tools import build_backends

        s = settings
        env = load_env(s)
        counter = get_token_counter()
        cache: dict[str, int] = {}

        def n_tokens(text: str) -> int:
            if text not in cache:
                cache[text] = counter.count([text])[0]
            return cache[text]

        state: dict[str, Any] = {}

        def backends():  # noqa: ANN202
            if "b" not in state:
                state["b"] = build_backends(s, env.scope, env.chunks, env.titles)
            return state["b"]

        def rag() -> QASystem:
            from novel_rag.qa.systems import FixedRAGSystem

            return FixedRAGSystem(
                "rag", backends(), get_chat_model("answer"), n_tokens, scope=env.scope, top_k=s.answer.fixed_top_k,
                pool=s.eval.context_pool, budget=s.answer.context_token_budget, abstain=s.answer.abstain_text,
            )  # fmt: skip

        def agent() -> QASystem:
            from novel_rag.agent.system import AgentSystem
            from novel_rag.tools import Toolbox

            return AgentSystem(Toolbox(backends()), s, scope=env.scope, count_tokens=n_tokens)

        chapters = sorted(
            {
                (c["book_id"], c["chapter_idx"]): Chapter(
                    c["book_id"], c["chapter_idx"], c.get("chapter_title") or "", int(c["reading_order"])
                )
                for c in env.chunks.chunks
            }.values(),
            key=lambda ch: ch.reading_order,
        )

        def where_chunk(cid: str) -> str:
            return backends().where_chunk(cid)

        return cls(s, env.scope, env.titles, chapters, where_chunk, {"rag": rag, "agent": agent})

    def system(self, mode: str) -> QASystem:
        if mode not in MODES:
            raise ValueError(f"未知模式 {mode!r}，可选：{', '.join(MODES)}")
        with self._lock:
            if mode not in self._systems:
                self._systems[mode] = self.factories[mode]()
            return self._systems[mode]

    # ------------------------------------------------------------------ 阅读进度

    def volumes(self) -> list[dict[str, Any]]:
        out = []
        for b in self.scope:
            chs = [ch for ch in self.chapters if ch.book_id == b]
            out.append({"book_id": b, "title": self.book_titles.get(b, b),
                        "chapters": [{"idx": ch.chapter_idx, "title": ch.title, "reading_order": ch.reading_order}
                                     for ch in chs]})  # fmt: skip
        return out

    def reading_order_of(self, volume: str | int, chapter: int | None = None) -> int:
        """volume：卷 ID，或从 1 开始的卷序号；chapter：章号（chapter_idx），None 表示读完这一卷。"""
        if isinstance(volume, int) or (isinstance(volume, str) and volume.isdigit()):
            k = int(volume)
            if not 1 <= k <= len(self.scope):
                raise ValueError(f"卷序号应在 1–{len(self.scope)} 之间")
            book = self.scope[k - 1]
        elif volume in self.scope:
            book = volume
        else:
            raise ValueError(f"没有卷 {volume!r}；可选：{', '.join(self.scope)}")
        chs = [ch for ch in self.chapters if ch.book_id == book]
        if chapter is None:
            return max(ch.reading_order for ch in chs)
        hit = [ch for ch in chs if ch.chapter_idx == chapter]
        if not hit:
            raise ValueError(f"{book} 没有第 {chapter} 章（章号 {chs[0].chapter_idx}–{chs[-1].chapter_idx}）")
        return hit[0].reading_order

    def describe_progress(self, max_reading_order: int | None) -> str:
        if max_reading_order is None:
            return "不限（全部卷）"
        ch = next((c for c in reversed(self.chapters) if c.reading_order <= max_reading_order), None)
        if ch is None:
            return f"reading_order ≤ {max_reading_order}"
        return f"读到《{self.book_titles.get(ch.book_id, ch.book_id)}》· {ch.title or f'第{ch.chapter_idx}章'}"

    # ------------------------------------------------------------------ 问答

    def ask(
        self, question: str, mode: str = "agent", max_reading_order: int | None = None,
        on_event: EventFn | None = None, sample: int = 0,
    ) -> dict[str, Any]:  # fmt: skip
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")
        if len(question) > 200:
            raise ValueError("问题太长（不超过 200 字）")
        system = self.system(mode)
        item = EvalItem.model_construct(
            id="ask", type="fact", question=question, answer="", evidence=[], source="human", split="dev",
            max_reading_order=max_reading_order, answer_aliases=[],
        )  # fmt: skip
        out = system.run(item, sample, on_event=on_event)  # type: ignore[call-arg]
        rec, trace = out.record, out.trace
        ans = rec["answer"]
        notes = [
            {"claim": n["claim"], "quote": n.get("quote", ""), "chunk_id": n["chunk_id"],
             "where": self.where_chunk(n["chunk_id"])}
            for n in (trace.get("notes") or [])
        ]  # fmt: skip
        route = rec.get("route")
        return {
            "question": question, "mode": mode, "mode_name": MODE_NAMES[mode],
            "max_reading_order": max_reading_order, "progress": self.describe_progress(max_reading_order),
            "answer": ans.get("answer", ""), "rationale": ans.get("rationale", ""),
            "abstained": bool(ans.get("abstained")),
            "citations": [{"chunk_id": c, "where": self.where_chunk(c)} for c in ans.get("citations", [])],
            "notes": notes, "plan": trace.get("plan"),
            "route": {k: route.get(k) for k in ("signal", "reason", "escalate")} if route else None,
            "cost": rec.get("cost", {}), "latency_ms": rec.get("latency_ms", {}),
            "termination": rec.get("termination"),
        }  # fmt: skip
