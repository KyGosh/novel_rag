"""四个对照系统共用的接口：每个系统对一道题返回一条不含原文的结构化记录 + 一份含原文的完整 trace。

- 闭卷：不给证据，只靠模型自身知识（1 次 LLM 调用）。
- 固定 RAG：原文检索工具的同一实现（run_search，P2 基线链路）取结果，前 8 个 chunk 装入 4000 token 作答。
- 固定 GraphRAG：同上，检索路加图谱第三路（P3 定的 tier 1）。
- Agent：见 agent/（LangGraph），通过 AgentSystem 适配到同一接口。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from novel_rag.eval.dataset import EvalItem
from novel_rag.llm.base import ChatModel
from novel_rag.qa.answer import EvidenceBlock, answer_question, chunks_context
from novel_rag.retrieval.filters import SearchFilter
from novel_rag.tools.backends import ToolBackends
from novel_rag.tools.text import run_search


@dataclass
class SystemOutput:
    """record：写进 eval/results 的结构化结果（不含原文）；trace：写进 data/traces 的完整记录（可含原文）。"""

    record: dict[str, Any]
    trace: dict[str, Any] = field(default_factory=dict)


class QASystem(Protocol):
    name: str

    def run(self, item: EvalItem, sample: int = 0) -> SystemOutput: ...

    def describe(self) -> dict[str, Any]: ...


def _usage(ans: dict[str, Any]) -> dict[str, int]:
    u = ans.get("usage") or {}
    return {"prompt_tokens": int(u.get("prompt_tokens", 0)), "completion_tokens": int(u.get("completion_tokens", 0))}


class ClosedBookSystem:
    name = "closed"

    def __init__(self, chat: ChatModel, abstain: str):
        self.chat, self.abstain = chat, abstain

    def run(self, item: EvalItem, sample: int = 0) -> SystemOutput:
        t0 = time.perf_counter()
        ans = answer_question(item.question, None, self.chat, allowed_citations=set(), abstain=self.abstain,
                              sample=sample)  # fmt: skip
        lat = (time.perf_counter() - t0) * 1000
        rec = {
            "answer": ans,
            "evidence_sets": {"context": []},
            "cost": {"llm_calls": 1, "tool_calls": 0, **_usage(ans)},
            "latency_ms": {"total": round(lat, 1), "llm": round(lat, 1)},
            "termination": "answered",
        }
        return SystemOutput(rec, {"answer": ans})

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "answer_model": self.chat.model_name, "prompt": "qa_answer"}


class FixedRAGSystem:
    """固定流程：检索 → 前 top_k 个 chunk 装入预算 → 作答。pool 个结果用于 R@K / MRR / cov@B（与 P2、P3 口径一致）。"""

    def __init__(
        self,
        name: str,
        backends: ToolBackends,
        chat: ChatModel,
        count_tokens: Callable[[str], int],
        *,
        scope: tuple[str, ...],
        top_k: int,
        pool: int,
        budget: int,
        abstain: str,
    ):
        self.name = name
        self.b, self.chat, self.count = backends, chat, count_tokens
        self.scope, self.top_k, self.pool, self.budget, self.abstain = scope, top_k, pool, budget, abstain

    def run(self, item: EvalItem, sample: int = 0) -> SystemOutput:
        t0 = time.perf_counter()
        flt = SearchFilter(book_ids=self.scope, max_reading_order=item.max_reading_order)
        hits, lat = run_search(self.b, item.question, self.pool, flt)
        t1 = time.perf_counter()
        blocks = []
        for h in hits[: self.top_k]:
            c = self.b.chunks.by_id(h.chunk_id)
            blocks.append(EvidenceBlock(h.chunk_id, self.b.where(c["book_id"], c["chapter_idx"]), c["text"]))
        ctx = chunks_context(blocks, self.budget, self.count)
        ans = answer_question(
            item.question, ctx, self.chat, allowed_citations={b.chunk_id for b in ctx.blocks}, abstain=self.abstain,
            sample=sample,
        )  # fmt: skip
        t2 = time.perf_counter()
        rec = {
            "answer": ans,
            "hits": [h.chunk_id for h in hits],
            "evidence_sets": {"shown": [b.chunk_id for b in ctx.blocks], "context": [b.chunk_id for b in ctx.blocks]},
            "context_tokens": ctx.tokens,
            "cost": {"llm_calls": 1, "tool_calls": 1, **_usage(ans)},
            "latency_ms": {
                "total": round((t2 - t0) * 1000, 1),
                "retrieval": round((t1 - t0) * 1000, 1),
                "rerank": round(lat.get("rerank", 0.0), 1),
                "llm": round((t2 - t1) * 1000, 1),
            },
            "termination": "answered",
        }
        return SystemOutput(rec, {"context": ctx.text, "answer": ans})

    def describe(self) -> dict[str, Any]:
        r = self.b.retriever
        return {
            "name": self.name,
            "retriever": r.describe() if r is not None else None,
            "top_k": self.top_k,
            "pool": self.pool,
            "budget": self.budget,
            "answer_model": self.chat.model_name,
        }
