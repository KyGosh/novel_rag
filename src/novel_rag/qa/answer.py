"""统一的作答环节：闭卷 / 固定 RAG / 固定 GraphRAG / Agent 共用同一个提示词、同一个模型、同一份 4000 token 预算。

- 输入：问题 + 证据上下文（闭卷时没有上下文）。
  - 固定 RAG：检索结果的前 8 个 chunk，按排名贪心装入预算。
  - Agent：证据笔记 + 笔记所引 chunk 的原文（笔记在前，原文按笔记顺序装入剩余预算）。
- 输出（json_object + pydantic 校验）：简短答案（专门用于判分）、一两句依据、引用列表。
- 引用校验：只保留出现在"允许引用集合"里的编号（固定 RAG = 送进上下文的 chunk；Agent = 笔记引用的 chunk），
  不合格的删掉并计数。
- 证据不足时回答"原文未提及"，单独统计为弃答。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from novel_rag.llm.base import ChatModel, StructuredOutputError
from novel_rag.prompts import Prompt, load_prompt


class AnswerOutput(BaseModel):
    answer: str = Field(description="简短答案（不超过 30 字）；证据不足时写弃答语")
    rationale: str = Field("", description="一两句依据")
    citations: list[str] = Field(default_factory=list, description="支撑答案的证据编号")


@dataclass
class EvidenceBlock:
    chunk_id: str
    where: str
    text: str


@dataclass
class AnswerContext:
    """作答环节实际看到的上下文（同预算召回按 blocks 的字符区间计算）。"""

    text: str
    blocks: list[EvidenceBlock] = field(default_factory=list)
    tokens: int = 0
    notes_tokens: int = 0
    dropped: list[str] = field(default_factory=list)  # 因预算放不下的 chunk


def pack_blocks(
    blocks: Sequence[EvidenceBlock], budget: int, count: Callable[[str], int], used: int = 0
) -> tuple[list[EvidenceBlock], int, list[str]]:
    """按顺序贪心装箱（与固定 RAG 的 cov@B 同一口径：放不下就跳过，继续看后面的）。"""
    out, dropped = [], []
    for b in blocks:
        n = count(b.text)
        if used + n <= budget:
            out.append(b)
            used += n
        else:
            dropped.append(b.chunk_id)
    return out, used, dropped


def render_blocks(blocks: Sequence[EvidenceBlock]) -> str:
    return "\n\n".join(f"[{b.chunk_id}] {b.where}\n{b.text}" for b in blocks)


def chunks_context(blocks: Sequence[EvidenceBlock], budget: int, count: Callable[[str], int]) -> AnswerContext:
    packed, used, dropped = pack_blocks(blocks, budget, count)
    return AnswerContext(
        "原文证据：\n" + render_blocks(packed) if packed else "原文证据：（无）", packed, used, 0, dropped
    )


def notes_context(
    notes_text: str, blocks: Sequence[EvidenceBlock], budget: int, count: Callable[[str], int]
) -> AnswerContext:
    n_notes = count(notes_text) if notes_text else 0
    packed, used, dropped = pack_blocks(blocks, budget, count, used=n_notes)
    parts = [f"证据笔记：\n{notes_text or '（无）'}"]
    if packed:
        parts.append("笔记引用的原文：\n" + render_blocks(packed))
    return AnswerContext("\n\n".join(parts), packed, used, n_notes, dropped)


def build_messages(prompt: Prompt, question: str, context: AnswerContext | None, abstain: str) -> list[dict[str, str]]:
    sec = prompt.sections()
    system = sec["system_closed" if context is None else "system"].replace("{abstain}", abstain)
    user = sec["user"].replace("{question}", question).replace("{context}", context.text if context else "")
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def is_abstain(answer: str, abstain: str) -> bool:
    a = answer.strip().strip("。.").strip()
    return not a or abstain in a


def answer_question(
    question: str,
    context: AnswerContext | None,
    chat: ChatModel,
    *,
    allowed_citations: set[str] | None,
    abstain: str,
    prompt: Prompt | None = None,
    sample: int = 0,
) -> dict[str, Any]:
    """返回不含原文的结构化结果（answer / rationale / citations / 计数 / 用量）。"""
    prompt = prompt or load_prompt("qa_answer")
    msgs = build_messages(prompt, question, context, abstain)
    try:
        res = chat.chat(msgs, schema=AnswerOutput, prompt_version=prompt.version_id, sample=sample)
        out: AnswerOutput = res.parsed  # type: ignore[assignment]
        usage, cached, error = res.usage, res.cached, None
    except StructuredOutputError as e:
        out = AnswerOutput(answer=abstain, rationale="", citations=[])
        usage, cached, error = None, False, f"结构化输出失败：{e}"
    cited = list(dict.fromkeys(c.strip() for c in out.citations if c.strip()))
    allowed = allowed_citations if allowed_citations is not None else set()
    valid = [c for c in cited if c in allowed]
    abstained = is_abstain(out.answer, abstain)
    return {
        "answer": out.answer.strip(),
        "rationale": out.rationale.strip(),
        "abstained": abstained,
        "citations": [] if abstained else valid,
        "invalid_citations": [c for c in cited if c not in allowed],
        "prompt_version": prompt.version_id,
        "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens} if usage else {},
        "cached": cached,
        "error": error,
    }
