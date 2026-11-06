"""检索指标。相关性只看字符区间重叠，与 chunk_id 无关，分块参数改了评测集照样能用。

- 命中：证据区间与 chunk 的重叠字数 ≥ min(区间长度 × ratio, chars)。
- Recall@K：按证据统计——一题所有证据中被 top-K 命中的比例，再对题目平均（多跳题只找到一半证据会体现出来）。
- Hit@K：top-K 中至少命中一段证据的题目比例。
- MRR：第一个命中 chunk 的排名倒数（只看检索返回的前 max(K) 个）。
- 证据覆盖率 cov@B：上下文（small-to-big 扩展、按 token 预算 B 装箱之后）覆盖的证据比例，相关性判定同上。
  扩展会让上下文变长，所以扩展前后、不同分块大小之间用它比较，而不是 Recall@K。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

from novel_rag.eval.dataset import Evidence


@dataclass(frozen=True)
class Span:
    book_id: str
    start: int
    end: int


def overlap(a: Span, b_book: str, b_start: int, b_end: int) -> int:
    if a.book_id != b_book:
        return 0
    return max(0, min(a.end, b_end) - max(a.start, b_start))


def is_relevant(chunk: Span, ev: Evidence, *, min_ratio: float, min_chars: int) -> bool:
    need = min((ev.char_end - ev.char_start) * min_ratio, min_chars)
    return overlap(chunk, ev.book_id, ev.char_start, ev.char_end) >= max(need, 1)


def score_ranking(
    ranked: Sequence[Span], evidence: Sequence[Evidence], ks: Sequence[int], *, min_ratio: float, min_chars: int
) -> dict[str, Any]:
    """返回每段证据首次被命中的排名（未命中为 None）以及该题的各项指标。"""
    first: list[int | None] = []
    for ev in evidence:
        rank = next(
            (r for r, ch in enumerate(ranked, 1) if is_relevant(ch, ev, min_ratio=min_ratio, min_chars=min_chars)), None
        )
        first.append(rank)
    hit_ranks = [r for r in first if r is not None]
    out: dict[str, Any] = {"evidence_ranks": first}
    for k in ks:
        out[f"recall@{k}"] = sum(1 for r in first if r is not None and r <= k) / len(first)
        out[f"hit@{k}"] = float(any(r <= k for r in hit_ranks))
    best = min(hit_ranks) if hit_ranks else None
    out["mrr"] = 1.0 / best if best is not None and best <= max(ks) else 0.0
    return out


def coverage(contexts: Sequence[Span], evidence: Sequence[Evidence], *, min_ratio: float, min_chars: int) -> float:
    """被任一上下文段命中的证据比例。"""
    covered = sum(
        1 for ev in evidence if any(is_relevant(c, ev, min_ratio=min_ratio, min_chars=min_chars) for c in contexts)
    )
    return covered / len(evidence)


def metric_keys(rows: Sequence[dict[str, Any]], ks: Sequence[int]) -> list[str]:
    """固定的排名指标在前，其余（如 cov@B）按首次出现的顺序排在后面。"""
    keys = [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks] + ["mrr"]
    for r in rows:
        keys += [k for k in r["metrics"] if k not in keys]
    return keys


def aggregate(rows: Sequence[dict[str, Any]], ks: Sequence[int]) -> dict[str, Any]:
    keys = metric_keys(rows, ks)

    def summarize(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"n": 0}
        return {
            "n": len(group),
            **{k: round(mean(r["metrics"][k] for r in group), 4) for k in keys if k in group[0]["metrics"]},
        }

    out: dict[str, Any] = {"overall": summarize(rows), "by_type": {}, "by_source": {}}
    for t in sorted({r["type"] for r in rows}):
        out["by_type"][t] = summarize([r for r in rows if r["type"] == t])
    for s in sorted({r["source"] for r in rows}):
        out["by_source"][s] = summarize([r for r in rows if r["source"] == s])
    return out
