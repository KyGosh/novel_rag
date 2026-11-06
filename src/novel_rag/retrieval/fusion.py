"""RRF（Reciprocal Rank Fusion）：N 路结果只按排名融合，score = Σ 1 / (k + rank)，rank 从 1 开始。

BM25 分数与余弦相似度量纲不同、分布也不同，不能直接加权相加；RRF 只用排名，不需要做分数归一化。
按 chunk_id 去重；每一路的排名与原始分数保留在 fields["routes"] 中，供 bad case 分析。
图谱召回（P3/P4）作为额外一路接入，不需要改这里。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from novel_rag.retrieval.base import Hit


def rrf_fuse(routes: Mapping[str, Sequence[Hit]], k: int = 60, top_k: int | None = None) -> list[Hit]:
    """融合多路结果。同分时按"最好的单路排名"、再按路名顺序与 chunk_id 打破平局，保证结果确定。"""
    if k < 0:
        raise ValueError("RRF 的 k 不能为负")
    order = list(routes)
    scores: dict[str, float] = {}
    fields: dict[str, dict] = {}
    debug: dict[str, dict[str, dict[str, float]]] = {}
    for name in order:
        seen: set[str] = set()
        for h in routes[name]:
            if h.chunk_id in seen:  # 同一路内重复出现只算第一次
                continue
            seen.add(h.chunk_id)
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (k + h.rank)
            debug.setdefault(h.chunk_id, {})[name] = {"rank": h.rank, "score": h.score}
            base = fields.setdefault(h.chunk_id, {})
            for key, value in h.fields.items():
                if key not in ("routes", "rrf", "rerank"):
                    base.setdefault(key, value)

    def sort_key(cid: str) -> tuple:
        best = min(d["rank"] for d in debug[cid].values())
        first_route = min(order.index(r) for r in debug[cid])
        return (-scores[cid], best, first_route, cid)

    ranked = sorted(scores, key=sort_key)
    if top_k is not None:
        ranked = ranked[:top_k]
    return [
        Hit(i, cid, scores[cid], {**fields[cid], "routes": debug[cid], "rrf": scores[cid]})
        for i, cid in enumerate(ranked, 1)
    ]
