"""混合检索：多路召回（BM25、向量……）→ RRF 融合 → 可选的 rerank。

漏斗（参数都在 config.retrieval）：每路取 route_top_k → RRF 取 fusion_top_k → 前 rerank_candidates 个送 reranker
重排，其余候选按融合顺序接在后面 → 返回前 top_k 个。small-to-big 扩展不在检索器里做，见 expand.py。

防剧透：同一个过滤条件传给每一路；融合、重排之后再用 SearchFilter.accepts 逐条复核一遍，
任何一路漏掉过滤都会在这里报错，而不是悄悄把后文送进上下文。
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from novel_rag.llm.base import Reranker
from novel_rag.retrieval.base import Hit, Retriever
from novel_rag.retrieval.filters import SearchFilter
from novel_rag.retrieval.fusion import rrf_fuse


class FilterViolation(RuntimeError):
    """检索结果中出现了不满足过滤条件（如超出阅读进度）的 chunk。"""


def check_filter(hits: list[Hit], flt: SearchFilter, stage: str) -> None:
    bad = [h.chunk_id for h in hits if not flt.accepts(h.fields)]
    if bad:
        raise FilterViolation(f"{stage} 返回了不满足过滤条件的 chunk：{bad[:5]}")


class HybridRetriever(Retriever):
    def __init__(
        self,
        routes: Mapping[str, Retriever],
        *,
        route_top_k: Mapping[str, int],
        rrf_k: int,
        fusion_top_k: int,
        reranker: Reranker | None = None,
        rerank_candidates: int = 0,
        base_filter: SearchFilter | None = None,
        parallel: bool = True,
    ):
        if not routes:
            raise ValueError("至少需要一路检索")
        self.routes = dict(routes)
        self.route_top_k = {n: route_top_k[n] for n in self.routes}
        self.rrf_k = rrf_k
        self.fusion_top_k = fusion_top_k
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates if reranker is not None else 0
        self.base_filter = base_filter or SearchFilter()
        self.parallel = parallel and len(self.routes) > 1
        self._pool = ThreadPoolExecutor(max_workers=len(self.routes)) if self.parallel else None
        self.name = "+".join(self.routes) + ("+rerank" if self.reranker else "")
        self.last_latency_ms: dict[str, float] = {}

    def _run_route(self, name: str, query: str, flt: SearchFilter) -> tuple[list[Hit], dict[str, float]]:
        r = self.routes[name]
        t0 = time.perf_counter()
        hits = r.search(query, self.route_top_k[name], flt)
        lat = dict(getattr(r, "last_latency_ms", {}) or {})
        lat["total"] = (time.perf_counter() - t0) * 1000
        return hits, lat

    def search(self, query: str, top_k: int, flt: SearchFilter | None = None) -> list[Hit]:
        flt = flt or self.base_filter
        t0 = time.perf_counter()
        if self._pool is not None:
            futures = {n: self._pool.submit(self._run_route, n, query, flt) for n in self.routes}
            results = {n: f.result() for n, f in futures.items()}
        else:
            results = {n: self._run_route(n, query, flt) for n in self.routes}
        t1 = time.perf_counter()
        latency: dict[str, float] = {"routes": (t1 - t0) * 1000}
        for n, (hits, lat) in results.items():
            check_filter(hits, flt, n)
            for key, v in lat.items():
                latency[f"{n}.{key}"] = v

        fused = rrf_fuse({n: hits for n, (hits, _) in results.items()}, k=self.rrf_k, top_k=self.fusion_top_k)
        t2 = time.perf_counter()
        latency["fusion"] = (t2 - t1) * 1000

        if self.reranker is not None and fused:
            head, tail = fused[: self.rerank_candidates], fused[self.rerank_candidates :]
            scores = self.reranker.score(query, [h.fields.get("text", "") for h in head])
            order = sorted(range(len(head)), key=lambda i: (-scores[i], head[i].rank))
            reranked = [head[i] for i in order] + tail
            fused = [
                Hit(
                    rank,
                    h.chunk_id,
                    scores[order[rank - 1]] if rank <= len(head) else h.score,
                    {**h.fields, "rerank": scores[order[rank - 1]]} if rank <= len(head) else h.fields,
                )
                for rank, h in enumerate(reranked, 1)
            ]
            latency["rerank"] = (time.perf_counter() - t2) * 1000

        out = fused[:top_k]
        check_filter(out, flt, self.name)
        latency["total"] = (time.perf_counter() - t0) * 1000
        self.last_latency_ms = latency
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "routes": {n: r.describe() for n, r in self.routes.items()},
            "route_top_k": self.route_top_k,
            "rrf_k": self.rrf_k,
            "fusion_top_k": self.fusion_top_k,
            "reranker": self.reranker.model_name if self.reranker else None,
            "rerank_candidates": self.rerank_candidates,
        }
