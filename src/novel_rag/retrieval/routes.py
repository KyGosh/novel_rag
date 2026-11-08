"""P3 检索价值验证用的两种新召回路（固定流程的变体，不是 Agent）：

- 摘要定位：先检索章节摘要取前 N 章，再在这些章内检索 chunk。
  route 模式：章内检索结果作为额外一路接入 RRF；restrict 模式：整条链路只在这些章内检索。
- 图谱第三路：链接问题中的实体，取实体之间直接关系与路径的证据 chunk、各实体主要关系的证据 chunk，以及
  高频提及的 chunk，排成一个列表作为第三路接入 RRF，再统一 rerank（不直接拼进 prompt）。

两路都接受同一个 SearchFilter（含 max_reading_order），结果再用 accepts 复核。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from novel_rag.graph_retrieval.linking import EntityLinker
from novel_rag.graph_retrieval.templates import GraphQueries
from novel_rag.retrieval.base import Hit, Retriever
from novel_rag.retrieval.expand import ChunkIndex
from novel_rag.retrieval.filters import SearchFilter
from novel_rag.retrieval.fusion import rrf_fuse
from novel_rag.retrieval.hybrid import check_filter
from novel_rag.retrieval.summary_store import SummaryRetriever


def locate_chapters(locator: SummaryRetriever, query: str, n: int, flt: SearchFilter) -> tuple[int, ...]:
    hits = locator.search(query, n, flt)
    return tuple(int(h.fields["reading_order"]) for h in hits)


class SummaryLocateRoute(Retriever):
    """摘要定位作为一路：在摘要选出的前 N 章内跑 BM25 + 向量，RRF 融合后返回。"""

    name = "summary_locate"

    def __init__(
        self,
        locator: SummaryRetriever,
        inner_routes: dict[str, Retriever],
        *,
        n_chapters: int,
        route_top_k: int,
        rrf_k: int,
    ):
        self.locator, self.inner_routes = locator, inner_routes
        self.n_chapters, self.route_top_k, self.rrf_k = n_chapters, route_top_k, rrf_k
        self.last_latency_ms: dict[str, float] = {}
        self.last_chapters: tuple[int, ...] = ()

    def search(self, query: str, top_k: int, flt: SearchFilter | None = None) -> list[Hit]:
        flt = flt or SearchFilter()
        t0 = time.perf_counter()
        chapters = locate_chapters(self.locator, query, self.n_chapters, flt)
        self.last_chapters = chapters
        t1 = time.perf_counter()
        if not chapters:
            return []
        restricted = flt.with_reading_orders(chapters)
        routes = {n: r.search(query, self.route_top_k, restricted) for n, r in self.inner_routes.items()}
        fused = rrf_fuse(routes, k=self.rrf_k, top_k=top_k)
        check_filter(fused, restricted, self.name)
        self.last_latency_ms = {"locate": (t1 - t0) * 1000, "total": (time.perf_counter() - t0) * 1000}
        return fused

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "locator": self.locator.describe(),
            "n_chapters": self.n_chapters,
            "route_top_k": self.route_top_k,
            "inner": list(self.inner_routes),
        }


class ChapterRestrictedRetriever(Retriever):
    """restrict 模式：先用摘要选出前 N 章，整条检索链路（含 rerank）只在这些章内进行。"""

    def __init__(self, inner: Retriever, locator: SummaryRetriever, n_chapters: int):
        self.inner, self.locator, self.n_chapters = inner, locator, n_chapters
        self.name = f"summary_restrict({inner.name})"
        self.last_latency_ms: dict[str, float] = {}

    def search(self, query: str, top_k: int, flt: SearchFilter | None = None) -> list[Hit]:
        flt = flt or SearchFilter()
        t0 = time.perf_counter()
        chapters = locate_chapters(self.locator, query, self.n_chapters, flt)
        t1 = time.perf_counter()
        hits = self.inner.search(query, top_k, flt.with_reading_orders(chapters)) if chapters else []
        lat = dict(getattr(self.inner, "last_latency_ms", {}) or {})
        lat["locate"] = (t1 - t0) * 1000
        lat["total"] = (time.perf_counter() - t0) * 1000
        self.last_latency_ms = lat
        return hits

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "inner": self.inner.describe(), "n_chapters": self.n_chapters}


class GraphRoute(Retriever):
    """图谱第三路：实体链接 → 实体间关系 / 路径证据、主要关系证据、高频提及 chunk → 排成列表。"""

    name = "graph"

    def __init__(
        self,
        queries: GraphQueries,
        linker: EntityLinker,
        chunks: ChunkIndex,
        *,
        max_entities: int = 4,
        per_entity_relations: int = 10,
        per_entity_mentions: int = 10,
        path_limit: int = 5,
        entity_types: Sequence[str] | None = None,
        max_tier: int = 3,
    ):
        self.q, self.linker, self.chunks = queries, linker, chunks
        self.max_entities = max_entities
        self.per_entity_relations = per_entity_relations
        self.per_entity_mentions = per_entity_mentions
        self.path_limit = path_limit
        self.entity_types = entity_types
        self.max_tier = max_tier  # 0 = 实体间直接关系；1 = + 路径；2 = + 各实体主要关系；3 = + 高频提及
        self.last_latency_ms: dict[str, float] = {}
        self.last_debug: dict[str, Any] = {}

    def _entities(self, query: str, max_ro: int | None) -> list[str]:
        spans = self.linker.link(query, max_ro, self.entity_types)
        out: list[str] = []
        for sp in spans:
            # 同名多个候选时取提及最多的一个（固定流程不反问）
            eid = sp.candidates[0].entity_id
            if eid not in out:
                out.append(eid)
        return out[: self.max_entities]

    def search(self, query: str, top_k: int, flt: SearchFilter | None = None) -> list[Hit]:
        flt = flt or SearchFilter()
        max_ro = flt.max_reading_order
        t0 = time.perf_counter()
        ents = self._entities(query, max_ro)
        ranked: list[tuple[int, int, str]] = []  # (层级, -证据数, chunk_id)

        def add(tier: int, weight: int, evidence: list[dict[str, Any]]) -> None:
            for e in evidence:
                ranked.append((tier, -weight, e["chunk_id"]))

        for i, a in enumerate(ents):
            for b in ents[i + 1 :]:
                for r in self.q.between(a, b, max_ro, limit=10).rows:
                    add(0, r["n_evidence"], r["evidence"])
                for p in self.q.paths(a, b, max_ro, limit=self.path_limit).rows:
                    for r in p["rels"]:
                        add(1, r["n_evidence"], r["evidence"])
        for a in ents:
            if self.max_tier >= 2:
                for r in self.q.neighbors(a, max_ro, limit=self.per_entity_relations).rows:
                    add(2, r["n_evidence"], r["evidence"])
            if self.max_tier >= 3:
                for m in self.q.mentions(a, max_ro, by_count=True, limit=self.per_entity_mentions).rows:
                    ranked.append((3, -int(m["count"]), m["chunk_id"]))
        ranked.sort()
        hits: list[Hit] = []
        seen: set[str] = set()
        for _tier, _w, cid in ranked:
            if cid in seen:
                continue
            seen.add(cid)
            try:
                fields = self.chunks.by_id(cid)
            except KeyError:
                continue  # 图谱里有、当前检索范围的 chunk 索引里没有
            if not flt.accepts(fields):
                continue
            hits.append(Hit(len(hits) + 1, cid, 1.0 / (len(hits) + 1), dict(fields)))
            if len(hits) >= top_k:
                break
        check_filter(hits, flt, self.name)
        self.last_debug = {"entities": ents}
        self.last_latency_ms = {"total": (time.perf_counter() - t0) * 1000}
        return hits

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_entities": self.max_entities,
            "per_entity_relations": self.per_entity_relations,
            "per_entity_mentions": self.per_entity_mentions,
            "path_limit": self.path_limit,
            "max_tier": self.max_tier,
        }
