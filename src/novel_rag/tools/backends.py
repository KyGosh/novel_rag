"""工具后端：检索器、chunk 索引、摘要、图谱等。真实环境由 build_backends 组装；单测直接构造并注入假实现。"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from novel_rag.retrieval.base import Retriever
from novel_rag.retrieval.expand import ChunkIndex
from novel_rag.retrieval.filters import SearchFilter


@dataclass
class ToolBackends:
    chunks: ChunkIndex
    book_titles: dict[str, str]  # book_id → 卷名
    retriever: Retriever | None = None  # 原文检索（P2 固定 RAG 那条链路）
    summary_retriever: Retriever | None = None  # 章节摘要检索
    summaries: dict[str, dict[str, Any]] | None = None  # 章节键 → 摘要记录
    graph: Any | None = None  # GraphQueries（或测试替身）
    # 实体 → 提及该实体的 chunk（受过滤条件约束）；None 表示不支持实体限定
    entity_chunks: Callable[[str, SearchFilter], list[str]] | None = None
    # 检索链路里的 GPU 模型（编码、rerank）由共享实例串行处理：题与题并发时不会同时占用显存
    gpu_lock: threading.Lock = field(default_factory=threading.Lock)
    chapter_titles: dict[tuple[str, int], str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chapter_titles:
            for c in self.chunks.chunks:
                self.chapter_titles.setdefault((c["book_id"], c["chapter_idx"]), c.get("chapter_title") or "")

    def where(self, book_id: str, chapter_idx: int) -> str:
        """ "卷名 · 章节标题"。"""
        title = self.chapter_titles.get((book_id, chapter_idx), f"第{chapter_idx}章")
        return f"{self.book_titles.get(book_id, book_id)} · {title}"

    def where_chunk(self, chunk_id: str) -> str:
        try:
            c = self.chunks.by_id(chunk_id)
        except KeyError:
            return chunk_id
        return self.where(c["book_id"], c["chapter_idx"])

    def chunk_ro(self, chunk_id: str) -> int | None:
        try:
            return int(self.chunks.by_id(chunk_id)["reading_order"])
        except KeyError:
            return None


def load_summaries(path: Path) -> dict[str, dict[str, Any]]:
    import json

    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line)
            out[r["chapter_key"]] = r
    return out


def es_entity_chunks(store: Any, max_chunks: int) -> Callable[[str, SearchFilter], list[str]]:
    """用 es2 索引的 entity_ids 字段取提及某实体的 chunk（受阅读进度等过滤）；超过 max_chunks 时返回空列表的
    哨兵由调用方处理（实体出现得太广，限定没有意义）。"""

    def fn(entity_id: str, flt: SearchFilter) -> list[str]:
        clauses = [*flt.to_es_filter(), {"term": {"entity_ids": entity_id}}]
        res = store.client.search(
            index=store.name, query={"bool": {"filter": clauses}}, size=max_chunks + 1, source=False,
            sort=[{"chunk_id": "asc"}],
        )  # fmt: skip
        return [h["_id"] for h in res["hits"]["hits"]]

    return fn


def build_backends(
    settings: Any,
    book_ids: Sequence[str],
    chunk_index: ChunkIndex,
    book_titles: dict[str, str],
    *,
    routes: tuple[str, ...] | None = None,
    with_graph: bool = True,
    with_summaries: bool = True,
) -> ToolBackends:
    """按 config 组装真实后端。routes 覆盖原文检索的检索路（固定 GraphRAG 用 bm25,vector,graph）。"""
    from novel_rag.ingestion.chunking import chunk_version
    from novel_rag.retrieval.es_store import ESChunkStore, index_name
    from novel_rag.retrieval.factory import build_retriever, build_summary_retriever, latest_summary_version

    s = settings
    if routes is not None:
        s = s.model_copy(update={"retrieval": s.retrieval.model_copy(update={"routes": routes})})
    book_ids = tuple(book_ids)
    retriever = build_retriever(s, book_ids, "hybrid", chunk_index=chunk_index)
    b = ToolBackends(chunks=chunk_index, book_titles=book_titles, retriever=retriever)
    if with_summaries:
        sver = s.summary.summary_version or latest_summary_version(s.paths.data_dir)
        b.summaries = load_summaries(s.paths.data_dir / "summaries" / sver / "summaries.jsonl")
        b.summary_retriever = build_summary_retriever(s, book_ids)
    if with_graph:
        from neo4j import GraphDatabase

        from novel_rag.graph_retrieval.templates import GraphQueries
        from novel_rag.kg.common import series_of

        nc = s.neo4j
        driver = GraphDatabase.driver(nc.uri, auth=(nc.user, nc.password.get_secret_value()))
        b.graph = GraphQueries(driver, nc.database, s.kg, series_of(s.paths.books_dir, book_ids[0]))
    if s.retrieval.es_analysis_version != "es1":
        name = index_name(chunk_version(s.chunk), s.retrieval.es_analysis_version)
        store = ESChunkStore.connect(s.es, name, s.retrieval, s.retrieval.es_analysis_version)
        b.entity_chunks = es_entity_chunks(store, s.tools.entity_filter_max_chunks)
    return b
