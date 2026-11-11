"""按配置组装检索器（命令行工具与评测运行器共用）。

kind：
- vector  只用 Milvus 向量检索（P1 baseline）
- bm25    只用 ES BM25（方案由 retrieval.bm25_scheme 决定，索引由 retrieval.es_analysis_version 决定）
- hybrid  retrieval.routes 中的各路 → RRF → 可选 rerank（retrieval.rerank）
          路可以是 bm25 / vector / summary_locate（摘要定位后章内检索）/ graph（图谱第三路）
summary_restrict > 0 时，整条链路只在摘要选出的前 N 章内检索（摘要定位的 restrict 模式）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from novel_rag.config import Settings
from novel_rag.ingestion.chunking import chunk_version
from novel_rag.llm import get_embedder, get_reranker
from novel_rag.retrieval.base import Retriever
from novel_rag.retrieval.es_store import BM25Retriever, ESChunkStore, index_name
from novel_rag.retrieval.expand import ChunkIndex
from novel_rag.retrieval.filters import SearchFilter
from novel_rag.retrieval.hybrid import HybridRetriever
from novel_rag.retrieval.milvus_store import MilvusChunkStore, VectorRetriever, collection_name

RetrieverKind = Literal["vector", "bm25", "hybrid"]


def build_vector_retriever(
    settings: Settings, book_ids: tuple[str, ...], *, version: str | None = None, template: str | None = None
) -> VectorRetriever:
    embedder = get_embedder()
    name = collection_name(
        embedder.model_name,
        embedder.dim,
        version or chunk_version(settings.chunk),
        template or settings.retrieval.embed_text_template,
    )
    store = MilvusChunkStore.connect(settings.milvus, name, embedder.dim, settings.retrieval)
    if not store.exists():
        raise RuntimeError(f"Milvus 中没有 collection {name}，先运行 scripts/index_milvus.py")
    return VectorRetriever(store, embedder, SearchFilter(book_ids=book_ids))


def build_bm25_retriever(
    settings: Settings, book_ids: tuple[str, ...], *, version: str | None = None, scheme: str | None = None
) -> BM25Retriever:
    name = index_name(version or chunk_version(settings.chunk), settings.retrieval.es_analysis_version)
    store = ESChunkStore.connect(settings.es, name, settings.retrieval, settings.retrieval.es_analysis_version)
    if not store.exists():
        raise RuntimeError(f"ES 中没有索引 {name}，先运行 scripts/index_es.py")
    return BM25Retriever(store, scheme or settings.retrieval.bm25_scheme, SearchFilter(book_ids=book_ids))


def latest_summary_version(data_dir: Path) -> str:
    cands = sorted((data_dir / "summaries").glob("*/summaries.jsonl"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise RuntimeError("没有章节摘要，先运行 scripts/summarize.py")
    return cands[-1].parent.name


def build_summary_retriever(settings: Settings, book_ids: tuple[str, ...]):  # noqa: ANN201
    from novel_rag.retrieval.summary_store import (
        ESSummaryStore,
        MilvusSummaryStore,
        SummaryRetriever,
        es_index_name,
        milvus_collection_name,
    )

    sc = settings.summary
    sver = sc.summary_version or latest_summary_version(settings.paths.data_dir)
    embedder = get_embedder()
    es = ESSummaryStore.connect(settings.es, es_index_name(sver, settings.retrieval.es_analysis_version))
    mv = MilvusSummaryStore.connect(
        settings.milvus,
        milvus_collection_name(embedder.model_name, embedder.dim, sver),
        embedder.dim,
        settings.retrieval,
    )
    if not es.exists() or not mv.exists():
        raise RuntimeError(f"摘要索引 {es.name} / {mv.name} 不存在，先运行 scripts/index_summaries.py")
    return SummaryRetriever(
        es, mv, embedder, bm25_top_k=sc.bm25_top_k, vector_top_k=sc.vector_top_k, rrf_k=sc.rrf_k,
        base_filter=SearchFilter(book_ids=book_ids),
    )  # fmt: skip


def build_graph_route(settings: Settings, chunk_index: ChunkIndex, series_id: str = "asoiaf"):  # noqa: ANN201
    from neo4j import GraphDatabase

    from novel_rag.graph_retrieval.linking import EntityLinker
    from novel_rag.graph_retrieval.templates import GraphQueries
    from novel_rag.kg.common import load_series_kg
    from novel_rag.retrieval.routes import GraphRoute

    nc = settings.neo4j
    driver = GraphDatabase.driver(nc.uri, auth=(nc.user, nc.password.get_secret_value()))
    books = sorted({c["book_id"] for c in chunk_index.chunks})
    rules = load_series_kg(settings.paths.books_dir, books).rules
    kc = settings.kg
    linker = EntityLinker.from_neo4j(driver, nc.database, series_id, rules, min_mentions=kc.route_min_entity_mentions)
    queries = GraphQueries(driver, nc.database, kc, series_id)
    return GraphRoute(
        queries, linker, chunk_index,
        max_tier=kc.route_max_tier, entity_types=kc.route_entity_types,
        path_node_types=kc.route_path_node_types, path_exclude_node_types=kc.route_path_exclude_node_types,
        kg_version=queries.kg_version(),
    )  # fmt: skip


def build_retriever(
    settings: Settings,
    book_ids: tuple[str, ...],
    kind: RetrieverKind = "hybrid",
    *,
    version: str | None = None,
    template: str | None = None,
    chunk_index: ChunkIndex | None = None,
    summary_restrict: int = 0,
) -> Retriever:
    rc = settings.retrieval
    if kind == "vector":
        return build_vector_retriever(settings, book_ids, version=version, template=template)
    if kind == "bm25":
        return build_bm25_retriever(settings, book_ids, version=version)

    def bm25() -> Retriever:
        return build_bm25_retriever(settings, book_ids, version=version)

    def vector() -> Retriever:
        return build_vector_retriever(settings, book_ids, version=version, template=template)

    def summary_locate() -> Retriever:
        from novel_rag.retrieval.routes import SummaryLocateRoute

        return SummaryLocateRoute(
            build_summary_retriever(settings, book_ids), {"bm25": bm25(), "vector": vector()},
            n_chapters=settings.summary.locate_top_chapters, route_top_k=max(rc.bm25_top_k, rc.vector_top_k),
            rrf_k=rc.rrf_k,
        )  # fmt: skip

    def graph() -> Retriever:
        if chunk_index is None:
            raise ValueError("图谱第三路需要 chunk_index（取 chunk 正文供 rerank）")
        return build_graph_route(settings, chunk_index)

    builders = {"bm25": bm25, "vector": vector, "summary_locate": summary_locate, "graph": graph}
    unknown = set(rc.routes) - set(builders)
    if unknown:
        raise ValueError(f"未知的检索路：{sorted(unknown)}")
    routes = {name: builders[name]() for name in rc.routes}
    hybrid = HybridRetriever(
        routes,
        route_top_k={
            "bm25": rc.bm25_top_k,
            "vector": rc.vector_top_k,
            "summary_locate": rc.summary_locate_top_k,
            "graph": rc.graph_top_k,
        },
        rrf_k=rc.rrf_k,
        fusion_top_k=max(rc.fusion_top_k, rc.rerank_candidates if rc.rerank else 0),
        reranker=get_reranker() if rc.rerank else None,
        rerank_candidates=rc.rerank_candidates,
        base_filter=SearchFilter(book_ids=book_ids),
    )
    if summary_restrict > 0:
        from novel_rag.retrieval.routes import ChapterRestrictedRetriever

        return ChapterRestrictedRetriever(hybrid, build_summary_retriever(settings, book_ids), summary_restrict)
    return hybrid
