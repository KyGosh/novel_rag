"""按配置组装检索器（命令行工具与评测运行器共用）。

kind：
- vector  只用 Milvus 向量检索（P1 baseline）
- bm25    只用 ES BM25（方案由 retrieval.bm25_scheme 决定）
- hybrid  retrieval.routes 中的各路 → RRF → 可选 rerank（retrieval.rerank）
"""

from __future__ import annotations

from typing import Literal

from novel_rag.config import Settings
from novel_rag.ingestion.chunking import chunk_version
from novel_rag.llm import get_embedder, get_reranker
from novel_rag.retrieval.base import Retriever
from novel_rag.retrieval.es_store import BM25Retriever, ESChunkStore, index_name
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
    name = index_name(version or chunk_version(settings.chunk))
    store = ESChunkStore.connect(settings.es, name, settings.retrieval)
    if not store.exists():
        raise RuntimeError(f"ES 中没有索引 {name}，先运行 scripts/index_es.py")
    return BM25Retriever(store, scheme or settings.retrieval.bm25_scheme, SearchFilter(book_ids=book_ids))


def build_retriever(
    settings: Settings,
    book_ids: tuple[str, ...],
    kind: RetrieverKind = "hybrid",
    *,
    version: str | None = None,
    template: str | None = None,
) -> Retriever:
    rc = settings.retrieval
    if kind == "vector":
        return build_vector_retriever(settings, book_ids, version=version, template=template)
    if kind == "bm25":
        return build_bm25_retriever(settings, book_ids, version=version)
    builders = {
        "bm25": lambda: build_bm25_retriever(settings, book_ids, version=version),
        "vector": lambda: build_vector_retriever(settings, book_ids, version=version, template=template),
    }
    unknown = set(rc.routes) - set(builders)
    if unknown:
        raise ValueError(f"未知的检索路：{sorted(unknown)}")
    routes = {name: builders[name]() for name in rc.routes}
    return HybridRetriever(
        routes,
        route_top_k={"bm25": rc.bm25_top_k, "vector": rc.vector_top_k},
        rrf_k=rc.rrf_k,
        fusion_top_k=max(rc.fusion_top_k, rc.rerank_candidates if rc.rerank else 0),
        reranker=get_reranker() if rc.rerank else None,
        rerank_candidates=rc.rerank_candidates,
        base_filter=SearchFilter(book_ids=book_ids),
    )
