"""按配置组装检索器（命令行工具与评测运行器共用）。"""

from __future__ import annotations

from novel_rag.config import Settings
from novel_rag.ingestion.chunking import chunk_version
from novel_rag.llm import get_embedder
from novel_rag.retrieval.filters import SearchFilter
from novel_rag.retrieval.milvus_store import MilvusChunkStore, VectorRetriever, collection_name


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
