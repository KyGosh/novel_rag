"""Milvus 中的 chunk 向量库与向量检索器。

collection 命名为 chunks_{模型简称}_{维度}_{chunk 版本}[_tp]：
不同分块版本 / 编码模板各自一个 collection，可以并存做消融。
一个 collection 可以装多卷（检索范围用 book_id / reading_order 过滤）；导入按主键 upsert，并删除本次所导入各卷中
多余的 chunk，因此重复导入同一版本结果不变，也不会误删其他卷；权威数据仍是 chunks.jsonl。
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from novel_rag.config import MilvusConfig, RetrievalConfig
from novel_rag.llm.base import Embedder
from novel_rag.retrieval.base import Hit, Retriever
from novel_rag.retrieval.filters import SearchFilter

SCALAR_FIELDS = ("book_id", "chapter_idx", "seq", "reading_order", "section_type", "chunk_type", "pov", "chapter_title")


def collection_name(model_name: str, dim: int, chunk_version: str, template: str) -> str:
    short = re.sub(r"[^a-z0-9]", "", model_name.rsplit("/", 1)[-1].lower())
    return f"chunks_{short}_{dim}_{chunk_version}" + ("_tp" if template == "title_prefix" else "")


class MilvusChunkStore:
    def __init__(self, client: Any, name: str, dim: int, cfg: RetrievalConfig):
        self.client = client
        self.name = name
        self.dim = dim
        self.cfg = cfg

    @classmethod
    def connect(cls, mcfg: MilvusConfig, name: str, dim: int, cfg: RetrievalConfig) -> MilvusChunkStore:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=mcfg.uri, token=mcfg.token.get_secret_value(), db_name=mcfg.db_name)
        return cls(client, name, dim, cfg)

    def exists(self) -> bool:
        return bool(self.client.has_collection(self.name))

    def ensure_collection(self, description: str = "") -> None:
        if self.exists():
            return
        from pymilvus import DataType

        schema = self.client.create_schema(auto_id=False, description=description)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("book_id", DataType.VARCHAR, max_length=64)
        schema.add_field("chapter_idx", DataType.INT32)
        schema.add_field("seq", DataType.INT32)
        schema.add_field("reading_order", DataType.INT32)
        schema.add_field("section_type", DataType.VARCHAR, max_length=32)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=16)
        schema.add_field("pov", DataType.VARCHAR, max_length=64)
        schema.add_field("chapter_title", DataType.VARCHAR, max_length=256)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        index = self.client.prepare_index_params()
        index.add_index(
            "vector",
            index_type="HNSW",
            metric_type=self.cfg.metric_type,
            params={"M": self.cfg.hnsw_m, "efConstruction": self.cfg.hnsw_ef_construction},
        )
        self.client.create_collection(self.name, schema=schema, index_params=index)

    def _all_ids(self, book_ids: Iterable[str] | None = None) -> set[str]:
        ids: set[str] = set()
        expr = (
            SearchFilter(book_ids=tuple(sorted(book_ids))).to_milvus_expr()
            if book_ids is not None
            else 'chunk_id != ""'
        )
        it = self.client.query_iterator(self.name, batch_size=1000, filter=expr, output_fields=["chunk_id"])
        try:
            while batch := it.next():
                ids.update(r["chunk_id"] for r in batch)
        finally:
            it.close()
        return ids

    def sync(
        self, chunks: Sequence[dict[str, Any]], vectors: np.ndarray, scope_books: Iterable[str] | None = None
    ) -> dict[str, int]:
        """把 scope_books（默认为 chunks 涉及的卷）在 collection 中的内容同步成与 chunks 完全一致：
        upsert 全部、删除这些卷中多余的、flush、load。"""
        rows = [
            {
                "chunk_id": c["chunk_id"],
                "vector": v.tolist(),
                "text": c["text"],
                **{f: (c.get(f) or "") if f in ("pov", "chapter_title") else c[f] for f in SCALAR_FIELDS},
            }
            for c, v in zip(chunks, vectors, strict=True)
        ]
        b = self.cfg.milvus_upsert_batch
        for i in range(0, len(rows), b):
            self.client.upsert(self.name, rows[i : i + b])
        scope = set(scope_books) if scope_books is not None else {c["book_id"] for c in chunks}
        stale = sorted(self._all_ids(scope) - {c["chunk_id"] for c in chunks}) if scope else []
        for i in range(0, len(stale), 500):
            self.client.delete(self.name, ids=stale[i : i + 500])
        self.client.flush(self.name)
        self.client.load_collection(self.name)
        return {"upserted": len(rows), "deleted": len(stale), "count": self.count()}

    def count(self, flt: SearchFilter | None = None) -> int:
        res = self.client.query(self.name, filter=(flt or SearchFilter()).to_milvus_expr(), output_fields=["count(*)"])
        return int(res[0]["count(*)"])

    def iter_rows(self, book_ids: Iterable[str]) -> Iterable[dict[str, Any]]:
        """遍历这些卷的全部标量字段与正文（一致性检查用，不取向量）。"""
        expr = SearchFilter(book_ids=tuple(sorted(book_ids))).to_milvus_expr()
        it = self.client.query_iterator(
            self.name, batch_size=1000, filter=expr, output_fields=["chunk_id", *SCALAR_FIELDS, "text"]
        )
        try:
            while batch := it.next():
                yield from batch
        finally:
            it.close()

    def search(self, vector: np.ndarray, top_k: int, expr: str = "") -> list[dict[str, Any]]:
        self.client.load_collection(self.name)
        res = self.client.search(
            self.name,
            data=[np.asarray(vector, dtype=np.float32).tolist()],
            limit=top_k,
            filter=expr,
            output_fields=[*SCALAR_FIELDS, "text"],
            search_params={"metric_type": self.cfg.metric_type, "params": {"ef": max(self.cfg.hnsw_ef, top_k)}},
        )
        return list(res[0])


class VectorRetriever(Retriever):
    name = "vector"

    def __init__(self, store: MilvusChunkStore, embedder: Embedder, base_filter: SearchFilter | None = None):
        self.store = store
        self.embedder = embedder
        self.base_filter = base_filter or SearchFilter()
        self.last_latency_ms: dict[str, float] = {}

    def search(self, query: str, top_k: int, flt: SearchFilter | None = None) -> list[Hit]:
        t0 = time.perf_counter()
        vec = self.embedder.embed_query(query)
        t1 = time.perf_counter()
        raw = self.store.search(vec, top_k, (flt or self.base_filter).to_milvus_expr())
        t2 = time.perf_counter()
        self.last_latency_ms = {"embed": (t1 - t0) * 1000, "search": (t2 - t1) * 1000, "total": (t2 - t0) * 1000}
        hits = []
        for rank, r in enumerate(raw, 1):
            entity = dict(r.get("entity") or {})
            # 主键在结果里的键名随 pymilvus 版本不同：可能是字段名 chunk_id，也可能是 id
            cid = r.get("chunk_id") or entity.get("chunk_id") or r.get("id")
            entity.setdefault("chunk_id", cid)
            hits.append(Hit(rank, str(cid), float(r["distance"]), entity))
        return hits

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "collection": self.store.name,
            "embedding_model": self.embedder.model_name,
            "metric": self.store.cfg.metric_type,
            "hnsw_ef": self.store.cfg.hnsw_ef,
        }
