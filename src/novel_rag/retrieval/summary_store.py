"""章节摘要索引：ES（IK BM25）与 Milvus（bge-m3）各一份，按章节检索摘要。

- 文档键为章节键 `{book_id}:{chapter_idx:04d}`，字段带 book_id / chapter_idx / reading_order / pov / section_type，
  防剧透与 chunk 索引一样用 SearchFilter（reading_order <= max_reading_order）在每一路过滤，融合后再复核。
- 检索文本 = 章节标题 + 摘要 + 关键事件描述；摘要不进 chunk 索引（它在 text.txt 中没有字符位置）。
- 索引名带摘要版本：summaries_{摘要版本}_{分析方案版本}，collection 名 summaries_{模型简称}_{维度}_{摘要版本}。
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from novel_rag.config import ESConfig, MilvusConfig, RetrievalConfig
from novel_rag.llm.base import Embedder
from novel_rag.retrieval.base import Hit, Retriever
from novel_rag.retrieval.es_store import ANALYSIS_VERSION
from novel_rag.retrieval.filters import SearchFilter
from novel_rag.retrieval.fusion import rrf_fuse
from novel_rag.retrieval.hybrid import check_filter

SUMMARY_FIELDS = ("chapter_key", "book_id", "chapter_idx", "reading_order", "section_type", "pov", "chapter_title")


def summary_text(rec: dict[str, Any]) -> str:
    events = "\n".join(e["description"] for e in rec.get("events", []))
    return f"{rec['chapter_title']}\n{rec['summary']}\n{events}".strip()


def summary_doc(rec: dict[str, Any]) -> dict[str, Any]:
    doc = {f: (rec.get(f) or "") if f in ("pov", "chapter_title") else rec[f] for f in SUMMARY_FIELDS}
    doc["characters"] = list(rec.get("characters", []))
    doc["text"] = summary_text(rec)
    return doc


def es_index_name(summary_version: str, analysis_version: str = ANALYSIS_VERSION) -> str:
    return f"summaries_{summary_version}_{analysis_version}"


def milvus_collection_name(model_name: str, dim: int, summary_version: str) -> str:
    short = re.sub(r"[^a-z0-9]", "", model_name.rsplit("/", 1)[-1].lower())
    return f"summaries_{short}_{dim}_{summary_version}"


class ESSummaryStore:
    def __init__(self, client: Any, name: str):
        self.client = client
        self.name = name

    @classmethod
    def connect(cls, ecfg: ESConfig, name: str) -> ESSummaryStore:
        from elasticsearch import Elasticsearch

        return cls(Elasticsearch(ecfg.url, request_timeout=ecfg.request_timeout), name)

    def exists(self) -> bool:
        return bool(self.client.indices.exists(index=self.name))

    def rebuild(self, records: Sequence[dict[str, Any]], description: str = "") -> int:
        """摘要只有几百条：删掉重建，结果与输入完全一致。"""
        from elasticsearch import helpers

        if self.exists():
            self.client.indices.delete(index=self.name)
        mappings = {
            "_meta": {"description": description},
            "dynamic": "strict",
            "properties": {
                "chapter_key": {"type": "keyword"},
                "book_id": {"type": "keyword"},
                "chapter_idx": {"type": "integer"},
                "reading_order": {"type": "integer"},
                "section_type": {"type": "keyword"},
                "pov": {"type": "keyword"},
                "chapter_title": {"type": "keyword"},
                "characters": {"type": "keyword"},
                "text": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
            },
        }
        self.client.indices.create(
            index=self.name, settings={"number_of_shards": 1, "number_of_replicas": 0}, mappings=mappings
        )
        docs = [summary_doc(r) for r in records]
        helpers.bulk(self.client, ({"_index": self.name, "_id": d["chapter_key"], "_source": d} for d in docs))
        self.client.indices.refresh(index=self.name)
        self.client.indices.forcemerge(index=self.name, max_num_segments=1)
        self.client.indices.refresh(index=self.name)
        return int(self.client.count(index=self.name)["count"])

    def search(self, query: str, top_k: int, flt: SearchFilter) -> list[Hit]:
        body = {"bool": {"must": [{"match": {"text": query}}], "filter": flt.to_es_filter()}}
        res = self.client.search(
            index=self.name, query=body, size=top_k, sort=[{"_score": "desc"}, {"chapter_key": "asc"}]
        )
        return [
            Hit(rank, h["_id"], float(h["_score"]), dict(h["_source"])) for rank, h in enumerate(res["hits"]["hits"], 1)
        ]


class MilvusSummaryStore:
    def __init__(self, client: Any, name: str, dim: int, cfg: RetrievalConfig):
        self.client = client
        self.name = name
        self.dim = dim
        self.cfg = cfg

    @classmethod
    def connect(cls, mcfg: MilvusConfig, name: str, dim: int, cfg: RetrievalConfig) -> MilvusSummaryStore:
        from pymilvus import MilvusClient

        return cls(
            MilvusClient(uri=mcfg.uri, token=mcfg.token.get_secret_value(), db_name=mcfg.db_name), name, dim, cfg
        )

    def exists(self) -> bool:
        return bool(self.client.has_collection(self.name))

    def rebuild(self, records: Sequence[dict[str, Any]], vectors: np.ndarray, description: str = "") -> int:
        from pymilvus import DataType

        if self.exists():
            self.client.drop_collection(self.name)
        schema = self.client.create_schema(auto_id=False, description=description)
        schema.add_field("chapter_key", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("book_id", DataType.VARCHAR, max_length=64)
        schema.add_field("chapter_idx", DataType.INT32)
        schema.add_field("reading_order", DataType.INT32)
        schema.add_field("section_type", DataType.VARCHAR, max_length=32)
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
        rows = []
        for r, v in zip(records, vectors, strict=True):
            d = summary_doc(r)
            rows.append({**{f: d[f] for f in SUMMARY_FIELDS}, "text": d["text"][:8000], "vector": v.tolist()})
        for i in range(0, len(rows), 256):
            self.client.insert(self.name, rows[i : i + 256])
        self.client.flush(self.name)
        self.client.load_collection(self.name)
        res = self.client.query(self.name, filter="", output_fields=["count(*)"])
        return int(res[0]["count(*)"])

    def search(self, vector: np.ndarray, top_k: int, flt: SearchFilter) -> list[Hit]:
        self.client.load_collection(self.name)
        res = self.client.search(
            self.name,
            data=[np.asarray(vector, dtype=np.float32).tolist()],
            limit=top_k,
            filter=flt.to_milvus_expr(),
            output_fields=[*SUMMARY_FIELDS, "text"],
            search_params={"metric_type": self.cfg.metric_type, "params": {"ef": max(self.cfg.hnsw_ef, top_k)}},
        )
        hits = []
        for rank, r in enumerate(res[0], 1):
            entity = dict(r.get("entity") or {})
            key = r.get("chapter_key") or entity.get("chapter_key") or r.get("id")
            entity.setdefault("chapter_key", key)
            hits.append(Hit(rank, str(key), float(r["distance"]), entity))
        return hits


class SummaryRetriever(Retriever):
    """摘要检索：BM25 与向量两路按章节 RRF 融合。Hit.chunk_id 是章节键，fields 带 reading_order 等。"""

    name = "summary"

    def __init__(
        self,
        es: ESSummaryStore | None,
        milvus: MilvusSummaryStore | None,
        embedder: Embedder | None,
        *,
        bm25_top_k: int,
        vector_top_k: int,
        rrf_k: int,
        base_filter: SearchFilter | None = None,
    ):
        if es is None and milvus is None:
            raise ValueError("摘要检索至少需要一路")
        self.es, self.milvus, self.embedder = es, milvus, embedder
        self.bm25_top_k, self.vector_top_k, self.rrf_k = bm25_top_k, vector_top_k, rrf_k
        self.base_filter = base_filter or SearchFilter()
        self.last_latency_ms: dict[str, float] = {}

    def search(self, query: str, top_k: int, flt: SearchFilter | None = None) -> list[Hit]:
        # 摘要索引里没有 chunk_type 字段；其余条件（卷、阅读进度、section_type、章节集合）照常生效
        flt = flt or self.base_filter
        flt = SearchFilter(flt.book_ids, flt.max_reading_order, flt.section_types, None, flt.reading_orders)
        t0 = time.perf_counter()
        routes: dict[str, list[Hit]] = {}
        if self.es is not None:
            routes["bm25"] = self.es.search(query, self.bm25_top_k, flt)
        if self.milvus is not None and self.embedder is not None:
            routes["vector"] = self.milvus.search(self.embedder.embed_query(query), self.vector_top_k, flt)
        for name, hits in routes.items():
            check_filter(hits, flt, f"summary.{name}")
        fused = rrf_fuse(routes, k=self.rrf_k, top_k=top_k)
        check_filter(fused, flt, "summary")
        self.last_latency_ms = {"total": (time.perf_counter() - t0) * 1000}
        return fused

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "es_index": self.es.name if self.es else None,
            "milvus_collection": self.milvus.name if self.milvus else None,
            "bm25_top_k": self.bm25_top_k,
            "vector_top_k": self.vector_top_k,
            "rrf_k": self.rrf_k,
        }
