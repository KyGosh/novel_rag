"""Elasticsearch 中的 chunk 全文索引与 BM25 检索器。

索引命名为 chunks_{chunk 版本}_{分析方案版本}：正文字段 `text` 用 IK（建索引 ik_max_word、查询 ik_smart），
另有子字段 `text.bigram`（standard 分词 + cjk_bigram），作为人名被 IK 切碎时的通用补救——不依赖任何书籍知识。
"只用 IK"与"IK + bigram"是查询时选择字段，同一个索引就能做消融；改分词方案（如 P3 回灌人名词典）就升级
ANALYSIS_VERSION，建新索引。

导入以 chunk_id 为文档 _id 覆盖写入，并删除同一批卷中多余的文档，所以重复导入结果不变；权威数据仍是 chunks.jsonl。
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any

from novel_rag.config import ESConfig, RetrievalConfig
from novel_rag.retrieval.base import Hit, Retriever
from novel_rag.retrieval.filters import SearchFilter

ANALYSIS_VERSION = "es1"
ES_FIELDS = (
    "chunk_id",
    "book_id",
    "chapter_idx",
    "seq",
    "char_start",
    "char_end",
    "reading_order",
    "section_type",
    "chunk_type",
    "pov",
    "chapter_title",
    "text",
)
BM25_SCHEMES: dict[str, tuple[str, ...]] = {"ik": ("text",), "ik_bigram": ("text", "text.bigram")}


def index_name(chunk_version: str, analysis_version: str = ANALYSIS_VERSION) -> str:
    return f"chunks_{chunk_version}_{analysis_version}"


def index_body(description: str = "", analysis_version: str = ANALYSIS_VERSION) -> dict[str, Any]:
    """索引 settings + mappings。字段与 Milvus 的标量字段一致，额外存字符区间。
    es2 起多一个 entity_ids（chunk 提及的实体 ID，来自图谱的提及记录），可用于"只在提到某人的段落里检索"。"""
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "cjk_bigram_std": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["cjk_width", "lowercase", "cjk_bigram"],
                    }
                }
            },
        },
        "mappings": {
            "_meta": {"description": description, "analysis_version": analysis_version},
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "book_id": {"type": "keyword"},
                "chapter_idx": {"type": "integer"},
                "seq": {"type": "integer"},
                "char_start": {"type": "integer"},
                "char_end": {"type": "integer"},
                "reading_order": {"type": "integer"},
                "section_type": {"type": "keyword"},
                "chunk_type": {"type": "keyword"},
                "pov": {"type": "keyword"},
                "chapter_title": {"type": "keyword"},
                "text": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_smart",
                    "fields": {"bigram": {"type": "text", "analyzer": "cjk_bigram_std"}},
                },
                **({} if analysis_version == "es1" else {"entity_ids": {"type": "keyword"}}),
            },
        },
    }


def to_doc(chunk: dict[str, Any]) -> dict[str, Any]:
    doc = {f: (chunk.get(f) or "") if f in ("pov", "chapter_title") else chunk[f] for f in ES_FIELDS}
    if "entity_ids" in chunk:
        doc["entity_ids"] = list(chunk["entity_ids"])
    return doc


class ESChunkStore:
    def __init__(self, client: Any, name: str, cfg: RetrievalConfig, analysis_version: str = ANALYSIS_VERSION):
        self.client = client
        self.name = name
        self.cfg = cfg
        self.analysis_version = analysis_version

    @classmethod
    def connect(
        cls, ecfg: ESConfig, name: str, cfg: RetrievalConfig, analysis_version: str = ANALYSIS_VERSION
    ) -> ESChunkStore:
        from elasticsearch import Elasticsearch

        return cls(Elasticsearch(ecfg.url, request_timeout=ecfg.request_timeout), name, cfg, analysis_version)

    def exists(self) -> bool:
        return bool(self.client.indices.exists(index=self.name))

    def ensure_index(self, description: str = "") -> None:
        if self.exists():
            return
        body = index_body(description, self.analysis_version)
        self.client.indices.create(index=self.name, settings=body["settings"], mappings=body["mappings"])

    def _scan(self, book_ids: Iterable[str], *, source: bool) -> Iterable[dict[str, Any]]:
        """按 chunk_id 顺序遍历这些卷的全部文档（search_after 分页）。"""
        after = None
        while True:
            kwargs: dict[str, Any] = {
                "index": self.name,
                "query": {"terms": {"book_id": sorted(book_ids)}},
                "size": 2000,
                "sort": [{"chunk_id": "asc"}],
                "source": source,
            }
            if after is not None:
                kwargs["search_after"] = after
            hits = self.client.search(**kwargs)["hits"]["hits"]
            if not hits:
                return
            yield from hits
            after = hits[-1]["sort"]

    def _bulk(self, actions: Iterable[dict[str, Any]]) -> tuple[int, list[Any]]:
        from elasticsearch import helpers

        return helpers.bulk(self.client, actions, chunk_size=self.cfg.es_bulk_batch, raise_on_error=False)

    def sync(self, chunks: Sequence[dict[str, Any]], scope_books: Iterable[str] | None = None) -> dict[str, int]:
        """把 scope_books（默认为 chunks 涉及的卷）在索引中的内容同步成与 chunks 完全一致：
        覆盖写入、删除多余、refresh。"""
        scope = set(scope_books) if scope_books is not None else {c["book_id"] for c in chunks}
        ok, errors = self._bulk(
            {"_op_type": "index", "_index": self.name, "_id": c["chunk_id"], "_source": to_doc(c)} for c in chunks
        )
        if errors:
            raise RuntimeError(f"ES 批量写入有 {len(errors)} 条失败，例：{errors[:2]}")
        keep = {c["chunk_id"] for c in chunks}
        stale = sorted({h["_id"] for h in self._scan(scope, source=False)} - keep) if scope else []
        if stale:
            _, errors = self._bulk({"_op_type": "delete", "_index": self.name, "_id": i} for i in stale)
            if errors:
                raise RuntimeError(f"ES 删除多余文档有 {len(errors)} 条失败，例：{errors[:2]}")
        self.client.indices.refresh(index=self.name)
        # 覆盖写入会留下已删除文档，它们仍计入 BM25 的文档频率；合并成一个段后打分只取决于当前内容，
        # 同一份 chunk 无论重建多少次分数都一样（评测可复现）
        self.client.indices.forcemerge(index=self.name, max_num_segments=1)
        self.client.indices.refresh(index=self.name)
        return {"indexed": int(ok), "deleted": len(stale), "count": self.count()}

    def count(self, flt: SearchFilter | None = None) -> int:
        clauses = (flt or SearchFilter()).to_es_filter()
        query = {"bool": {"filter": clauses}} if clauses else {"match_all": {}}
        return int(self.client.count(index=self.name, query=query)["count"])

    def iter_docs(self, book_ids: Iterable[str]) -> Iterable[dict[str, Any]]:
        """按 chunk_id 顺序遍历这些卷的全部文档（一致性检查用）。"""
        return (h["_source"] for h in self._scan(book_ids, source=True))

    def search(self, query: str, fields: Sequence[str], top_k: int, flt: SearchFilter) -> list[dict[str, Any]]:
        should = [{"match": {f: {"query": query, "boost": self.cfg.bm25_field_boosts.get(f, 1.0)}}} for f in fields]
        body = {"bool": {"should": should, "minimum_should_match": 1, "filter": flt.to_es_filter()}}
        # 同分时按 chunk_id 排，结果顺序确定；按 _score 排序时 ES 照常返回分数
        sort = [{"_score": "desc"}, {"chunk_id": "asc"}]
        res = self.client.search(index=self.name, query=body, size=top_k, sort=sort)
        return list(res["hits"]["hits"])


class BM25Retriever(Retriever):
    def __init__(self, store: ESChunkStore, scheme: str = "ik", base_filter: SearchFilter | None = None):
        if scheme not in BM25_SCHEMES:
            raise ValueError(f"未知的 BM25 方案：{scheme}（可选 {sorted(BM25_SCHEMES)}）")
        self.store = store
        self.scheme = scheme
        self.fields = BM25_SCHEMES[scheme]
        self.name = f"bm25_{scheme}"
        self.base_filter = base_filter or SearchFilter()
        self.last_latency_ms: dict[str, float] = {}

    def search(self, query: str, top_k: int, flt: SearchFilter | None = None) -> list[Hit]:
        t0 = time.perf_counter()
        raw = self.store.search(query, self.fields, top_k, flt or self.base_filter)
        self.last_latency_ms = {"search": (time.perf_counter() - t0) * 1000}
        self.last_latency_ms["total"] = self.last_latency_ms["search"]
        return [Hit(rank, h["_id"], float(h["_score"]), dict(h["_source"])) for rank, h in enumerate(raw, 1)]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "index": self.store.name,
            "fields": list(self.fields),
            "boosts": {f: self.store.cfg.bm25_field_boosts.get(f, 1.0) for f in self.fields},
        }
