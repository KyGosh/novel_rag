"""检索过滤条件的统一定义。防剧透（max_reading_order）等条件在这里拼装，Milvus、ES、内存复核共用，
Neo4j（P3）沿用同一个定义。"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SearchFilter:
    book_ids: tuple[str, ...] | None = None
    max_reading_order: int | None = None  # 防剧透：只返回 reading_order <= 该值的内容
    section_types: tuple[str, ...] | None = None
    chunk_types: tuple[str, ...] | None = None
    # 只在这些章节内检索（reading_order 全系列唯一，即章节集合）；与 max_reading_order 同时生效
    reading_orders: tuple[int, ...] | None = None

    def with_max_reading_order(self, value: int | None) -> SearchFilter:
        return replace(self, max_reading_order=value)

    def with_reading_orders(self, values: tuple[int, ...] | None) -> SearchFilter:
        return replace(self, reading_orders=None if values is None else tuple(sorted({int(v) for v in values})))

    def to_milvus_expr(self) -> str:
        """Milvus 标量过滤表达式；没有任何条件时返回空字符串。"""
        parts = []
        if self.book_ids is not None:
            parts.append(f"book_id in {json.dumps(list(self.book_ids), ensure_ascii=False)}")
        if self.max_reading_order is not None:
            parts.append(f"reading_order <= {int(self.max_reading_order)}")
        if self.section_types is not None:
            parts.append(f"section_type in {json.dumps(list(self.section_types), ensure_ascii=False)}")
        if self.chunk_types is not None:
            parts.append(f"chunk_type in {json.dumps(list(self.chunk_types), ensure_ascii=False)}")
        if self.reading_orders is not None:
            parts.append(f"reading_order in {json.dumps([int(x) for x in self.reading_orders])}")
        return " and ".join(parts)

    def to_es_filter(self) -> list[dict]:
        """ES bool 查询的 filter 子句列表（不参与打分）；没有任何条件时返回空列表。"""
        clauses: list[dict] = []
        if self.book_ids is not None:
            clauses.append({"terms": {"book_id": list(self.book_ids)}})
        if self.max_reading_order is not None:
            clauses.append({"range": {"reading_order": {"lte": int(self.max_reading_order)}}})
        if self.section_types is not None:
            clauses.append({"terms": {"section_type": list(self.section_types)}})
        if self.chunk_types is not None:
            clauses.append({"terms": {"chunk_type": list(self.chunk_types)}})
        if self.reading_orders is not None:
            clauses.append({"terms": {"reading_order": [int(x) for x in self.reading_orders]}})
        return clauses

    def accepts(self, fields: dict) -> bool:
        """在内存中判断一条记录是否满足条件（用于 mock 检索器和结果复核）。"""
        if self.book_ids is not None and fields.get("book_id") not in self.book_ids:
            return False
        if self.max_reading_order is not None and fields.get("reading_order", 0) > self.max_reading_order:
            return False
        if self.section_types is not None and fields.get("section_type") not in self.section_types:
            return False
        if self.reading_orders is not None and fields.get("reading_order") not in self.reading_orders:
            return False
        return not (self.chunk_types is not None and fields.get("chunk_type") not in self.chunk_types)
