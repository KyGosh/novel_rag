"""检索过滤条件的统一定义。防剧透（max_reading_order）等条件在这里拼装，ES（P2）、Neo4j（P3）沿用同一个定义。"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchFilter:
    book_ids: tuple[str, ...] | None = None
    max_reading_order: int | None = None  # 防剧透：只返回 reading_order <= 该值的内容
    section_types: tuple[str, ...] | None = None
    chunk_types: tuple[str, ...] | None = None

    def with_max_reading_order(self, value: int | None) -> SearchFilter:
        return SearchFilter(self.book_ids, value, self.section_types, self.chunk_types)

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
        return " and ".join(parts)

    def accepts(self, fields: dict) -> bool:
        """在内存中判断一条记录是否满足条件（用于 mock 检索器和结果复核）。"""
        if self.book_ids is not None and fields.get("book_id") not in self.book_ids:
            return False
        if self.max_reading_order is not None and fields.get("reading_order", 0) > self.max_reading_order:
            return False
        if self.section_types is not None and fields.get("section_type") not in self.section_types:
            return False
        return not (self.chunk_types is not None and fields.get("chunk_type") not in self.chunk_types)
