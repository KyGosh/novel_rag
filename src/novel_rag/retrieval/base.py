"""检索器接口。向量检索（P1）、BM25 与 RRF 融合（P2）、图谱召回（P4）都实现同一个接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from novel_rag.retrieval.filters import SearchFilter


@dataclass(frozen=True)
class Hit:
    rank: int  # 从 1 开始
    chunk_id: str
    score: float
    fields: dict[str, Any] = field(default_factory=dict)


class Retriever(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, top_k: int, flt: SearchFilter | None = None) -> list[Hit]: ...

    def describe(self) -> dict[str, Any]:
        """写进实验记录的检索器参数。"""
        return {"name": self.name}
