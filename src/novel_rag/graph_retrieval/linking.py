"""查询期实体链接：在问题中用全部实体名和别名做最长匹配，只匹配阅读进度内可见的名字。

名字只有几千个，按长度从长到短逐个子串查字典就够快，不需要 Aho-Corasick 自动机。没有命中时由调用方
退回图谱全文索引的模糊匹配（GraphQueries.resolve）。带与不带间隔号的写法都收录。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from novel_rag.kg.names import NameRules, normalize_name

Q_NAMES = """
MATCH (n:Entity) WHERE n.series_id = $sid AND NOT n:Event
RETURN n.entity_id AS entity_id, n.type AS type, n.name AS name, n.name_visible AS name_visible,
       n.aliases AS aliases, n.alias_visible AS alias_visible, n.first_ro AS first_ro, n.n_mentions AS n_mentions
"""


@dataclass(frozen=True)
class NameEntry:
    entity_id: str
    type: str
    visible_from: int  # 这个写法从哪个 reading_order 起可见
    n_mentions: int


@dataclass
class LinkedSpan:
    text: str
    start: int
    end: int
    candidates: list[NameEntry] = field(default_factory=list)


class EntityLinker:
    def __init__(self, rows: Iterable[dict[str, Any]], rules: NameRules | None = None, min_len: int = 2):
        self.rules = rules or NameRules()
        self.min_len = min_len
        self.names: dict[str, list[NameEntry]] = defaultdict(list)
        for r in rows:
            first = int(r["first_ro"])
            forms = [(r["name"], max(first, int(r.get("name_visible") or first)))]
            forms += [
                (a, max(first, int(v)))
                for a, v in zip(r.get("aliases") or [], r.get("alias_visible") or [], strict=False)
            ]
            for name, vis in forms:
                for f in self._forms(name):
                    if len(f) >= self.min_len and not self.rules.is_generic(f):
                        self.names[f].append(NameEntry(r["entity_id"], r["type"], vis, int(r.get("n_mentions") or 0)))
        self.max_len = max((len(k) for k in self.names), default=0)

    def _forms(self, name: str) -> set[str]:
        n = normalize_name(name)
        return {n, n.replace("·", ""), self.rules.canonical(n), self.rules.canonical(n).replace("·", "")} - {""}

    @classmethod
    def from_neo4j(cls, driver: Any, database: str, series_id: str, rules: NameRules | None = None) -> EntityLinker:
        with driver.session(database=database) as s:
            rows = [r.data() for r in s.run(Q_NAMES, sid=series_id)]
        return cls(rows, rules)

    def visible(self, name: str, max_reading_order: int | None) -> list[NameEntry]:
        best: dict[str, NameEntry] = {}
        for e in self.names.get(name, []):
            visible = max_reading_order is None or e.visible_from <= max_reading_order
            if visible and (e.entity_id not in best or e.visible_from < best[e.entity_id].visible_from):
                best[e.entity_id] = e
        return sorted(best.values(), key=lambda e: (-e.n_mentions, e.entity_id))

    def link(
        self, question: str, max_reading_order: int | None, types: Sequence[str] | None = None
    ) -> list[LinkedSpan]:
        """最长匹配：从左到右，每个位置取最长的可见名字；匹配到的区间不再参与更短的匹配。"""
        # 只统一间隔号（逐字符替换，位置不变）；不做其他改写，返回的区间对应原问题
        q = question.replace("•", "·").replace("・", "·").replace("‧", "·")
        out: list[LinkedSpan] = []
        i = 0
        while i < len(q):
            found = None
            for L in range(min(self.max_len, len(q) - i), self.min_len - 1, -1):
                sub = q[i : i + L]
                cands = self.visible(sub, max_reading_order)
                if types:
                    cands = [c for c in cands if c.type in types]
                if cands:
                    found = LinkedSpan(sub, i, i + L, cands)
                    break
            if found:
                out.append(found)
                i = found.end
            else:
                i += 1
        return out
