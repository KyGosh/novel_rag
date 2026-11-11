"""查询期实体链接：在问题中用全部实体名和别名做最长匹配，只匹配阅读进度内可见的名字。

名字只有几千个，按长度从长到短逐个子串查字典就够快，不需要 Aho-Corasick 自动机。没有命中时由调用方
退回图谱全文索引的模糊匹配（GraphQueries.resolve）。带与不带间隔号的写法都收录。

弱匹配：实体的 weak_names（构图时判定：写法本身是常用词、原文里多数时候不是在指这个名字，如"夏天""淑女"）
命中时，候选标为 weak。固定流程（GraphRoute）不采用弱候选；Agent 的实体解析工具照样返回，并标注"也是常用词"。
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
       n.aliases AS aliases, n.alias_visible AS alias_visible, n.first_ro AS first_ro, n.n_mentions AS n_mentions,
       coalesce(n.weak_names, []) AS weak_names, n.category AS category
"""


@dataclass(frozen=True)
class NameEntry:
    entity_id: str
    type: str
    visible_from: int  # 这个写法从哪个 reading_order 起可见
    n_mentions: int
    weak: bool = False  # 这个写法本身是常用词（弱匹配）


@dataclass
class LinkedSpan:
    text: str
    start: int
    end: int
    candidates: list[NameEntry] = field(default_factory=list)

    @property
    def strong_candidates(self) -> list[NameEntry]:
        return [c for c in self.candidates if not c.weak]

    @property
    def weak(self) -> bool:
        """所有候选都是弱匹配（这段文字更可能是普通词）。"""
        return bool(self.candidates) and not self.strong_candidates


class EntityLinker:
    def __init__(self, rows: Iterable[dict[str, Any]], rules: NameRules | None = None, min_len: int = 2):
        self.rules = rules or NameRules()
        self.min_len = min_len
        self.names: dict[str, list[NameEntry]] = defaultdict(list)
        for r in rows:
            first = int(r["first_ro"])
            weak_names = set(r.get("weak_names") or [])
            forms = [(r["name"], max(first, int(r.get("name_visible") or first)))]
            forms += [
                (a, max(first, int(v)))
                for a, v in zip(r.get("aliases") or [], r.get("alias_visible") or [], strict=False)
            ]
            for name, vis in forms:
                for f in self._forms(name):
                    if len(f) >= self.min_len and not self.rules.is_generic(f):
                        self.names[f].append(
                            NameEntry(r["entity_id"], r["type"], vis, int(r.get("n_mentions") or 0), name in weak_names)
                        )
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
            if not visible:
                continue
            cur = best.get(e.entity_id)
            # 同一实体的多个写法落到同一个字符串上：强匹配优先，其次可见得最早的
            if cur is None or (e.weak, e.visible_from) < (cur.weak, cur.visible_from):
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
