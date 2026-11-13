"""工具层单测用的假后端：两卷自造小说（不用版权原文）、内存检索器、假摘要检索器与假图谱。"""

from __future__ import annotations

from typing import Any

from novel_rag.graph_retrieval.templates import QueryResult, check_visible
from novel_rag.retrieval.base import Hit, Retriever
from novel_rag.retrieval.expand import ChunkIndex
from novel_rag.retrieval.filters import SearchFilter
from novel_rag.tools.backends import ToolBackends

# (book, chapter_idx, reading_order, title, [段落])
CHAPTERS = [
    ("demo-1", 0, 0, "序章", ["守林人阿青在北岭发现了一只白狐。", "阿青把白狐带回了青石村。"]),
    ("demo-1", 1, 1, "阿青", ["村长柳伯说白狐是山神的使者。", "柳伯的女儿柳月想收养白狐。", "阿青答应了柳月。"]),
    ("demo-1", 2, 2, "柳月", ["柳月给白狐取名雪团。", "雪团在夜里引来了狼群。"]),
    ("demo-2", 0, 3, "黑河", ["多年后，柳月嫁给了黑河城的城主石岩。", "石岩的兄长石峰早年战死。"]),
    ("demo-2", 1, 4, "石岩", ["石岩发现雪团其实是山神本身。", "真相大白：阿青是石峰的私生子。"]),
]
TITLES = {"demo-1": "北岭记", "demo-2": "黑河记"}


def make_chunks() -> list[dict[str, Any]]:
    out, pos = [], {"demo-1": 0, "demo-2": 0}
    for book, chap, ro, title, paras in CHAPTERS:
        for seq, text in enumerate(paras):
            start = pos[book]
            out.append(
                {
                    "chunk_id": f"{book}:{chap:04d}:{seq:04d}", "book_id": book, "chapter_idx": chap, "seq": seq,
                    "reading_order": ro, "chapter_title": title, "section_type": "narrative", "chunk_type": "text",
                    "char_start": start, "char_end": start + len(text), "text": text,
                }
            )  # fmt: skip
            pos[book] = start + len(text) + 1
    return out


class MemRetriever(Retriever):
    """按查询与正文的共同字符数排序；严格按过滤条件（含 chunk_ids / reading_orders）筛选。"""

    name = "mem"

    def __init__(self, chunks: list[dict[str, Any]], leak: bool = False):
        self.chunks = chunks
        self.leak = leak  # True：故意不按阅读进度过滤（测兜底断言）
        self.calls: list[SearchFilter] = []

    def search(self, query: str, top_k: int, flt: SearchFilter | None = None) -> list[Hit]:
        flt = flt or SearchFilter()
        self.calls.append(flt)
        use = flt.with_max_reading_order(None) if self.leak else flt
        q = set(query)
        cands = [c for c in self.chunks if use.accepts(c)]
        cands.sort(key=lambda c: (-len(q & set(c["text"])), c["chunk_id"]))
        return [Hit(i, c["chunk_id"], float(len(q & set(c["text"]))), dict(c)) for i, c in enumerate(cands[:top_k], 1)]


def make_summaries() -> dict[str, dict[str, Any]]:
    out = {}
    for book, chap, ro, title, paras in CHAPTERS:
        key = f"{book}:{chap:04d}"
        out[key] = {
            "chapter_key": key, "book_id": book, "chapter_idx": chap, "chapter_title": title, "reading_order": ro,
            "pov": title if chap else None, "summary": "；".join(paras),
            "events": [{"description": paras[0], "participants": ["阿青"], "chunk_ids": [f"{key}:0000"]}],
        }  # fmt: skip
    return out


class MemSummaryRetriever(Retriever):
    name = "summary"

    def __init__(self, summaries: dict[str, dict[str, Any]]):
        self.summaries = summaries

    def search(self, query: str, top_k: int, flt: SearchFilter | None = None) -> list[Hit]:
        flt = flt or SearchFilter()
        q = set(query)
        recs = [r for r in self.summaries.values() if flt.accepts(r)]
        recs.sort(key=lambda r: (-len(q & set(r["summary"])), r["chapter_key"]))
        return [Hit(i, r["chapter_key"], 1.0, dict(r)) for i, r in enumerate(recs[:top_k], 1)]


class FakeGraph:
    """GraphQueries 的替身：实体首次出现位置、关系证据都带 reading_order，并按阅读进度过滤（leak=True 时不过滤）。"""

    ENTITIES = {
        "demo:person:阿青": {"name": "阿青", "type": "Person", "first_ro": 0, "aliases": []},
        "demo:person:柳月": {"name": "柳月", "type": "Person", "first_ro": 1, "aliases": []},
        "demo:person:石峰": {"name": "石峰", "type": "Person", "first_ro": 3, "aliases": []},
    }
    RELS = [  # (a, b, type, role, [(chunk_id, ro)])
        ("demo:person:柳月", "demo:person:阿青", "ALLY", None, [("demo-1:0001:0002", 1)]),
        ("demo:person:石峰", "demo:person:阿青", "KIN", "father", [("demo-2:0001:0001", 4)]),
    ]

    def __init__(self, leak: bool = False):
        self.leak = leak
        self.calls: list[tuple[str, int | None]] = []

    def _ro(self, max_ro: int | None) -> int:
        return 10**9 if (max_ro is None or self.leak) else max_ro

    def _rel_rows(self, eid: str, max_ro: int | None, other: str | None = None) -> list[dict[str, Any]]:
        ro = self._ro(max_ro)
        rows = []
        for a, b, t, role, ev in self.RELS:
            if eid not in (a, b) or (other and other not in (a, b)):
                continue
            o = b if a == eid else a
            vis = [{"chunk_id": c, "ro": r} for c, r in ev if r <= ro]
            if not vis or self.ENTITIES[o]["first_ro"] > ro:
                continue
            rows.append({
                "type": t, "role": role, "outgoing": a == eid, "a_to_b": a == eid, "other_id": o,
                "other": self.ENTITIES[o]["name"], "other_type": self.ENTITIES[o]["type"], "other_category": "人",
                "n_evidence": len(vis), "first_ro": vis[0]["ro"], "last_ro": vis[-1]["ro"], "evidence": vis,
            })  # fmt: skip
        return rows

    def resolve(self, name: str, max_reading_order: int | None, limit: int | None = None) -> QueryResult:
        self.calls.append(("resolve", max_reading_order))
        ro = self._ro(max_reading_order)
        rows = [
            {"entity_id": eid, "type": e["type"], "name": e["name"], "qualifier": None, "category": "人",
             "aliases": [], "first_ro": e["first_ro"], "n_mentions": 3, "common_word": False, "facts": []}
            for eid, e in self.ENTITIES.items() if name in e["name"] and e["first_ro"] <= ro
        ]  # fmt: skip
        return QueryResult("resolve", {"name": name}, rows)

    def entity_type(self, entity_id: str, max_reading_order: int | None) -> str | None:
        e = self.ENTITIES.get(entity_id)
        return None if e is None or e["first_ro"] > self._ro(max_reading_order) else e["type"]

    def profile(self, entity_id: str, max_reading_order: int | None, limit: int | None = None) -> QueryResult:
        self.calls.append(("profile", max_reading_order))
        e = self.ENTITIES.get(entity_id)
        if e is None or e["first_ro"] > self._ro(max_reading_order):
            return QueryResult("profile", {}, [])
        rels = self._rel_rows(entity_id, max_reading_order)
        row = {"entity_id": entity_id, "type": e["type"], "name": e["name"], "qualifier": None, "category": "人",
               "aliases": [], "first_ro": e["first_ro"], "first_seen": e["first_ro"], "last_seen": e["first_ro"],
               "mentions": 3, "groups": [], "relations": rels}  # fmt: skip
        if max_reading_order is not None and not self.leak:
            check_visible([row], max_reading_order, "profile")
        return QueryResult("profile", {}, [row])

    def between(self, a: str, b: str, max_reading_order: int | None, limit: int | None = None) -> QueryResult:
        self.calls.append(("between", max_reading_order))
        return QueryResult("between", {}, self._rel_rows(a, max_reading_order, b))

    def neighbors(self, entity_id, max_reading_order, *, types=None, other_types=None, limit=None) -> QueryResult:
        from novel_rag.graph_retrieval.templates import QueryError

        if types and any(t not in ("KIN", "ALLY", "SPOUSE") for t in types):
            raise QueryError(f"未知的关系类型：{types}")
        self.calls.append(("neighbors", max_reading_order))
        rows = self._rel_rows(entity_id, max_reading_order)
        return QueryResult("neighbors", {}, [r for r in rows if not other_types or r["other_type"] in other_types])

    def paths(self, a, b, max_reading_order, **kw) -> QueryResult:
        self.calls.append(("paths", max_reading_order))
        self.last_paths_kw = kw
        return QueryResult("paths", {}, [])

    def timeline(self, entity_id, max_reading_order, limit=None) -> QueryResult:
        return QueryResult("timeline", {}, [])

    def title_holders(self, title_id, max_reading_order, limit=None) -> QueryResult:
        return QueryResult("title_holders", {}, [])

    def mentions(self, entity_id, max_reading_order, **kw) -> QueryResult:
        ro = self._ro(max_reading_order)
        name = self.ENTITIES.get(entity_id, {}).get("name", "?")
        rows = [
            {"chunk_id": c["chunk_id"], "ro": c["reading_order"], "count": 1}
            for c in make_chunks()
            if name in c["text"] and c["reading_order"] <= ro
        ]
        return QueryResult("mentions", {}, rows)


def make_backends(*, leak_text: bool = False, leak_graph: bool = False) -> ToolBackends:
    chunks = make_chunks()
    summaries = make_summaries()
    graph = FakeGraph(leak=leak_graph)

    def entity_chunks(eid: str, flt: SearchFilter) -> list[str]:
        name = FakeGraph.ENTITIES.get(eid, {}).get("name")
        return [c["chunk_id"] for c in chunks if name and name in c["text"] and flt.accepts(c)]

    return ToolBackends(
        chunks=ChunkIndex(chunks), book_titles=TITLES, retriever=MemRetriever(chunks, leak=leak_text),
        summary_retriever=MemSummaryRetriever(summaries), summaries=summaries, graph=graph,
        entity_chunks=entity_chunks,
    )  # fmt: skip
