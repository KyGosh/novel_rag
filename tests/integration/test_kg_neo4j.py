"""集成测试：真实 Neo4j 上的图谱写入与模板查询，重点覆盖防剧透（max_reading_order）过滤。

用自拟的小图，series_id 为 kgtest（只删除 / 写入这个系列的节点，不影响真实图谱）。
运行：uv run pytest -m integration tests/integration/test_kg_neo4j.py
"""

import pytest

from novel_rag.config import get_settings
from novel_rag.graph_retrieval.linking import EntityLinker
from novel_rag.graph_retrieval.templates import GraphQueries
from novel_rag.kg.neo4j_store import Neo4jKGStore

pytestmark = pytest.mark.integration
SID = "kgtest"


def ent(name, type_="Person", first=1, aliases=(), name_visible=None, n=10, category=None, weak=()):
    return {
        "entity_id": f"{SID}:{type_.lower()}:{name}", "type": type_, "name": name,
        "name_visible_from": name_visible if name_visible is not None else first,
        "first_reading_order": first, "last_reading_order": 9, "n_mentions": n, "n_chunks": 3,
        "aliases": [{"alias": a, "visible_from": v, "first_chunk": "", "link": "x"} for a, v in aliases],
        "groups": [], "category": category if category else ("人" if type_ == "Person" else None), "qualifier": None,
        "weak_names": list(weak),
    }  # fmt: skip


def rel(s, t, o, ros, role=None):
    return {
        "subject": f"{SID}:person:{s}" if ":" not in s else s, "type": t, "role": role,
        "object": f"{SID}:person:{o}" if ":" not in o else o,
        "first_reading_order": min(ros), "last_reading_order": max(ros), "n_evidence": len(ros),
        "evidence": [{"chunk_id": f"kt-1:{ro:04d}:0000", "reading_order": ro, "start": 0, "end": 5, "book_id": "kt-1"}
                     for ro in ros],
    }  # fmt: skip


CHUNKS = [
    {"chunk_id": f"kt-1:{ro:04d}:0000", "book_id": "kt-1", "chapter_idx": ro, "seq": 0, "reading_order": ro,
     "section_type": "narrative", "chapter_title": f"第{ro}章", "book_title": "测试卷"}
    for ro in range(1, 10)
]  # fmt: skip
TITLE = f"{SID}:title:灰岩城主"
TABLES = {
    "chunks": CHUNKS,
    "entities": [
        ent("甲·铁杉", aliases=[("甲", 1), ("灰狼", 6)]),
        ent("乙·铁杉"),
        ent("丙·白桦", first=5),  # 第 5 章才出场
        ent("灰岩城主", "Title"),
        ent("灰影", first=2, category="动物", weak=["灰影"]),  # 有名字的动物；名字本身当作常用词
    ],
    "events": [
        {
            "event_id": "kt-1:0002:e01",
            "description": "甲与乙在城门会面",
            "book_id": "kt-1",
            "chapter_idx": 2,
            "chapter_title": "第2章",
            "reading_order": 2,
            "chunk_ids": ["kt-1:0002:0000"],
            "participants": [],
            "location": None,
        },
        {
            "event_id": "kt-1:0007:e01",
            "description": "甲与丙决裂",
            "book_id": "kt-1",
            "chapter_idx": 7,
            "chapter_title": "第7章",
            "reading_order": 7,
            "chunk_ids": ["kt-1:0007:0000"],
            "participants": [],
            "location": None,
        },
    ],  # fmt: skip
    "relations": [
        rel("甲·铁杉", "ALLY", "乙·铁杉", [1, 3]),
        rel("甲·铁杉", "ENEMY", "乙·铁杉", [8]),  # 后来反目：两条关系共存
        rel("甲·铁杉", "KIN", "丙·白桦", [6, 7], role="father"),
        rel(f"{SID}:person:甲·铁杉", "HOLDS_TITLE", TITLE, [2]),
        rel(f"{SID}:person:乙·铁杉", "HOLDS_TITLE", TITLE, [9]),
        rel("灰影", "COMPANION", "甲·铁杉", [6]),
        rel(f"{SID}:person:甲·铁杉", "PARTICIPATES", "kt-1:0002:e01", [2]),
        rel(f"{SID}:person:甲·铁杉", "PARTICIPATES", "kt-1:0007:e01", [7]),
    ],
    "mentions": [
        {"chunk_id": f"kt-1:{ro:04d}:0000", "entity_id": f"{SID}:person:甲·铁杉", "count": ro} for ro in range(1, 10)
    ],
}


@pytest.fixture(scope="module")
def graph():
    s = get_settings()
    store = Neo4jKGStore.connect(s.neo4j)
    counts = store.replace(SID, TABLES, {"kg_version": "test"}, batch=100)
    assert counts["entities"] == 5 and counts["events"] == 2 and counts["relations"] == 8 and counts["next"] == 0
    g = GraphQueries(store.driver, s.neo4j.database, s.kg, SID)
    yield g, store
    store.delete_series(SID)
    store.close()


A, B, C = (f"{SID}:person:{x}" for x in ("甲·铁杉", "乙·铁杉", "丙·白桦"))


def test_alias_visibility(graph):
    g, _ = graph
    assert [r["entity_id"] for r in g.resolve("灰狼", 9).rows] == [A]
    assert g.resolve("灰狼", 5).rows == [] or all("灰狼" not in r["aliases"] for r in g.resolve("灰狼", 5).rows)
    assert "灰狼" not in g.resolve("甲", 5).rows[0]["aliases"]


def test_relations_filtered_by_reading_order(graph):
    g, _ = graph
    at4 = {(r["type"], r["other_id"]): r for r in g.neighbors(A, 4).rows}
    assert set(at4) == {("ALLY", B), ("HOLDS_TITLE", TITLE)}  # 敌对（ro 8）和丙（ro 5 出场）都不可见
    assert at4[("ALLY", B)]["n_evidence"] == 2 and at4[("ALLY", B)]["last_ro"] == 3
    at9 = {(r["type"], r["other_id"]) for r in g.neighbors(A, 9).rows}
    assert {("ALLY", B), ("ENEMY", B), ("KIN", C)} <= at9
    kin6 = next(r for r in g.neighbors(A, 6).rows if r["type"] == "KIN")
    assert kin6["n_evidence"] == 1 and [e["ro"] for e in kin6["evidence"]] == [6]  # 证据也按进度过滤
    assert [r["type"] for r in g.between(A, B, 5).rows] == ["ALLY"]


def test_hidden_entities_and_paths(graph):
    g, _ = graph
    assert g.profile(C, 4).rows == []
    assert g.paths(B, C, 4).rows == []  # 丙尚未出场
    p = g.paths(B, C, 9).rows
    assert p and [n["id"] for n in p[0]["nodes"]] in ([B, A, C], [B, A, C][::-1])


def test_timeline_mentions_titles(graph):
    g, _ = graph
    tl = g.timeline(A, 5).rows
    assert all(x["ro"] <= 5 for x in tl) and any(x["kind"] == "event" for x in tl)
    assert [x["ro"] for x in g.mentions(A, 4).rows] == [1, 2, 3, 4]
    assert [x["ro"] for x in g.mentions(A, 9, by_count=True, limit=2).rows] == [9, 8]
    assert [x["holder_id"] for x in g.title_holders(TITLE, 5).rows] == [A]
    assert [x["holder_id"] for x in g.title_holders(TITLE, 9).rows] == [A, B]
    assert g.where("kt-1:0003:0000") == "测试卷 · 第3章"


def test_linker_from_neo4j(graph):
    g, store = graph
    lk = EntityLinker.from_neo4j(store.driver, get_settings().neo4j.database, SID)
    assert [s.text for s in lk.link("灰狼和丙·白桦", 4)] == []
    assert [s.text for s in lk.link("灰狼和丙·白桦", 9)] == ["灰狼", "丙·白桦"]


def test_categories_companion_and_weak_names(graph):
    g, store = graph
    wolf = f"{SID}:person:灰影"
    r = g.resolve("灰影", 9).rows[0]
    assert r["entity_id"] == wolf and r["category"] == "动物" and r["common_word"] is True
    assert g.resolve("甲·铁杉", 9).rows[0]["common_word"] is False
    assert g.profile(wolf, 9).rows[0]["category"] == "动物"
    # 伙伴关系同样按阅读进度过滤（证据在 ro 6）
    assert not [x for x in g.neighbors(A, 5).rows if x["type"] == "COMPANION"]
    [comp] = [x for x in g.neighbors(A, 9).rows if x["type"] == "COMPANION"]
    assert comp["other_id"] == wolf and comp["other_category"] == "动物" and comp["outgoing"] is False
    lk = EntityLinker.from_neo4j(store.driver, get_settings().neo4j.database, SID)
    [sp] = lk.link("灰影在哪", 9)
    assert sp.weak and sp.candidates[0].entity_id == wolf
