"""图谱查询模板（不连 Neo4j）：模板只读、强制 LIMIT、全部带 $max_ro；参数校验；结果复核；实体链接；渲染上限。"""

import re

import pytest

from novel_rag.config import KGConfig
from novel_rag.graph_retrieval.linking import EntityLinker
from novel_rag.graph_retrieval.render import render
from novel_rag.graph_retrieval.templates import (
    ALL_TEMPLATES,
    Q_CHUNK_META,
    GraphQueries,
    QueryError,
    QueryResult,
    SpoilerViolation,
    check_visible,
    lucene_escape,
)

WRITE = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH)\b|apoc\.", re.I)


@pytest.mark.parametrize("q", ALL_TEMPLATES)
def test_templates_are_read_only_limited_and_spoiler_filtered(q):
    assert not WRITE.search(q)
    assert "$max_ro" in q
    # 单实体查找（Q_ENTITY / Q_TOP_FACTS 按 ID，结果条数有界）之外的模板都要有 LIMIT 参数
    if "UNWIND $ids" not in q and "MATCH (n:Entity {entity_id: $eid})\nWHERE n.series_id" not in q:
        assert "LIMIT $limit" in q


def test_templates_fully_formatted():
    """模板在导入时就格式化完毕：不残留 Python 占位符，用户输入只能通过 $参数 进入。"""
    for q in (*ALL_TEMPLATES, Q_CHUNK_META):
        assert not re.search(r"\{(n|r|m|x|p|hops|disp|ev)\}", q)
        assert "%s" not in q and "{{" not in q


class FakeTx:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))

        class Rec:
            def __init__(self, d):
                self.d = d

            def data(self):
                return self.d

        return [Rec(r) for r in self.rows]


class FakeSession:
    def __init__(self, tx):
        self.tx = tx

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute_read(self, work):
        return work(self.tx)


class FakeDriver:
    def __init__(self, rows):
        self.tx = FakeTx(rows)
        self.kwargs = None

    def session(self, **kw):
        self.kwargs = kw
        return FakeSession(self.tx)


def test_limit_clamp_and_params():
    drv = FakeDriver([])
    g = GraphQueries(drv, "neo4j", KGConfig(query_max_limit=50, default_limit=20), "demo")
    g.neighbors("demo:person:甲", 7, limit=10_000)
    q, params = drv.tx.calls[-1]
    assert params["limit"] == 50 and params["max_ro"] == 7 and params["eid"] == "demo:person:甲"
    assert drv.kwargs["default_access_mode"] == "READ"
    g.neighbors("demo:person:甲", None, limit=0)
    assert drv.tx.calls[-1][1]["limit"] == 1 and drv.tx.calls[-1][1]["max_ro"] >= 10**9
    with pytest.raises(QueryError):
        g.neighbors("x", 3, types=["NOT_A_TYPE"])
    with pytest.raises(QueryError):
        g.paths("a", "b", 3, max_hops=5)


def test_result_recheck_raises_on_spoiler():
    rows = [{"type": "KIN", "first_ro": 3, "evidence": [{"chunk_id": "c1", "ro": 3}, {"chunk_id": "c2", "ro": 9}]}]
    with pytest.raises(SpoilerViolation):
        check_visible(rows, 5, "neighbors")
    check_visible(rows, 9, "neighbors")
    g = GraphQueries(FakeDriver(rows), "neo4j", KGConfig(), "demo")
    with pytest.raises(SpoilerViolation):
        g.neighbors("x", 5)


def test_lucene_escape():
    assert lucene_escape('阿岚"~*') == '阿岚\\"\\~\\*'
    assert lucene_escape("a:b/c") == "a\\:b\\/c"


ROWS = [
    {"entity_id": "demo:person:阿岚·铁杉", "type": "Person", "name": "阿岚·铁杉", "name_visible": 1,
     "aliases": ["阿岚", "老灰狼"], "alias_visible": [1, 6], "first_ro": 1, "n_mentions": 30},
    {"entity_id": "demo:person:阿岚·白桦", "type": "Person", "name": "阿岚·白桦", "name_visible": 4,
     "aliases": ["阿岚"], "alias_visible": [4], "first_ro": 4, "n_mentions": 5},
    {"entity_id": "demo:location:灰岩堡", "type": "Location", "name": "灰岩堡", "name_visible": 1,
     "aliases": [], "alias_visible": [], "first_ro": 1, "n_mentions": 9},
]  # fmt: skip


def test_linker_longest_match_and_visibility():
    lk = EntityLinker(ROWS)
    spans = lk.link("阿岚·铁杉和老灰狼在灰岩堡", 10)
    assert [s.text for s in spans] == ["阿岚·铁杉", "老灰狼", "灰岩堡"]
    assert spans[1].candidates[0].entity_id == "demo:person:阿岚·铁杉"
    # 别名揭示之前（ro 5）："老灰狼"不可见
    assert [s.text for s in lk.link("老灰狼在灰岩堡", 5)] == ["灰岩堡"]
    # 同名：两个候选按提及数排序；白桦在 ro 4 之前不可见
    assert [c.entity_id for c in lk.link("阿岚来了", 10)[0].candidates] == [
        "demo:person:阿岚·铁杉",
        "demo:person:阿岚·白桦",
    ]
    assert [c.entity_id for c in lk.link("阿岚来了", 3)[0].candidates] == ["demo:person:阿岚·铁杉"]
    assert [s.text for s in lk.link("阿岚铁杉", 10)] == ["阿岚铁杉"]  # 不带间隔号的写法
    assert lk.link("阿岚•铁杉", 10)[0].candidates[0].entity_id == "demo:person:阿岚·铁杉"


def test_render_budget_and_evidence():
    rows = [
        {"type": "KIN", "role": "father", "outgoing": True, "other": f"孩子{i}", "other_id": f"demo:person:孩子{i}",
         "n_evidence": 2, "evidence": [{"chunk_id": "demo-1:0001:0000", "ro": 1}]}
        for i in range(50)
    ]  # fmt: skip
    text = render(QueryResult("neighbors", {}, rows, truncated=True), lambda c: "卷一 · 米拉", 600)
    assert len(text) <= 600 and text.endswith("（结果过长，已截断）")
    assert "demo-1:0001:0000（卷一 · 米拉）" in text
    assert "没有找到" in render(QueryResult("profile", {}, []), lambda c: c, 100)


def test_linker_min_mentions_threshold():
    lk = EntityLinker(ROWS, min_mentions=6)
    # 白桦只有 5 次提及：不参与链接
    assert [c.entity_id for c in lk.link("阿岚来了", 10)[0].candidates] == ["demo:person:阿岚·铁杉"]
    assert lk.link("阿岚·白桦", 10)[0].text != "阿岚·白桦"


def test_paths_node_type_params():
    drv = FakeDriver([])
    g = GraphQueries(drv, "neo4j", KGConfig(), "demo")
    g.paths("a", "b", 5)
    p = drv.tx.calls[-1][1]
    assert p["node_types"] is None and p["exclude_node_types"] is None
    g.paths("a", "b", 5, node_types=("Person",), exclude_node_types=("Event",))
    q, p = drv.tx.calls[-1]
    assert p["node_types"] == ["Person"] and p["exclude_node_types"] == ["Event"]
    assert "$node_types" in q and "$exclude_node_types" in q
