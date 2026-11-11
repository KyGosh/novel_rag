"""图谱查询模板（P4 图谱工具的底座）：全部参数化、只读、强制 LIMIT、带超时，全部带 max_reading_order。

防剧透规则（与 chunk 检索一致，是一等参数）：
- 实体：first_ro <= max_ro 才可见；规范名在 name_visible 之前不可用，改用可见的别名称呼。
- 别名：alias_visible <= max_ro 才可见（身份揭示之前不能用化名查到这个人）。
- 关系：至少有一条 ev_ros <= max_ro 的证据才显示；返回的证据、证据数、首末次位置都只按可见证据统计。
- 事件、提及：reading_order <= max_ro。
查询结果在 Python 端再复核一遍（check_visible），任何一处漏过滤都会报错，而不是把后文返回给调用方。
不做 Text2Cypher：只有这里列出的固定操作。关系类型、跳数这类不能参数化的部分只从白名单 / 整数里取值。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from novel_rag.config import KGConfig
from novel_rag.kg.schema import REL_BY_NAME

NO_LIMIT_RO = 10**9  # max_reading_order=None（不过滤）时代入的值
STRUCTURAL = ("PARTICIPATES", "OCCURS_AT", "CAUSES")  # 事件相关的边，档案和邻居默认不列

# 可见证据：只保留 reading_order <= $max_ro 的证据（按阅读顺序）
_EVS = (
    "[i IN range(0, size({r}.ev_ros) - 1) WHERE {r}.ev_ros[i] <= $max_ro | "
    "{{chunk_id: {r}.ev_chunks[i], ro: {r}.ev_ros[i], start: {r}.ev_starts[i], end: {r}.ev_ends[i]}}]"
)
# 可见名字：规范名可见时用规范名，否则用最早可见的别名
_DISPLAY = (
    "CASE WHEN {n}.name_visible <= $max_ro THEN {n}.name "
    "ELSE head([i IN range(0, size({n}.aliases) - 1) WHERE {n}.alias_visible[i] <= $max_ro | {n}.aliases[i]]) END"
)
_ALIASES = "[i IN range(0, size({n}.aliases) - 1) WHERE {n}.alias_visible[i] <= $max_ro | {n}.aliases[i]]"


def evs(r: str) -> str:
    return _EVS.format(r=r)


def display(n: str) -> str:
    return _DISPLAY.format(n=n)


def aliases(n: str) -> str:
    return _ALIASES.format(n=n)


class SpoilerViolation(RuntimeError):
    """图谱查询结果中出现了超出阅读进度的实体、证据或别名。"""


class QueryError(ValueError):
    """参数不合法（关系类型不在白名单、跳数越界等）。"""


_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


def lucene_escape(text: str) -> str:
    return _LUCENE_SPECIAL.sub(r"\\\1", text)


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    reading_order: int
    start: int | None = None
    end: int | None = None


@dataclass
class QueryResult:
    op: str
    params: dict[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False


# ---- 模板（固定字符串；只有白名单内的跳数会被格式化进去）----

Q_RESOLVE_EXACT = f"""
MATCH (n:Entity)
WHERE n.series_id = $sid AND n.first_ro <= $max_ro AND NOT n:Event
  AND (($name = n.name AND n.name_visible <= $max_ro)
       OR any(i IN range(0, size(n.aliases) - 1) WHERE n.aliases[i] = $name AND n.alias_visible[i] <= $max_ro))
RETURN n.entity_id AS entity_id, n.type AS type, {display("n")} AS name, n.qualifier AS qualifier,
       n.category AS category, {aliases("n")} AS aliases, n.first_ro AS first_ro, n.n_mentions AS n_mentions,
       $name IN coalesce(n.weak_names, []) AS common_word
ORDER BY n.n_mentions DESC, n.entity_id
LIMIT $limit
"""

Q_RESOLVE_FULLTEXT = f"""
CALL db.index.fulltext.queryNodes('kg_entity_names', $q) YIELD node AS n, score
WHERE n.series_id = $sid AND n.first_ro <= $max_ro AND NOT n:Event
RETURN n.entity_id AS entity_id, n.type AS type, {display("n")} AS name, n.qualifier AS qualifier,
       n.category AS category, {aliases("n")} AS aliases, n.first_ro AS first_ro, n.n_mentions AS n_mentions, score
ORDER BY score DESC, n.n_mentions DESC
LIMIT $limit
"""

Q_TOP_FACTS = f"""
UNWIND $ids AS eid
MATCH (n:Entity {{entity_id: eid}})-[r:REL]-(m:Entity)
WHERE r.first_ro <= $max_ro AND m.first_ro <= $max_ro AND NOT m:Event AND NOT r.type IN $structural
WITH n, r, m, size([x IN r.ev_ros WHERE x <= $max_ro]) AS k
ORDER BY k DESC, m.entity_id
WITH n, collect({{type: r.type, role: r.role, outgoing: startNode(r) = n, other: {display("m")},
                  other_id: m.entity_id, other_type: m.type, other_category: m.category, k: k,
                  ros: [x IN r.ev_ros WHERE x <= $max_ro]}})[..$per] AS facts
RETURN n.entity_id AS entity_id, facts
"""

Q_ENTITY = f"""
MATCH (n:Entity {{entity_id: $eid}})
WHERE n.series_id = $sid AND n.first_ro <= $max_ro
OPTIONAL MATCH (c:Chunk)-[m:MENTIONS]->(n) WHERE c.reading_order <= $max_ro
WITH n, min(c.reading_order) AS first_seen, max(c.reading_order) AS last_seen, sum(m.count) AS mentions
RETURN n.entity_id AS entity_id, n.type AS type, {display("n")} AS name, n.qualifier AS qualifier,
       n.category AS category, {aliases("n")} AS aliases, n.first_ro AS first_ro,
       first_seen, last_seen, mentions
"""

Q_RELATIONS = f"""
MATCH (n:Entity {{entity_id: $eid}})-[r:REL]-(m:Entity)
WHERE n.first_ro <= $max_ro AND r.first_ro <= $max_ro AND m.first_ro <= $max_ro
  AND ($types IS NULL OR r.type IN $types) AND ($exclude IS NULL OR NOT r.type IN $exclude)
  AND ($other_types IS NULL OR m.type IN $other_types)
WITH n, r, m, {evs("r")} AS ev
WHERE size(ev) > 0
RETURN r.type AS type, r.role AS role, startNode(r) = n AS outgoing, m.entity_id AS other_id,
       {display("m")} AS other, m.type AS other_type, m.category AS other_category, size(ev) AS n_evidence,
       ev[0].ro AS first_ro, ev[-1].ro AS last_ro, ev[..$ev_per] AS evidence
ORDER BY n_evidence DESC, first_ro, other_id
LIMIT $limit
"""

Q_BETWEEN = f"""
MATCH (a:Entity {{entity_id: $a}})-[r:REL]-(b:Entity {{entity_id: $b}})
WHERE a.first_ro <= $max_ro AND b.first_ro <= $max_ro AND r.first_ro <= $max_ro
WITH a, b, r, {evs("r")} AS ev
WHERE size(ev) > 0
RETURN r.type AS type, r.role AS role, startNode(r) = a AS a_to_b, size(ev) AS n_evidence,
       ev[0].ro AS first_ro, ev[-1].ro AS last_ro, ev[..$ev_per] AS evidence
ORDER BY first_ro, type
LIMIT $limit
"""

_Q_PATHS = """
MATCH (a:Entity {{entity_id: $a}}), (b:Entity {{entity_id: $b}})
WHERE a.first_ro <= $max_ro AND b.first_ro <= $max_ro
MATCH p = allShortestPaths((a)-[:REL*..{hops}]-(b))
WHERE all(r IN relationships(p) WHERE r.first_ro <= $max_ro AND ($types IS NULL OR r.type IN $types))
  AND all(x IN nodes(p) WHERE x.first_ro <= $max_ro
          AND ($node_types IS NULL OR x.type IN $node_types)
          AND ($exclude_node_types IS NULL OR NOT x.type IN $exclude_node_types))
WITH p LIMIT $limit
RETURN [x IN nodes(p) | {{id: x.entity_id, name: {disp}, type: x.type, category: x.category}}] AS nodes,
       [r IN relationships(p) | {{type: r.type, role: r.role, from: startNode(r).entity_id,
          to: endNode(r).entity_id, evidence: {ev}[..$ev_per], n_evidence: size({ev})}}] AS rels
"""
Q_PATHS = {h: _Q_PATHS.format(hops=h, disp=display("x"), ev=evs("r")) for h in (1, 2, 3)}

Q_KG_VERSION = """
MATCH (m:KGMeta) WHERE m.series_id = $sid
RETURN m.kg_version AS kg_version
LIMIT 1
"""

Q_TIMELINE_EVENTS = """
MATCH (n:Entity {entity_id: $eid})-[r:REL {type: 'PARTICIPATES'}]->(e:Event)
WHERE n.first_ro <= $max_ro AND e.reading_order <= $max_ro
RETURN e.entity_id AS event_id, e.description AS description, e.reading_order AS ro,
       e.chunk_ids AS chunk_ids
ORDER BY e.reading_order, e.entity_id
LIMIT $limit
"""

Q_MENTIONS = """
MATCH (c:Chunk)-[m:MENTIONS]->(n:Entity {entity_id: $eid})
WHERE n.first_ro <= $max_ro AND c.reading_order <= $max_ro
  AND ($books IS NULL OR c.book_id IN $books)
  AND ($ro_from IS NULL OR c.reading_order >= $ro_from)
RETURN c.chunk_id AS chunk_id, c.reading_order AS ro, m.count AS count
ORDER BY CASE WHEN $by_count THEN -m.count ELSE 0 END, c.reading_order, c.chunk_id
LIMIT $limit
"""

Q_TITLE_HOLDERS = f"""
MATCH (p:Entity)-[r:REL {{type: 'HOLDS_TITLE'}}]->(t:Title {{entity_id: $tid}})
WHERE t.first_ro <= $max_ro AND p.first_ro <= $max_ro AND r.first_ro <= $max_ro
WITH p, r, {evs("r")} AS ev
WHERE size(ev) > 0
RETURN p.entity_id AS holder_id, {display("p")} AS holder, size(ev) AS n_evidence,
       ev[0].ro AS first_ro, ev[-1].ro AS last_ro, ev[..$ev_per] AS evidence
ORDER BY first_ro, holder_id
LIMIT $limit
"""

Q_CHUNK_META = """
MATCH (c:Chunk) WHERE c.series_id = $sid
RETURN c.chunk_id AS chunk_id, c.book_id AS book_id, c.book_title AS book_title,
       c.chapter_title AS chapter_title, c.reading_order AS ro
"""

ALL_TEMPLATES = (
    Q_RESOLVE_EXACT, Q_RESOLVE_FULLTEXT, Q_TOP_FACTS, Q_ENTITY, Q_RELATIONS, Q_BETWEEN, *Q_PATHS.values(),
    Q_TIMELINE_EVENTS, Q_MENTIONS, Q_TITLE_HOLDERS,
)  # fmt: skip


def check_visible(rows: Sequence[dict[str, Any]], max_ro: int, op: str) -> None:
    """复核：结果中所有 reading_order 类字段（ro / first_ro / last_ro / 证据的 ro）都不超过阅读进度。"""

    def walk(x: Any, path: str) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("ro", "first_ro", "last_ro", "first_seen", "last_seen") and isinstance(v, int) and v > max_ro:
                    raise SpoilerViolation(f"{op} 返回了超出阅读进度的 {path}.{k}={v}（max_reading_order={max_ro}）")
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    walk(list(rows), op)


class GraphQueries:
    """模板化图谱查询。driver 为 neo4j 驱动（测试可注入假驱动）；所有查询都在只读事务里执行。"""

    def __init__(self, driver: Any, database: str, cfg: KGConfig, series_id: str):
        self.driver, self.database, self.cfg, self.series_id = driver, database, cfg, series_id
        self._chunk_meta: dict[str, dict[str, Any]] | None = None

    # ---- 执行 ----

    def _limit(self, limit: int | None) -> int:
        lim = self.cfg.default_limit if limit is None else int(limit)
        return max(1, min(lim, self.cfg.query_max_limit))

    def _read(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """只读事务（READ 访问模式 + execute_read）+ 超时。"""
        from neo4j import READ_ACCESS, unit_of_work

        @unit_of_work(timeout=self.cfg.query_timeout_s)
        def work(tx):  # noqa: ANN001
            return [r.data() for r in tx.run(query, **params)]

        with self.driver.session(
            database=self.database, default_access_mode=READ_ACCESS, notifications_min_severity="OFF"
        ) as s:
            return s.execute_read(work)

    def _run(self, op: str, query: str, max_ro: int | None, **params: Any) -> list[dict[str, Any]]:
        ro = NO_LIMIT_RO if max_ro is None else int(max_ro)
        rows = self._read(query, sid=self.series_id, max_ro=ro, **params)
        check_visible(rows, ro, op)
        return rows

    @staticmethod
    def _types(types: Sequence[str] | None) -> list[str] | None:
        if types is None:
            return None
        bad = [t for t in types if t not in REL_BY_NAME]
        if bad:
            raise QueryError(f"未知的关系类型：{bad}（可选 {sorted(REL_BY_NAME)}）")
        return list(types)

    def chunk_meta(self) -> dict[str, dict[str, Any]]:
        if self._chunk_meta is None:
            self._chunk_meta = {r["chunk_id"]: r for r in self._read(Q_CHUNK_META, sid=self.series_id)}
        return self._chunk_meta

    def where(self, chunk_id: str) -> str:
        """证据位置："卷名 · 章节标题"。"""
        m = self.chunk_meta().get(chunk_id)
        return f"{m['book_title']} · {m['chapter_title']}" if m else chunk_id

    # ---- 操作 ----

    def resolve(self, name: str, max_reading_order: int | None, limit: int | None = None) -> QueryResult:
        """名字 → 候选实体（精确匹配可见的规范名 / 别名；没有命中再走全文索引）。同名时返回多个候选。"""
        lim = self._limit(limit)
        rows = self._run("resolve", Q_RESOLVE_EXACT, max_reading_order, name=name.strip(), limit=lim)
        how = "exact"
        if not rows:
            q = lucene_escape(name.strip())
            rows = self._run("resolve", Q_RESOLVE_FULLTEXT, max_reading_order, q=q, limit=lim) if q else []
            how = "fulltext"
        if rows:
            facts = self._run(
                "resolve.facts", Q_TOP_FACTS, max_reading_order,
                ids=[r["entity_id"] for r in rows], structural=list(STRUCTURAL), per=3,
            )  # fmt: skip
            by_id = {f["entity_id"]: f["facts"] for f in facts}
            for r in rows:
                r["facts"] = by_id.get(r["entity_id"], [])
        return QueryResult("resolve", {"name": name, "match": how}, rows)

    def profile(self, entity_id: str, max_reading_order: int | None, limit: int | None = None) -> QueryResult:
        """人物 / 实体档案：可见别名、所属群体、按可见证据数排序的主要关系（带首末次位置）。"""
        lim = self._limit(limit)
        head = self._run("profile", Q_ENTITY, max_reading_order, eid=entity_id)
        if not head:
            return QueryResult("profile", {"entity_id": entity_id}, [])
        rels = self._run(
            "profile.relations", Q_RELATIONS, max_reading_order, eid=entity_id, types=None,
            exclude=list(STRUCTURAL), other_types=None, ev_per=2, limit=lim,
        )  # fmt: skip
        row = {**head[0], "groups": [r["other"] for r in rels if r["type"] == "MEMBER_OF" and r["outgoing"]]}
        row["relations"] = rels
        return QueryResult("profile", {"entity_id": entity_id}, [row], truncated=len(rels) >= lim)

    def between(self, a: str, b: str, max_reading_order: int | None, limit: int | None = None) -> QueryResult:
        rows = self._run("between", Q_BETWEEN, max_reading_order, a=a, b=b, ev_per=3, limit=self._limit(limit))
        return QueryResult("between", {"a": a, "b": b}, rows)

    def paths(
        self,
        a: str,
        b: str,
        max_reading_order: int | None,
        *,
        max_hops: int | None = None,
        types: Sequence[str] | None = None,
        node_types: Sequence[str] | None = None,
        exclude_node_types: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> QueryResult:
        """两个实体之间不超过 3 跳的最短路径（可限定关系类型、途经节点的类型），每条边带可见证据。
        node_types：路径上所有节点（含两端）都必须是这些类型；exclude_node_types：路径不经过这些类型的节点。"""
        hops = self.cfg.path_max_hops if max_hops is None else int(max_hops)
        if hops not in Q_PATHS:
            raise QueryError(f"跳数只能是 1–3，收到 {max_hops}")
        rows = self._run(
            "paths", Q_PATHS[hops], max_reading_order, a=a, b=b, types=self._types(types), ev_per=2,
            node_types=list(node_types) if node_types else None,
            exclude_node_types=list(exclude_node_types) if exclude_node_types else None,
            limit=self._limit(limit),
        )  # fmt: skip
        return QueryResult("paths", {"a": a, "b": b, "max_hops": hops, "types": types}, rows)

    def kg_version(self) -> str | None:
        """当前写在 Neo4j 里的图谱版本（KGMeta），评测时记入运行配置。"""
        rows = self._read(Q_KG_VERSION, sid=self.series_id)
        return rows[0]["kg_version"] if rows else None

    def neighbors(
        self,
        entity_id: str,
        max_reading_order: int | None,
        *,
        types: Sequence[str] | None = None,
        other_types: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> QueryResult:
        """邻居：按关系类型 / 对方类型筛选，按可见证据数排序，强制上限（超级节点也只返回前若干个）。"""
        lim = self._limit(limit)
        rows = self._run(
            "neighbors", Q_RELATIONS, max_reading_order, eid=entity_id, types=self._types(types),
            exclude=None if types else list(STRUCTURAL), other_types=list(other_types) if other_types else None,
            ev_per=2, limit=lim,
        )  # fmt: skip
        return QueryResult("neighbors", {"entity_id": entity_id, "types": types}, rows, truncated=len(rows) >= lim)

    def timeline(self, entity_id: str, max_reading_order: int | None, limit: int | None = None) -> QueryResult:
        """时间线：参与的事件 + 关键关系首次出现，按 reading_order 排序（阅读顺序不等于故事时间）。"""
        lim = self._limit(limit)
        events = self._run("timeline.events", Q_TIMELINE_EVENTS, max_reading_order, eid=entity_id, limit=lim)
        rels = self._run(
            "timeline.relations", Q_RELATIONS, max_reading_order, eid=entity_id, types=None,
            exclude=list(STRUCTURAL), other_types=None, ev_per=1, limit=lim,
        )  # fmt: skip
        items = [{"kind": "event", **e} for e in events]
        items += [
            {"kind": "relation", "ro": r["first_ro"], "type": r["type"], "role": r["role"], "outgoing": r["outgoing"],
             "other": r["other"], "other_id": r["other_id"], "other_type": r["other_type"],
             "other_category": r["other_category"], "evidence": r["evidence"]}
            for r in rels
        ]  # fmt: skip
        items.sort(key=lambda x: (x["ro"], 0 if x["kind"] == "event" else 1))
        return QueryResult("timeline", {"entity_id": entity_id}, items[:lim], truncated=len(items) > lim)

    def mentions(
        self,
        entity_id: str,
        max_reading_order: int | None,
        *,
        book_ids: Sequence[str] | None = None,
        ro_from: int | None = None,
        by_count: bool = False,
        limit: int | None = None,
    ) -> QueryResult:
        """提及位置：某实体在指定范围内被提及的 chunk（图谱 → 原文的桥梁）。"""
        rows = self._run(
            "mentions", Q_MENTIONS, max_reading_order, eid=entity_id, books=list(book_ids) if book_ids else None,
            ro_from=ro_from, by_count=by_count, limit=self._limit(limit),
        )  # fmt: skip
        return QueryResult("mentions", {"entity_id": entity_id}, rows)

    def title_holders(self, title_id: str, max_reading_order: int | None, limit: int | None = None) -> QueryResult:
        """头衔的历任持有者（按首次可见证据排序）。"""
        rows = self._run(
            "title_holders", Q_TITLE_HOLDERS, max_reading_order, tid=title_id, ev_per=2, limit=self._limit(limit)
        )
        return QueryResult("title_holders", {"title_id": title_id}, rows)
