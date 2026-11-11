"""把图谱导出文件写进 Neo4j（查询副本，可随时从 data/kg/{版本}/ 一键重建）。

结构：
- 节点：(:Entity:Person|Organization|Location|Title)、(:Entity:Event)、(:Chunk)、(:KGMeta)，都带 series_id；
  chunk 节点只存编号、卷、章、reading_order，不存正文。
- 边：实体之间统一用 (:Entity)-[:REL {type, role, ...}]->(:Entity)。Cypher 的关系类型不能作参数，统一以后
  查询模板可以完全参数化，防剧透过滤写法也一致；代价是索引利用略差，图规模小（实体几千、关系几万）可以接受。
  证据存成平行列表 ev_chunks / ev_ros / ev_starts / ev_ends，查询时按 ev_ros <= max_reading_order 过滤。
  (:Chunk)-[:MENTIONS {count, ro}]->(:Entity)，(:Chunk)-[:NEXT]->(:Chunk)（同章相邻）。
- 约束与索引：实体 / chunk ID 唯一；实体名与别名的全文索引（cjk 分析器）；reading_order 索引。
写入只删除同一 series_id 的节点，再用 UNWIND 批量写入（不逐条开事务）。
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from novel_rag.config import Neo4jConfig
from novel_rag.kg.schema import ENTITY_LABELS

SCHEMA_STATEMENTS = (
    "CREATE CONSTRAINT kg_entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.entity_id IS UNIQUE",
    "CREATE CONSTRAINT kg_chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE",
    "CREATE INDEX kg_entity_first_ro IF NOT EXISTS FOR (n:Entity) ON (n.first_ro)",
    "CREATE INDEX kg_entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
    "CREATE INDEX kg_entity_series IF NOT EXISTS FOR (n:Entity) ON (n.series_id)",
    "CREATE INDEX kg_chunk_ro IF NOT EXISTS FOR (n:Chunk) ON (n.reading_order)",
    "CREATE INDEX kg_rel_type IF NOT EXISTS FOR ()-[r:REL]-() ON (r.type)",
    "CREATE FULLTEXT INDEX kg_entity_names IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.aliases] "
    "OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk'}}",
)
ENTITY_TYPES = tuple(ENTITY_LABELS)  # 标签只能取这个白名单，拼进语句是安全的


def _batches(rows: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for i in range(0, len(rows), size):
        yield list(rows[i : i + size])


def entity_rows(entities: Sequence[dict[str, Any]], series_id: str) -> list[dict[str, Any]]:
    rows = []
    for e in entities:
        aliases = e.get("aliases", [])
        rows.append(
            {
                "entity_id": e["entity_id"],
                "series_id": series_id,
                "type": e["type"],
                "name": e["name"],
                "name_visible": e.get("name_visible_from", e["first_reading_order"]),
                "qualifier": e.get("qualifier"),
                "category": e.get("category"),
                "groups": e.get("groups", []),
                "first_ro": e["first_reading_order"],
                "last_ro": e["last_reading_order"],
                "n_mentions": e["n_mentions"],
                "n_chunks": e["n_chunks"],
                "first_chunk": e.get("first_chunk"),
                "aliases": [a["alias"] for a in aliases],
                "alias_visible": [int(a["visible_from"]) for a in aliases],
                "alias_link": [a["link"] for a in aliases],
                "weak_names": list(e.get("weak_names", [])),  # 本身是常用词的写法：查询期链接只给弱匹配
            }
        )
    return rows


def event_rows(events: Sequence[dict[str, Any]], series_id: str) -> list[dict[str, Any]]:
    return [
        {
            "entity_id": e["event_id"],
            "series_id": series_id,
            "type": "Event",
            "name": e["description"],
            "name_visible": e["reading_order"],
            "description": e["description"],
            "book_id": e["book_id"],
            "chapter_idx": e["chapter_idx"],
            "chapter_title": e["chapter_title"],
            "reading_order": e["reading_order"],
            "first_ro": e["reading_order"],
            "last_ro": e["reading_order"],
            "chunk_ids": e["chunk_ids"],
            "location_text": e.get("location_text") or "",
            "aliases": [],
            "alias_visible": [],
            "n_mentions": 0,
        }
        for e in events
    ]


def relation_rows(relations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in relations:
        ev = r["evidence"]
        rows.append(
            {
                "s": r["subject"],
                "o": r["object"],
                "type": r["type"],
                "role": r["role"] or "",
                "first_ro": r["first_reading_order"],
                "last_ro": r["last_reading_order"],
                "n_evidence": r["n_evidence"],
                "ev_chunks": [e["chunk_id"] for e in ev],
                "ev_ros": [int(e["reading_order"]) for e in ev],
                "ev_starts": [-1 if e["start"] is None else int(e["start"]) for e in ev],  # 列表里不能有 null
                "ev_ends": [-1 if e["end"] is None else int(e["end"]) for e in ev],
            }
        )
    return rows


def next_rows(chunks: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    by_ch: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        by_ch[(c["book_id"], c["chapter_idx"])].append(c)
    rows = []
    for cs in by_ch.values():
        cs.sort(key=lambda c: c["seq"])
        rows.extend({"a": x["chunk_id"], "b": y["chunk_id"]} for x, y in zip(cs, cs[1:], strict=False))
    return rows


class Neo4jKGStore:
    def __init__(self, driver: Any, database: str):
        self.driver = driver
        self.database = database

    @classmethod
    def connect(cls, cfg: Neo4jConfig) -> Neo4jKGStore:
        from neo4j import GraphDatabase

        return cls(GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password.get_secret_value())), cfg.database)

    def close(self) -> None:
        self.driver.close()

    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as s:
            return [r.data() for r in s.run(query, **params)]

    def ensure_schema(self) -> None:
        for stmt in SCHEMA_STATEMENTS:
            self._run(stmt)
        self._run("CALL db.awaitIndexes(300)")

    def delete_series(self, series_id: str) -> None:
        # 大批量删除用 CALL {} IN TRANSACTIONS（必须在隐式事务中执行，session.run 即是）
        self._run(
            "MATCH (n) WHERE n.series_id = $sid CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS",
            sid=series_id,
        )

    def replace(
        self, series_id: str, tables: dict[str, list[dict[str, Any]]], manifest: dict[str, Any], batch: int
    ) -> dict[str, int]:
        self.ensure_schema()
        self.delete_series(series_id)
        chunks = [{**c, "series_id": series_id} for c in tables["chunks"]]
        for rows in _batches(chunks, batch):
            self._run(
                "UNWIND $rows AS row CREATE (c:Chunk) SET c = row, c.reading_order = toInteger(row.reading_order)",
                rows=rows,
            )
        ents = entity_rows(tables["entities"], series_id)
        for t in ENTITY_TYPES:
            typed = [r for r in ents if r["type"] == t]
            for rows in _batches(typed, batch):
                self._run(f"UNWIND $rows AS row CREATE (n:Entity:{t}) SET n = row", rows=rows)
        for rows in _batches(event_rows(tables["events"], series_id), batch):
            self._run("UNWIND $rows AS row CREATE (n:Entity:Event) SET n = row", rows=rows)
        for rows in _batches(relation_rows(tables["relations"]), batch):
            self._run(
                "UNWIND $rows AS row "
                "MATCH (a:Entity {entity_id: row.s}) MATCH (b:Entity {entity_id: row.o}) "
                "CREATE (a)-[r:REL]->(b) SET r = row {.type, .role, .first_ro, .last_ro, .n_evidence, "
                ".ev_chunks, .ev_ros, .ev_starts, .ev_ends}",
                rows=rows,
            )
        for rows in _batches(tables["mentions"], batch):
            self._run(
                "UNWIND $rows AS row "
                "MATCH (c:Chunk {chunk_id: row.chunk_id}) MATCH (e:Entity {entity_id: row.entity_id}) "
                "CREATE (c)-[:MENTIONS {count: row.count, ro: c.reading_order}]->(e)",
                rows=rows,
            )
        for rows in _batches(next_rows(tables["chunks"]), batch):
            self._run(
                "UNWIND $rows AS row MATCH (a:Chunk {chunk_id: row.a}) MATCH (b:Chunk {chunk_id: row.b}) "
                "CREATE (a)-[:NEXT]->(b)",
                rows=rows,
            )
        counts = self.counts(series_id)
        self._run(
            "CREATE (m:KGMeta {series_id: $sid, kg_version: $ver, chunk_version: $cver, created: $created, "
            "counts: $counts, manifest: $manifest})",
            sid=series_id,
            ver=manifest.get("kg_version", ""),
            cver=manifest.get("version_parts", {}).get("chunk_version", ""),
            created=manifest.get("created", ""),
            counts=json.dumps(counts, ensure_ascii=False),
            manifest=json.dumps(manifest, ensure_ascii=False),
        )
        return counts

    def counts(self, series_id: str) -> dict[str, int]:
        q = {
            "entities": "MATCH (n:Entity) WHERE n.series_id = $sid AND NOT n:Event RETURN count(n) AS n",
            "events": "MATCH (n:Event) WHERE n.series_id = $sid RETURN count(n) AS n",
            "chunks": "MATCH (n:Chunk) WHERE n.series_id = $sid RETURN count(n) AS n",
            "relations": "MATCH (a:Entity)-[r:REL]->() WHERE a.series_id = $sid RETURN count(r) AS n",
            "mentions": "MATCH (c:Chunk)-[r:MENTIONS]->() WHERE c.series_id = $sid RETURN count(r) AS n",
            "next": "MATCH (c:Chunk)-[r:NEXT]->() WHERE c.series_id = $sid RETURN count(r) AS n",
        }
        return {k: int(self._run(v, sid=series_id)[0]["n"]) for k, v in q.items()}

    def meta(self, series_id: str) -> dict[str, Any] | None:
        rows = self._run("MATCH (m:KGMeta) WHERE m.series_id = $sid RETURN m", sid=series_id)
        return rows[0]["m"] if rows else None
