"""由抽取记录 + 消歧结果 + 章节摘要组装图谱导出文件（data/kg/{图谱版本}/），它是图谱的权威数据，Neo4j 只是查询副本。

- 关系聚合：键为（主体、类型、角色、客体），汇总全部证据（chunk_id、reading_order、引文区间），记首末次出现，
  从不覆盖：先盟友后敌人两条关系共存，靠 reading_order 区分先后。对称关系按实体 ID 排序存一个方向。
- 提及：chunk → 实体（次数）。
- 事件来自章节摘要：参与者与地点按"所引 chunk → 本章"的顺序映射到实体，映射不了的留空并计数；
  证据就是摘要所引的 chunk。因果（CAUSES）只来自同一章内、原文引文能定位的判断（见 events.py）。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from novel_rag.kg.names import NameRules, normalize_name
from novel_rag.kg.schema import LOC, PERSON, is_symmetric

EXPORT_FILES = ("entities", "relations", "events", "mentions", "chunks")


def kg_version(parts: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]


def aggregate_relations(records: Sequence[dict[str, Any]], m2e: dict[str, str], stats: Counter) -> list[dict[str, Any]]:
    agg: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        for r in rec["relations"]:
            s = m2e.get(f"{rec['chunk_id']}#{r['s']}")
            o = m2e.get(f"{rec['chunk_id']}#{r['o']}")
            if s is None or o is None:
                stats["relation_unresolved_entity"] += 1
                continue
            if s == o:
                stats["relation_self_after_merge"] += 1
                continue
            if is_symmetric(r["type"], r["role"]) and o < s:
                s, o = o, s
            agg[(s, r["type"], r["role"], o)].append(
                {
                    "chunk_id": rec["chunk_id"],
                    "reading_order": rec["reading_order"],
                    "start": r["evidence"][0],
                    "end": r["evidence"][1],
                    "book_id": rec["book_id"],
                }
            )
    for s, t, role, o in list(agg):
        if t == "KIN" and role == "parent":
            target = next((k for k in ((s, t, "father", o), (s, t, "mother", o)) if k in agg), None)
            if target is not None:
                agg[target].extend(agg.pop((s, t, role, o)))
                stats["kin_parent_merged"] += 1
    out = []
    for (s, t, role, o), evs in sorted(agg.items()):
        evs.sort(key=lambda e: (e["reading_order"], e["chunk_id"], e["start"]))
        dedup, seen = [], set()
        for e in evs:
            k = (e["chunk_id"], e["start"])
            if k not in seen:
                seen.add(k)
                dedup.append(e)
        out.append(
            {
                "subject": s,
                "type": t,
                "role": role,
                "object": o,
                "first_reading_order": dedup[0]["reading_order"],
                "last_reading_order": dedup[-1]["reading_order"],
                "n_evidence": len(dedup),
                "evidence": dedup,
            }
        )
    stats["relations"] = len(out)
    return out


def chunk_mentions(records: Sequence[dict[str, Any]], m2e: dict[str, str]) -> list[dict[str, Any]]:
    counts: Counter = Counter()
    for rec in records:
        for m in rec["mentions"]:
            eid = m2e.get(f"{rec['chunk_id']}#{m['m']}")
            if eid:
                counts[(rec["chunk_id"], eid)] += m.get("count", 1)
    return [{"chunk_id": c, "entity_id": e, "count": n} for (c, e), n in sorted(counts.items())]


class ChapterNameIndex:
    """章节内的名字 → 实体：用于把摘要事件的参与者 / 地点（原文写法）映射到实体 ID。"""

    def __init__(self, records: Sequence[dict[str, Any]], m2e: dict[str, str], rules: NameRules):
        self.rules = rules
        self.by_chunk: dict[str, dict[tuple[str, str], Counter]] = defaultdict(lambda: defaultdict(Counter))
        self.by_chapter: dict[tuple[str, int], dict[tuple[str, str], Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        for rec in records:
            for m in rec["mentions"]:
                eid = m2e.get(f"{rec['chunk_id']}#{m['m']}")
                if not eid:
                    continue
                for v in {m["name"], m["full_name"], m["key"]} - {None}:
                    for form in (normalize_name(v), rules.canonical(v)):
                        self.by_chunk[rec["chunk_id"]][(m["type"], form)][eid] += 1
                        self.by_chapter[(rec["book_id"], rec["chapter_idx"])][(m["type"], form)][eid] += 1

    def lookup(self, name: str, type_: str, chunk_ids: Iterable[str], chapter: tuple[str, int]) -> str | None:
        forms = {normalize_name(name), self.rules.canonical(name)}
        pooled: Counter = Counter()
        for cid in chunk_ids:
            for f in forms:
                pooled.update(self.by_chunk.get(cid, {}).get((type_, f), Counter()))
        if not pooled:
            for f in forms:
                pooled.update(self.by_chapter.get(chapter, {}).get((type_, f), Counter()))
        if not pooled:
            return None
        top = pooled.most_common(2)
        if len(top) == 2 and top[0][1] == top[1][1]:
            return None  # 同样常见的两个候选：不猜
        return top[0][0]


def build_events(
    summaries: Sequence[dict[str, Any]], index: ChapterNameIndex, stats: Counter
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """事件节点与 参与 / 发生于 关系（附录章节不产生事件）。"""
    events, rels = [], []
    for s in summaries:
        if s["section_type"] != "narrative":
            continue
        chapter = (s["book_id"], s["chapter_idx"])
        for e in s["events"]:
            parts, unmapped = [], []
            for p in e["participants"]:
                eid = index.lookup(p, PERSON, e["chunk_ids"], chapter)
                (parts if eid else unmapped).append(eid or p)
            loc = index.lookup(e["location"], LOC, e["chunk_ids"], chapter) if e.get("location") else None
            stats["event_participants"] += len(e["participants"])
            stats["event_participants_unmapped"] += len(unmapped)
            if e.get("location"):
                stats["event_locations"] += 1
                stats["event_locations_unmapped"] += loc is None
            evidence = [
                {
                    "chunk_id": c,
                    "reading_order": s["reading_order"],
                    "start": None,
                    "end": None,
                    "book_id": s["book_id"],
                }
                for c in e["chunk_ids"]
            ]
            events.append(
                {
                    "event_id": e["event_id"],
                    "description": e["description"],
                    "book_id": s["book_id"],
                    "chapter_idx": s["chapter_idx"],
                    "chapter_title": s["chapter_title"],
                    "reading_order": s["reading_order"],
                    "chunk_ids": e["chunk_ids"],
                    "participants": sorted(set(parts)),
                    "unmapped_participants": unmapped,
                    "location": loc,
                    "location_text": e.get("location"),
                }
            )
            for pid in sorted(set(parts)):
                rels.append(_event_rel(pid, "PARTICIPATES", e["event_id"], evidence))
            if loc:
                rels.append(_event_rel(e["event_id"], "OCCURS_AT", loc, evidence))
    stats["events"] = len(events)
    return events, rels


def _event_rel(s: str, t: str, o: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    ro = evidence[0]["reading_order"]
    return {
        "subject": s,
        "type": t,
        "role": None,
        "object": o,
        "first_reading_order": ro,
        "last_reading_order": ro,
        "n_evidence": len(evidence),
        "evidence": evidence,
    }


def chunk_rows(chunks: Sequence[dict[str, Any]], book_titles: dict[str, str]) -> list[dict[str, Any]]:
    """chunk 节点只存编号与位置，不存正文（正文按 chunk_id 从语料读取）。"""
    keys = ("chunk_id", "book_id", "chapter_idx", "seq", "reading_order", "section_type", "chapter_title")
    return [{**{k: c.get(k) for k in keys}, "book_title": book_titles.get(c["book_id"], "")} for c in chunks]


def write_export(out_dir: Path, tables: dict[str, list[dict[str, Any]]], manifest: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in EXPORT_FILES:
        rows = tables.get(name, [])
        (out_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline=""
        )
    manifest = {**manifest, "counts": {n: len(tables.get(n, [])) for n in EXPORT_FILES}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_export(directory: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not (directory / "manifest.json").is_file():
        raise FileNotFoundError(f"{directory} 不是图谱导出目录（缺 manifest.json）")
    tables = {}
    for name in EXPORT_FILES:
        p = directory / f"{name}.jsonl"
        tables[name] = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x] if p.is_file() else []
    return tables, json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
