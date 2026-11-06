"""ES 与 Milvus 中 chunk 集合的一致性检查：以 chunks.jsonl 为权威，逐条比对 ID 与字段。

ES、Milvus（以及 P3 的 Neo4j）共用 chunk_id，任何一处少了、多了或字段不一致，融合与引用就会出错。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# 两个库都存的字段；pov / chapter_title 在库中以 "" 表示空
COMPARED_FIELDS = (
    "book_id",
    "chapter_idx",
    "seq",
    "reading_order",
    "section_type",
    "chunk_type",
    "pov",
    "chapter_title",
    "text",
)
ES_ONLY_FIELDS = ("char_start", "char_end")


def _norm(field: str, value: Any) -> Any:
    return (value or "") if field in ("pov", "chapter_title") else value


def compare_store(
    name: str, expected: dict[str, dict[str, Any]], rows: Iterable[dict[str, Any]], fields: tuple[str, ...]
) -> dict[str, Any]:
    seen: set[str] = set()
    mismatched: dict[str, list[str]] = {}
    duplicated: list[str] = []
    extra: list[str] = []
    for r in rows:
        cid = r["chunk_id"]
        if cid in seen:
            duplicated.append(cid)
            continue
        seen.add(cid)
        exp = expected.get(cid)
        if exp is None:
            extra.append(cid)
            continue
        bad = [f for f in fields if _norm(f, r.get(f)) != _norm(f, exp.get(f))]
        if bad:
            mismatched[cid] = bad
    missing = sorted(set(expected) - seen)
    return {
        "store": name,
        "count": len(seen),
        "expected": len(expected),
        "missing": missing[:20],
        "n_missing": len(missing),
        "extra": sorted(extra)[:20],
        "n_extra": len(extra),
        "duplicated": duplicated[:20],
        "mismatched": dict(list(mismatched.items())[:20]),
        "n_mismatched": len(mismatched),
        "ok": not (missing or extra or duplicated or mismatched),
    }


def check_consistency(
    chunks: Iterable[dict[str, Any]], es_docs: Iterable[dict[str, Any]], milvus_rows: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    expected = {c["chunk_id"]: c for c in chunks}
    es = compare_store("es", expected, es_docs, COMPARED_FIELDS + ES_ONLY_FIELDS)
    mv = compare_store("milvus", expected, milvus_rows, COMPARED_FIELDS)
    return {"es": es, "milvus": mv, "ok": es["ok"] and mv["ok"]}
