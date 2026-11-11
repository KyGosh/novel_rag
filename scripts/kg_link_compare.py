"""查询期实体链接对比（P3.1 验收 5）：同一批问题分别用新旧两版图谱的名字表做链接，逐题比较。

用法：
    uv run python scripts/kg_link_compare.py --old d9c84c78 --new 93785ea8 [--dataset asoiaf_v1] [--splits dev test]
名字表直接取自图谱导出文件（与写进 Neo4j 的字段相同），不需要 Neo4j。每题按自己的 max_reading_order 链接。
结果写 eval/results/{时间}_kg_link_compare/（只有题目 ID、命中的写法与实体 ID，不含原文）。

差异归类（同一题可以有多类）：
  - weak：新版本里这段文字的候选全是弱匹配（常用词写法），固定流程不再采用；
  - removed：旧候选实体在新图谱里不存在了（判为非角色删掉，或被并进别的实体）；
  - added：新版本多出的候选实体或命中（多半来自称呼归属新增的别名）；
  - span：命中的文字区间不同（新别名更长，或旧名字没了）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime

from novel_rag.config import get_settings
from novel_rag.eval.runner import git_info
from novel_rag.graph_retrieval.linking import EntityLinker
from novel_rag.ingestion.chunking import chunk_version, series_book_ids
from novel_rag.kg.build import load_export
from novel_rag.kg.common import kg_root, load_series_kg
from novel_rag.kg.neo4j_store import entity_rows


def linker_for(s, version: str, rules) -> tuple[EntityLinker, set[str]]:
    tables, manifest = load_export(kg_root(s.paths.data_dir) / version)
    rows = entity_rows(tables["entities"], manifest.get("series_id", ""))
    return EntityLinker(rows, rules), {e["entity_id"] for e in tables["entities"]}


def snapshot(linker: EntityLinker, q: str, ro: int | None) -> dict[tuple[int, int, str], dict[str, bool]]:
    """(起, 止, 文字) → {实体 ID: 是否弱匹配}"""
    return {(sp.start, sp.end, sp.text): {c.entity_id: c.weak for c in sp.candidates} for sp in linker.link(q, ro)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--dataset", default="asoiaf_v1")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    args = parser.parse_args()

    s = get_settings()
    cver = chunk_version(s.chunk)
    rules = load_series_kg(s.paths.books_dir, series_book_ids(s.paths.books_dir, cver)).rules
    old, old_ids = linker_for(s, args.old, rules)
    new, new_ids = linker_for(s, args.new, rules)
    items = [json.loads(line) for line in (s.paths.datasets_dir / args.dataset / "items.jsonl").open(encoding="utf-8")]
    items = [it for it in items if it["split"] in args.splits]

    kinds: Counter = Counter()
    diffs = []
    weak_hits = []
    for it in items:
        q, ro = it["question"], it.get("max_reading_order")
        a, b = snapshot(old, q, ro), snapshot(new, q, ro)
        for (_i, _j, text), cands in b.items():
            if cands and all(cands.values()):
                weak_hits.append({"id": it["id"], "text": text, "entities": sorted(cands)})
        if a == b:
            continue
        found: set[str] = set()
        detail = []
        for key in sorted(set(a) | set(b)):
            ca, cb = a.get(key, {}), b.get(key, {})
            if ca == cb:
                continue
            if key not in a or key not in b:
                found.add("span")
            if cb and all(cb.values()) and not (ca and all(ca.values())):
                found.add("weak")
            gone = [e for e in ca if e not in cb]
            if any(e not in new_ids for e in gone):
                found.add("removed")
            if any(e not in ca for e in cb):
                found.add("added")
            detail.append({"text": key[2], "start": key[0], "old": ca, "new": cb})
        if not found:
            found.add("other")
        kinds.update(found)
        diffs.append({"id": it["id"], "split": it["split"], "kinds": sorted(found), "spans": detail})

    out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "dataset": args.dataset,
        "splits": args.splits,
        "old_kg": args.old,
        "new_kg": args.new,
        "n_questions": len(items),
        "identical": len(items) - len(diffs),
        "changed": len(diffs),
        "changed_by_kind": dict(kinds),
        "changed_only_weak": sum(1 for d in diffs if d["kinds"] == ["weak"]),
        "weak_hits_new": weak_hits,
        "diffs": diffs,
    }
    res = s.paths.results_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_kg_link_compare"
    res.mkdir(parents=True)
    (res / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("git", "diffs", "weak_hits_new")}, ensure_ascii=False))
    print(f"结果：{res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
