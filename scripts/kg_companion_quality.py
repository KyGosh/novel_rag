"""“伙伴/坐骑”关系全部抽查（P3.1 验收 3）：导出审阅表 → 金标（Claude 预标、用户核对）→ 打分。

用法：
    uv run python scripts/kg_companion_quality.py export --kg <版本>   # 写 companion_sheet.md
    uv run python scripts/kg_companion_quality.py score --kg <版本>    # 读 companion_gold.jsonl
审阅表与金标在 data/reports/kg_quality/{kg}/（含原文片段），结果写 eval/results/{时间}_kg_companion/（只有 ID 与指标）。

每条关系列出主体、客体、改写来源（原关系类型）和全部证据（最多 MAX_EVIDENCE 条，证据区间前后各取 WINDOW 字）。
金标 companion_gold.jsonl：{"key", "verdict": "supported|partial|unsupported", "note"}
  - supported：证据写明一方是另一方的动物伙伴、坐骑或宠物；
  - partial：能看出两者有关联，但看不出伙伴/坐骑关系（或方向反了）；
  - unsupported：证据不支持，或有一方不是动物。
打分：支持率（supported 的比例）与含部分支持的比例，并按证据条数分组。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime

from novel_rag.config import get_settings
from novel_rag.eval.runner import git_info
from novel_rag.ingestion.chunking import chunk_version, load_books, series_book_ids
from novel_rag.kg.build import load_export
from novel_rag.kg.common import kg_root

WINDOW, MAX_EVIDENCE = 60, 3
REL_TYPE = "COMPANION"


def rel_key(r: dict) -> str:
    return f"{r['subject']}|{r['type']}|{r['role'] or ''}|{r['object']}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["export", "score"])
    parser.add_argument("--kg", required=True, help="图谱版本")
    args = parser.parse_args()

    s = get_settings()
    kgdir = kg_root(s.paths.data_dir) / args.kg
    tables, _ = load_export(kgdir)
    ents = {e["entity_id"]: e for e in tables["entities"]}
    rels = sorted((r for r in tables["relations"] if r["type"] == REL_TYPE), key=rel_key)
    work = s.paths.reports_dir / "kg_quality" / args.kg
    work.mkdir(parents=True, exist_ok=True)

    if args.action == "export":
        cver = chunk_version(s.chunk)
        books, chunks = load_books(s.paths.books_dir, series_book_ids(s.paths.books_dir, cver), cver)
        where = {c["chunk_id"]: f"{c['chapter_title']}（ro {c['reading_order']}）" for c in chunks}
        lines = [f"# 伙伴/坐骑关系审阅表（{len(rels)} 条，全部）", ""]
        for i, r in enumerate(rels, 1):
            s_e, o_e = ents[r["subject"]], ents[r["object"]]
            src = "、".join(f"{k}×{v}" for k, v in (r.get("rewritten_from") or {}).items()) or "抽取即为伙伴"
            lines += [
                f"## C{i:02d}　{s_e['name']}（{s_e.get('category')}）"
                f"—伙伴/坐骑→ {o_e['name']}（{o_e.get('category')}）",
                f"`{rel_key(r)}`　证据 {r['n_evidence']} 条　改写自：{src}",
                "",
            ]
            for ev in r["evidence"][:MAX_EVIDENCE]:
                text = books[ev["book_id"]].text
                a, b = ev["start"], ev["end"]
                snip = text[max(0, a - WINDOW) : a] + "【" + text[a:b] + "】" + text[b : b + WINDOW]
                snip = snip.replace("\n", " ")
                lines += [f"- [{where.get(ev['chunk_id'], ev['chunk_id'])}] {snip}", ""]
        (work / "companion_sheet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"审阅表：{work / 'companion_sheet.md'}（{len(rels)} 条）")
        return 0

    gold = {g["key"]: g for g in map(json.loads, (work / "companion_gold.jsonl").open(encoding="utf-8"))}
    missing = [rel_key(r) for r in rels if rel_key(r) not in gold]
    if missing:
        print(f"金标缺 {len(missing)} 条：{missing[:5]}")
        return 1

    def rates(rs: list[dict]) -> dict:
        c = Counter(gold[rel_key(r)]["verdict"] for r in rs)
        n = max(len(rs), 1)
        return {"n": len(rs), "verdicts": dict(c), "support_rate": round(c["supported"] / n, 3),
                "support_rate_incl_partial": round((c["supported"] + c["partial"]) / n, 3)}  # fmt: skip

    out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "kg_version": args.kg,
        **rates(rels),
        "by_evidence_count": {
            "n_evidence=1": rates([r for r in rels if r["n_evidence"] == 1]),
            "n_evidence>=2": rates([r for r in rels if r["n_evidence"] >= 2]),
        },
        "not_supported": [
            {"key": rel_key(r), "verdict": gold[rel_key(r)]["verdict"], "note": gold[rel_key(r)].get("note", "")}
            for r in rels
            if gold[rel_key(r)]["verdict"] != "supported"
        ],
    }
    review = work / "companion_review_status.txt"
    out["review"] = review.read_text(encoding="utf-8").strip() if review.is_file() else "Claude 预标（待用户核对）"
    res = s.paths.results_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_kg_companion"
    res.mkdir(parents=True)
    (res / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "git"}, ensure_ascii=False, indent=2))
    print(f"结果：{res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
