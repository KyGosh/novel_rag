"""章节摘要忠实度抽查：随机抽 N 章，导出审阅表（摘要 + 每条事件 + 所引段落原文），审阅结论写回后汇总支持率。

用法：
    uv run python scripts/summary_spotcheck.py export --n 20   # 写 data/reports/summary_spotcheck/{版本}/sheet.md
    uv run python scripts/summary_spotcheck.py score           # 读 verdicts.jsonl，写 eval/results/ 下的结果
verdicts.jsonl 每行一条事件或一章摘要的结论（不含原文）：
    {"chapter_key": "...", "event_id": "...", "verdict": "supported|partial|unsupported", "note": "..."}
    {"chapter_key": "...", "event_id": null, "out_of_chapter": 0, "note": "摘要中越出本章的内容条数"}
审阅流程：Claude 预审写 verdicts.jsonl，用户确认（或修改）后再 score。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime

from novel_rag.config import get_settings
from novel_rag.corpus import BookText
from novel_rag.eval.runner import git_info
from novel_rag.ingestion.chunking import load_books, series_book_ids
from novel_rag.ingestion.summary import chapter_segments, load_summaries

SEED = 20261107


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["export", "score"])
    parser.add_argument("--version", help="摘要版本，默认取最近生成的")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    s = get_settings()
    root = s.paths.data_dir / "summaries"
    version = args.version or max(root.glob("*/summaries.jsonl"), key=lambda p: p.stat().st_mtime).parent.name
    records = load_summaries(root / version)
    work = s.paths.reports_dir / "summary_spotcheck" / version
    work.mkdir(parents=True, exist_ok=True)
    narrative = [r for r in records if r["section_type"] == "narrative"]
    sample = sorted(
        random.Random(SEED).sample(narrative, min(args.n, len(narrative))), key=lambda r: r["reading_order"]
    )

    if args.action == "export":
        cver = records[0]["chunk_version"]
        books, chunks = load_books(s.paths.books_dir, series_book_ids(s.paths.books_dir, cver), cver)
        lines = [f"# 摘要忠实度抽查（摘要版本 {version}，{len(sample)} 章，种子 {SEED}）", ""]
        for r in sample:
            book: BookText = books[r["book_id"]]
            cs = [c for c in chunks if c["book_id"] == r["book_id"] and c["chapter_idx"] == r["chapter_idx"]]
            segs = {seg.chunk_id: seg for seg in chapter_segments(book, cs)}
            lines += [
                f"## {r['chapter_key']} 《{r['book_title']}》{r['chapter_title']}",
                "",
                f"摘要：{r['summary']}",
                "",
            ]
            for e in r["events"]:
                lines.append(
                    f"### {e['event_id']}：{e['description']}（参与者 {e['participants']}，地点 {e['location']}）"
                )
                for cid in e["chunk_ids"]:
                    lines.append(f"> 【{segs[cid].label}】{segs[cid].text}")
                lines.append("")
        (work / "sheet.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"审阅表：{work / 'sheet.md'}（{sum(len(r['events']) for r in sample)} 条事件）")
        return 0

    verdicts = [json.loads(x) for x in (work / "verdicts.jsonl").read_text(encoding="utf-8").splitlines() if x]
    ev = [v for v in verdicts if v.get("event_id")]
    ch = [v for v in verdicts if not v.get("event_id")]
    want = {e["event_id"] for r in sample for e in r["events"]}
    missing = want - {v["event_id"] for v in ev}
    if missing:
        print(f"还有 {len(missing)} 条事件没有结论：{sorted(missing)[:5]}", file=sys.stderr)
        return 1
    counts = Counter(v["verdict"] for v in ev)
    n = len(ev)
    result = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "summary_version": version,
        "chunk_version": records[0]["chunk_version"],
        "prompt_version": records[0]["prompt_version"],
        "model": records[0]["model"],
        "sample": {"seed": SEED, "chapters": [r["chapter_key"] for r in sample], "events": n},
        "event_verdicts": dict(counts),
        "event_support_rate": round(counts["supported"] / n, 4),
        "event_support_rate_incl_partial": round((counts["supported"] + counts["partial"]) / n, 4),
        "out_of_chapter_items": sum(int(v.get("out_of_chapter", 0)) for v in ch),
        "chapters_with_out_of_chapter": sum(1 for v in ch if int(v.get("out_of_chapter", 0)) > 0),
        "review": "Claude 预审 + 用户确认"
        if all(v.get("confirmed") for v in verdicts)
        else "Claude 预审（待用户确认）",
        "verdicts": verdicts,
    }
    out = s.paths.results_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_kg_quality_summary"
    out.mkdir(parents=True)
    (out / "summary_faithfulness.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "verdicts"}, ensure_ascii=False, indent=2))
    print(f"结果：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
