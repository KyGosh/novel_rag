"""章节摘要生成（P3 C 线）：每章一次 LLM 调用，结果按内容 hash 缓存，可中断后续跑。

用法：
    uv run python scripts/summarize.py                          # 该分块版本下的全部卷
    uv run python scripts/summarize.py --books asoiaf-1 --limit 3   # 试跑
产物：data/summaries/{摘要版本}/summaries.jsonl、manifest.json、report.json（含原文派生内容，data/ 不入库）。
只有全部章节都成功时才写 summaries.jsonl；有失败时重跑即可（成功的部分命中缓存）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

from novel_rag.batch import run_jobs
from novel_rag.config import get_settings
from novel_rag.ingestion.chunking import chunk_version, load_books, series_book_ids
from novel_rag.ingestion.summary import (
    SUMMARY_SCHEMA_VERSION,
    summaries_dir,
    summarize_chapter,
    summary_version,
)
from novel_rag.llm import get_chat_model, get_usage_tracker
from novel_rag.prompts import load_prompt


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--books", help="逗号分隔的 book_id，默认为该分块版本下的全部卷")
    parser.add_argument("--chunk-version")
    parser.add_argument("--limit", type=int, help="只跑前 N 章（试跑；不写正式产物，只打印）")
    args = parser.parse_args()

    s = get_settings()
    version = args.chunk_version or chunk_version(s.chunk)
    book_ids = [x.strip() for x in args.books.split(",")] if args.books else series_book_ids(s.paths.books_dir, version)
    books, chunks = load_books(s.paths.books_dir, book_ids, version)
    by_chapter: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for c in chunks:
        by_chapter[(c["book_id"], c["chapter_idx"])].append(c)
    jobs = [(books[b], ch) for b in book_ids for ch in books[b].chapters if (b, ch.chapter_idx) in by_chapter]
    if args.limit:
        jobs = jobs[: args.limit]

    chat = get_chat_model("summary")
    prompt = load_prompt("chapter_summary")
    sver = summary_version(prompt, chat.cache_identity(), version)

    async def fn(job):
        book, ch = job
        return await summarize_chapter(
            book, ch, by_chapter[(book.book_id, ch.chapter_idx)], chat, prompt, chunk_version=version
        )

    t0 = time.perf_counter()
    results = asyncio.run(run_jobs(jobs, fn, concurrency=s.summary.jobs_in_flight, label="summary"))
    elapsed = time.perf_counter() - t0
    ok = [r.value for r in results if r.ok]
    failed = [
        {"chapter": f"{r.job[0].book_id}:{r.job[1].chapter_idx:04d}", "error": repr(r.error)[:300]}
        for r in results
        if not r.ok
    ]
    lengths = [r["summary_chars"] for r in ok if r["section_type"] == "narrative"]
    n_events = [len(r["events"]) for r in ok if r["section_type"] == "narrative"]
    report = {
        "summary_version": sver,
        "chunk_version": version,
        "prompt_version": prompt.version_id,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "model_identity": chat.cache_identity(),
        "books": book_ids,
        "chapters": len(jobs),
        "succeeded": len(ok),
        "failed": failed,
        "seconds": round(elapsed, 1),
        "llm_usage": get_usage_tracker().summary(),
        "events_total": sum(len(r["events"]) for r in ok),
        "dropped_events": sum(r["dropped_events"] for r in ok),
        "invalid_refs": sum(r["invalid_refs"] for r in ok),
        "narrative_summary_chars": {
            "min": min(lengths, default=0),
            "max": max(lengths, default=0),
            "mean": round(sum(lengths) / max(len(lengths), 1), 1),
            "below_target": sum(x < s.summary.target_min_chars for x in lengths),
            "above_target": sum(x > s.summary.target_max_chars for x in lengths),
        },
        "narrative_events_per_chapter": dict(sorted(Counter(n_events).items())),
        "attempts_gt1": sum(r["attempts"] > 1 for r in ok),
    }
    if args.limit:
        for r in ok:
            print(json.dumps({k: r[k] for k in ("chapter_key", "summary", "events", "characters")}, ensure_ascii=False))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if not failed else 1

    out = summaries_dir(s.paths.data_dir, sver)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failed:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"有 {len(failed)} 章失败，未写 summaries.jsonl；重跑即可续上（成功的命中缓存）", file=sys.stderr)
        return 1
    ok.sort(key=lambda r: r["reading_order"])
    (out / "summaries.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ok), encoding="utf-8", newline=""
    )
    manifest = {
        "summary_version": sver,
        "created": datetime.now().isoformat(timespec="seconds"),
        "chunk_version": version,
        "prompt_version": prompt.version_id,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "model_identity": chat.cache_identity(),
        "books": {b: books[b].text_sha256 for b in book_ids},
        "n_chapters": len(ok),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"摘要目录：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
