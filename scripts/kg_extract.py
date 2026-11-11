"""知识图谱抽取（P3 A 线）：逐 chunk 抽取实体、关系、别名并校验。结果按内容 hash 缓存，可中断后续跑。

用法：
    uv run python scripts/kg_extract.py --books asoiaf-1 --chapters 1-8 --appendix-sample 2 --tag pilot   # 试点
    uv run python scripts/kg_extract.py --tag all                                                     # 全部卷
    uv run python scripts/kg_extract.py --tag all --revalidate      # 只在已保存的原始输出上重跑校验，不调用 LLM
产物：data/kg/extract/{抽取版本}/{tag}.jsonl 与 {tag}.report.json（含原文派生内容，data/ 不入库）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import Counter

from novel_rag.batch import run_jobs
from novel_rag.config import get_settings
from novel_rag.ingestion.chunking import chunk_version, load_books, series_book_ids
from novel_rag.kg.common import extract_dir, load_series_kg
from novel_rag.kg.extract import VALIDATION_VERSION, extract_chunk, extract_version, revalidate
from novel_rag.llm import get_chat_model, get_usage_tracker
from novel_rag.prompts import load_prompt


def _parse_range(s: str) -> tuple[int, int]:
    a, _, b = s.partition("-")
    return int(a), int(b or a)


def report_for(records: list[dict], failed: list[dict], seconds: float, extra: dict) -> dict:
    drops: Counter = Counter()
    for r in records:
        drops.update(r["drops"])
    llm = [r["llm"] for r in records]
    fresh = [x for x in llm if not x["cached"]]
    n = max(len(records), 1)
    return {
        **extra,
        "chunks": len(records) + len(failed),
        "succeeded": len(records),
        "failed": failed,
        "seconds": round(seconds, 1),
        "mentions": sum(len(r["mentions"]) for r in records),
        "relations": sum(len(r["relations"]) for r in records),
        "aliases": sum(len(r["aliases"]) for r in records),
        "per_chunk": {
            "mentions": round(sum(len(r["mentions"]) for r in records) / n, 2),
            "relations": round(sum(len(r["relations"]) for r in records) / n, 2),
        },
        "raw_counts": {
            "entities": sum(len(r["raw"]["entities"]) for r in records),
            "relations": sum(len(r["raw"]["relations"]) for r in records),
            "aliases": sum(len(r["raw"]["aliases"]) for r in records),
        },
        "drops": dict(drops.most_common()),
        "relation_types": dict(Counter(x["type"] for r in records for x in r["relations"]).most_common()),
        "evidence_match": dict(Counter(x["match"] for r in records for x in r["relations"]).most_common()),
        "llm": {
            "calls_fresh": len(fresh),
            "cache_hits": len(llm) - len(fresh),
            "retries": sum(x["attempts"] - 1 for x in llm),
            "prompt_tokens": sum(x["prompt_tokens"] for x in fresh),
            "completion_tokens": sum(x["completion_tokens"] for x in fresh),
            "mean_prompt_tokens": round(sum(x["prompt_tokens"] for x in fresh) / max(len(fresh), 1)),
            "mean_completion_tokens": round(sum(x["completion_tokens"] for x in fresh) / max(len(fresh), 1)),
            "mean_latency_s": round(sum(x["latency_s"] for x in fresh) / max(len(fresh), 1), 2),
        },
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--books", help="逗号分隔的 book_id，默认为该分块版本下的全部卷")
    parser.add_argument("--chapters", help="章节号范围（如 1-8），只在单卷时使用")
    parser.add_argument("--appendix-sample", type=int, default=0, help="另外随机抽几个附录 chunk（试点用）")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", required=True, help="产物文件名，如 pilot / all")
    parser.add_argument("--chunk-version")
    parser.add_argument("--revalidate", action="store_true", help="只在已保存的原始输出上重跑校验")
    args = parser.parse_args()

    s = get_settings()
    k = s.kg
    version = args.chunk_version or chunk_version(s.chunk)
    book_ids = [x.strip() for x in args.books.split(",")] if args.books else series_book_ids(s.paths.books_dir, version)
    books, chunks = load_books(s.paths.books_dir, book_ids, version)
    series = load_series_kg(s.paths.books_dir, book_ids)
    chat = get_chat_model("extraction")
    prompt = load_prompt("kg_extract")
    xver = extract_version(series.schema, prompt, chat.cache_identity(), version)
    out_dir = extract_dir(s.paths.data_dir, xver)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}.jsonl"
    meta = {
        "extract_version": xver,
        "tag": args.tag,
        "chunk_version": version,
        "schema_version": series.schema.extract_version,
        "prompt_version": prompt.version_id,
        "validation_version": VALIDATION_VERSION,
        "model_identity": chat.cache_identity(),
        "books": book_ids,
    }

    if args.revalidate:
        records = [json.loads(x) for x in out_path.read_text(encoding="utf-8").splitlines() if x]
        by_id = {c["chunk_id"]: c for c in chunks}
        records = revalidate(
            records, books, by_id, series.schema, series.rules,
            context_chars=k.context_chars, fuzzy_threshold=k.evidence_fuzzy_threshold,
        )  # fmt: skip
        failed: list[dict] = []
        elapsed = 0.0
    else:
        selected = chunks
        if args.chapters:
            lo, hi = _parse_range(args.chapters)
            selected = [c for c in chunks if c["section_type"] == "narrative" and lo <= c["chapter_idx"] <= hi]
        if args.appendix_sample:
            pool = [c for c in chunks if c["section_type"] == "appendix"]
            selected = selected + random.Random(20261107).sample(pool, min(args.appendix_sample, len(pool)))
        if args.limit:
            selected = selected[: args.limit]

        async def fn(c):
            return await extract_chunk(
                books[c["book_id"]], c, chat, prompt, series.schema, series.rules,
                context_chars=k.context_chars, fuzzy_threshold=k.evidence_fuzzy_threshold,
            )  # fmt: skip

        t0 = time.perf_counter()
        results = asyncio.run(run_jobs(selected, fn, concurrency=k.jobs_in_flight, label="kg_extract"))
        elapsed = time.perf_counter() - t0
        records = [r.value for r in results if r.ok]
        failed = [{"chunk_id": r.job["chunk_id"], "error": repr(r.error)[:300]} for r in results if not r.ok]

    records.sort(key=lambda r: (r["reading_order"], r["chunk_id"]))
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    report = report_for(records, failed, elapsed, meta)
    report["llm_usage_tracker"] = get_usage_tracker().summary()
    (out_dir / f"{args.tag}.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k2: v for k2, v in report.items() if k2 != "failed"}, ensure_ascii=False, indent=2))
    if failed:
        print(f"{len(failed)} 个 chunk 失败（重跑可续上）：{failed[:3]}", file=sys.stderr)
    print(f"产物：{out_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
