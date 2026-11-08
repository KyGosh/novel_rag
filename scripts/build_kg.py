"""构建知识图谱导出文件：抽取记录 → 实体消歧 → 关系聚合 → 事件（来自章节摘要）→ data/kg/{图谱版本}/。

用法：
    uv run python scripts/build_kg.py --extract-tag all                       # 全部卷
    uv run python scripts/build_kg.py --extract-tag all --books asoiaf-1       # 只用某几卷的抽取记录
    uv run python scripts/build_kg.py --extract-tag all --no-overrides        # 不用人工覆盖文件（消融）
写入 Neo4j 用 scripts/load_kg.py（Neo4j 是可以随时重建的查询副本）。
LLM 调用（拆分检查、候选对判定、因果判断）都按内容 hash 缓存，重跑只花新增的部分。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from datetime import datetime

from novel_rag.config import get_settings
from novel_rag.ingestion.chunking import chunk_version, load_books, series_book_ids
from novel_rag.ingestion.summary import load_summaries
from novel_rag.kg.build import (
    ChapterNameIndex,
    aggregate_relations,
    build_events,
    chunk_mentions,
    chunk_rows,
    kg_version,
    write_export,
)
from novel_rag.kg.common import extract_dir, kg_root, load_series_kg
from novel_rag.kg.events import extract_causes
from novel_rag.kg.extract import VALIDATION_VERSION, extract_version
from novel_rag.kg.names import normalize_name
from novel_rag.kg.resolve import RESOLVE_VERSION, Mention, Overrides, Resolver
from novel_rag.llm import get_chat_model, get_embedder, get_usage_tracker
from novel_rag.prompts import load_prompt
from novel_rag.retrieval.embed_cache import EmbeddingCache, open_embedding_store


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extract-tag", default="all")
    parser.add_argument("--books", help="只用这些卷（逗号分隔）的抽取记录；默认全部")
    parser.add_argument("--summary-version", help="默认取最近生成的")
    parser.add_argument("--no-overrides", action="store_true", help="不使用 aliases.toml（消融）")
    parser.add_argument("--no-causes", action="store_true", help="跳过事件因果判断")
    args = parser.parse_args()

    s = get_settings()
    k = s.kg
    cver = chunk_version(s.chunk)
    all_books = series_book_ids(s.paths.books_dir, cver)
    books, chunks = load_books(s.paths.books_dir, all_books, cver)
    series = load_series_kg(s.paths.books_dir, all_books)
    extraction_chat = get_chat_model("extraction")
    xver = extract_version(series.schema, load_prompt("kg_extract"), extraction_chat.cache_identity(), cver)
    path = extract_dir(s.paths.data_dir, xver) / f"{args.extract_tag}.jsonl"
    records = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    use_books = set(args.books.split(",")) if args.books else set(all_books)
    records = [r for r in records if r["book_id"] in use_books]
    bad_val = {r.get("validation_version") for r in records} - {VALIDATION_VERSION}
    if bad_val:
        print(f"抽取记录的校验版本 {bad_val} 与当前 {VALIDATION_VERSION} 不一致，先运行 kg_extract.py --revalidate")
        return 1
    chunk_text = {c["chunk_id"]: c["text"] for c in chunks}
    sroot = s.paths.data_dir / "summaries"
    sver = args.summary_version or max(sroot.glob("*/summaries.jsonl"), key=lambda p: p.stat().st_mtime).parent.name
    summaries = [r for r in load_summaries(sroot / sver) if r["book_id"] in use_books]

    def snippet(m: Mention) -> str:
        text = chunk_text[m.chunk_id]
        i = normalize_name(text).find(m.name)
        if i < 0:
            return ""
        return text[max(0, i - 30) : i + len(m.name) + 30].replace("\n", " ")

    disamb = get_chat_model("disambiguation")
    split_prompt, judge_prompt = load_prompt("kg_split"), load_prompt("kg_judge")
    causes_prompt = load_prompt("kg_causes")
    embedder = EmbeddingCache(get_embedder(), open_embedding_store(s.paths.embedding_cache_dir))
    overrides_path = s.paths.books_dir / series.series_id / "aliases.toml"
    use_overrides = not args.no_overrides and overrides_path.is_file()

    resolver = Resolver(records, series.rules, k, series.series_id, snippet)
    t0 = time.perf_counter()
    stats: Counter = Counter()

    async def run() -> list[dict]:
        resolver.build_initial()
        await resolver.split_check(disamb, split_prompt, embedder)  # type: ignore[arg-type]
        resolver.link_short_names()
        resolver.drop_surname_clusters()
        resolver.apply_alias_statements()
        if use_overrides:
            resolver.apply_overrides(Overrides.load(overrides_path))
        pairs = resolver.candidate_pairs(embedder)  # type: ignore[arg-type]
        await resolver.judge_pairs(pairs, disamb, judge_prompt)
        if args.no_causes:
            return []
        return await extract_causes(summaries, books, chunks, extraction_chat, causes_prompt, k, stats)

    cause_rels = asyncio.run(run())
    entities, m2e = resolver.finalize()
    relations = aggregate_relations(records, m2e, stats)
    index = ChapterNameIndex(records, m2e, series.rules)
    events, event_rels = build_events(summaries, index, stats)
    mentions = chunk_mentions(records, m2e)
    book_titles = {b: books[b].title for b in all_books}
    used_chunks = [c for c in chunks if c["book_id"] in use_books]

    parts = {
        "schema": series.schema.version,
        "extract_version": xver,
        "validation_version": VALIDATION_VERSION,
        "resolve_version": RESOLVE_VERSION,
        "name_rules": series.rules.fingerprint(),
        "resolve_params": k.model_dump(
            include={
                "split_min_mentions",
                "split_max_descriptions",
                "candidate_top_k",
                "candidate_min_sim",
                "auto_merge_sim",
                "judge_max_pairs",
                "judge_evidence_quotes",
                "alias_merge_min_support",
                "short_link_min_ratio",
            }
        ),  # fmt: skip
        "prompts": [split_prompt.version_id, judge_prompt.version_id, causes_prompt.version_id],
        "disambiguation_model": disamb.cache_identity(),
        "summary_version": sver,
        "chunk_version": cver,
        "books": sorted(use_books),
        "overrides": overrides_path.read_text(encoding="utf-8") if use_overrides else None,
        "causes": not args.no_causes,
    }
    ver = kg_version(parts)
    out = kg_root(s.paths.data_dir) / ver
    manifest = {
        "kg_version": ver,
        "series_id": series.series_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "version_parts": {kk: v for kk, v in parts.items() if kk != "overrides"},
        "overrides_used": use_overrides,
        "book_text_sha256": {b: books[b].text_sha256 for b in sorted(use_books)},
        "book_titles": book_titles,
    }
    write_export(
        out,
        {
            "entities": entities,
            "relations": relations + event_rels + cause_rels,
            "events": events,
            "mentions": mentions,
            "chunks": chunk_rows(used_chunks, book_titles),
        },
        manifest,
    )
    # 内部产物（消歧描述等，可能含后续剧情，只供质量评估）：不进导出、不进 Neo4j
    internal = out / "internal"
    internal.mkdir(exist_ok=True)
    (internal / "mention_entity.json").write_text(json.dumps(m2e, ensure_ascii=False), encoding="utf-8")
    (internal / "decisions.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in resolver.decisions), encoding="utf-8"
    )
    by_type = Counter(e["type"] for e in entities)
    rel_types = Counter(r["type"] for r in relations + event_rels + cause_rels)
    report = {
        "kg_version": ver,
        "seconds": round(time.perf_counter() - t0, 1),
        "mentions_in": len(resolver.mentions),
        "entities_by_type": dict(by_type),
        "relations_by_type": dict(rel_types.most_common()),
        "resolver": dict(sorted(resolver.stats.items())),
        "build": dict(sorted(stats.items())),
        "aliases_ungrounded": sum(1 for e in entities for a in e["aliases"] if a["link"] == "ungrounded"),
        "llm_usage": get_usage_tracker().summary(),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"图谱导出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
