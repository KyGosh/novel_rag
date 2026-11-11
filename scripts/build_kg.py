"""构建知识图谱导出文件：抽取记录 → 实体消歧 → 人物类别判定 → 关系聚合与清洗 → 事件（来自章节摘要）→
data/kg/{图谱版本}/。

用法：
    uv run python scripts/build_kg.py --extract-tag all                       # 全部卷
    uv run python scripts/build_kg.py --extract-tag all --books asoiaf-1       # 只用某几卷的抽取记录
    uv run python scripts/build_kg.py --extract-tag all --no-overrides        # 不用人工覆盖文件（消融）
    uv run python scripts/build_kg.py --extract-tag all --no-category         # 不做人物类别判定（全部按"人"）
常用词表（判定"弱名字"用）是 IK 自带主词典的副本，从 ES 容器导出一次即可（不存在时不做常用词判定）：
    docker cp novel-rag-es:/usr/share/elasticsearch/config/analysis-ik/main.dic data/lexicon/ik_main.dic
写入 Neo4j 用 scripts/load_kg.py（Neo4j 是可以随时重建的查询副本）。
LLM 调用（拆分检查、候选对判定、类别判定、因果判断）都按内容 hash 缓存，重跑只花新增的部分。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
from novel_rag.kg.category import (
    CATEGORY_VERSION,
    CorpusCounter,
    apply_category_overrides,
    assign_categories,
    clean_relations,
    drop_entities,
    entity_cards,
    judge_categories,
    linked_form_counts,
    load_word_list,
    person_category_of,
    weak_names,
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
    parser.add_argument("--no-category", action="store_true", help="不做人物类别判定（全部按「人」，不清洗关系）")
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
    category_prompt = None if args.no_category else load_prompt("kg_category")
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
    entities, m2e_full = resolver.finalize()

    # 人物类别判定（人 / 动物 / 非角色）→ 去掉非角色 → 关系按类别规则清洗
    cat_stats: Counter = Counter()
    judgments = {}
    cards = entity_cards(
        entities, m2e_full, {mid: m.description for mid, m in resolver.mentions.items()},
        max_descriptions=k.category_max_descriptions,
    )  # fmt: skip
    if category_prompt is not None:
        judgments = asyncio.run(
            judge_categories(
                cards,
                disamb,
                category_prompt,
                batch_size=k.category_batch_size,
                concurrency=k.jobs_in_flight,
                stats=cat_stats,
            )  # fmt: skip
        )
        if use_overrides:
            apply_category_overrides(judgments, entities, Overrides.load(overrides_path).categories, cat_stats)
    dropped = assign_categories(entities, judgments)
    relations = aggregate_relations(records, m2e_full, stats)
    entities, m2e = drop_entities(entities, m2e_full, dropped, cat_stats)
    index = ChapterNameIndex(records, m2e, series.rules)
    events, event_rels = build_events(summaries, index, stats)
    mentions = chunk_mentions(records, m2e)
    all_relations = clean_relations(
        relations + event_rels + cause_rels, person_category_of(entities), dropped, cat_stats
    )

    # 常用词名字（弱匹配）
    used_ids = {c["chunk_id"] for c in chunks if c["book_id"] in use_books}
    common_words = load_word_list(s.paths.common_words_file)
    weak_details: list[dict] = []
    if common_words is not None:
        weak, weak_details = weak_names(
            entities, linked_form_counts(records), CorpusCounter(chunk_text[c] for c in sorted(used_ids)),
            common_words, min_linked_ratio=k.weak_name_min_linked_ratio, stats=cat_stats,
        )  # fmt: skip
        for e in entities:
            e["weak_names"] = weak.get(e["entity_id"], [])
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
        "category": None
        if category_prompt is None
        else {
            "version": CATEGORY_VERSION,
            "prompt": category_prompt.version_id,
            "params": k.model_dump(include={"category_batch_size", "category_max_descriptions"}),
        },
        "common_words": None
        if common_words is None
        else {
            "sha256": hashlib.sha256(s.paths.common_words_file.read_bytes()).hexdigest()[:16],
            "min_linked_ratio": k.weak_name_min_linked_ratio,
        },
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
            "relations": all_relations,
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
    by_cat = {e["entity_id"]: (e.get("category"), e["type"]) for e in entities}
    (internal / "categories.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "entity_id": c.entity_id, "name": c.name, "aliases": list(c.aliases), "groups": list(c.groups),
                    "n_mentions": c.n_mentions, "descriptions": list(c.descriptions),
                    "category": judgments[c.entity_id].category if c.entity_id in judgments else "人",
                    "source": judgments[c.entity_id].source if c.entity_id in judgments else "default",
                    "reason": judgments[c.entity_id].reason if c.entity_id in judgments else "",
                },
                ensure_ascii=False,
            ) + "\n"
            for c in cards
        ),
        encoding="utf-8",
    )  # fmt: skip
    (internal / "weak_names.json").write_text(json.dumps(weak_details, ensure_ascii=False, indent=1), encoding="utf-8")
    by_type = Counter(e["type"] for e in entities)
    rel_types = Counter(r["type"] for r in all_relations)
    # 类别清洗后的自检：人与动物之间不应再有亲属、配偶、婚约、头衔、成员关系
    cross = Counter(
        r["type"]
        for r in all_relations
        if r["type"] in ("KIN", "SPOUSE", "BETROTHED", "HOLDS_TITLE", "MEMBER_OF", "SERVES", "MENTOR")
        and "动物" in {by_cat.get(r["subject"], (None,))[0], by_cat.get(r["object"], (None,))[0]}
        and not (r["type"] == "KIN" and by_cat[r["subject"]][0] == by_cat[r["object"]][0])
    )
    report = {
        "kg_version": ver,
        "seconds": round(time.perf_counter() - t0, 1),
        "mentions_in": len(resolver.mentions),
        "entities_by_type": dict(by_type),
        "relations_by_type": dict(rel_types.most_common()),
        "resolver": dict(sorted(resolver.stats.items())),
        "build": dict(sorted(stats.items())),
        "persons_by_category": dict(Counter(c for c, t in by_cat.values() if t == "Person")),
        "category": dict(sorted(cat_stats.items())),
        "animal_human_forbidden_relations": dict(cross),
        "aliases_ungrounded": sum(1 for e in entities for a in e["aliases"] if a["link"] == "ungrounded"),
        "llm_usage": get_usage_tracker().summary(),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"图谱导出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
