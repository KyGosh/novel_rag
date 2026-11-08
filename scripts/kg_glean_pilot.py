"""试点对比："补漏第二遍"（把第一遍结果给模型，让它只补遗漏）能多抽出多少条通过校验的实体和关系。

只在试点上跑（P3 方案第五节：召回提升明显才采用）。没有全量金标时，用"第二遍新增的、通过同样校验的条目数
占第一遍的比例"衡量召回潜力，并打印新增关系供抽查精度。
用法：
    uv run python scripts/kg_glean_pilot.py --tag pilot
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter

from novel_rag.batch import run_jobs
from novel_rag.config import get_settings
from novel_rag.ingestion.chunking import chunk_version, load_books
from novel_rag.kg.common import extract_dir, load_series_kg
from novel_rag.kg.extract import (
    SECTION_NOTES,
    ChunkExtractionOut,
    chunk_context,
    extract_version,
    validate_extraction,
)
from novel_rag.kg.schema import KIN_ROLES, REL_BY_NAME
from novel_rag.llm import get_chat_model
from novel_rag.prompts import load_prompt


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--show", type=int, default=30, help="打印多少条新增关系供抽查")
    args = parser.parse_args()

    s = get_settings()
    k = s.kg
    version = chunk_version(s.chunk)
    chat = get_chat_model("extraction")
    first_prompt = load_prompt("kg_extract")
    glean_prompt = load_prompt("kg_glean")
    records = None
    book_ids: list[str] = []
    for b in ("asoiaf-1",):
        book_ids.append(b)
    books, chunks = load_books(s.paths.books_dir, book_ids, version)
    series = load_series_kg(s.paths.books_dir, book_ids)
    xver = extract_version(series.schema, first_prompt, chat.cache_identity(), version)
    path = extract_dir(s.paths.data_dir, xver) / f"{args.tag}.jsonl"
    records = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    by_id = {c["chunk_id"]: c for c in chunks}

    async def fn(rec):
        chunk = by_id[rec["chunk_id"]]
        book = books[chunk["book_id"]]
        ctx = chunk_context(book, chunk, k.context_chars)
        msgs = glean_prompt.render(
            entity_types=series.schema.prompt_entity_types(),
            relation_types=series.schema.prompt_relation_types(),
            book_title=book.title,
            chapter_title=chunk.get("chapter_title") or "",
            pov=chunk.get("pov") or "无",
            section_note=SECTION_NOTES.get(chunk["section_type"], ""),
            context=ctx or "（无，本段是章节开头）",
            text=chunk["text"],
            first_pass=json.dumps(rec["raw"], ensure_ascii=False),
        )
        res = await chat.achat(msgs, schema=ChunkExtractionOut, prompt_version=glean_prompt.version_id)
        first = ChunkExtractionOut.model_validate(rec["raw"])
        extra = res.parsed
        merged = ChunkExtractionOut(
            entities=first.entities + extra.entities,
            relations=first.relations + extra.relations,
            aliases=first.aliases + extra.aliases,
        )
        new, _ = validate_extraction(
            merged, book, chunk, ctx, series.schema, series.rules, fuzzy_threshold=k.evidence_fuzzy_threshold
        )
        return rec, new, res

    results = asyncio.run(run_jobs(records, fn, concurrency=k.jobs_in_flight, label="glean"))
    ok = [r.value for r in results if r.ok]
    base_rel = base_ent = add_rel = add_ent = add_alias = 0
    tokens = Counter()
    shown = 0
    for rec, new, res in ok:

        def rel_keys(r):
            ms = r["mentions"]
            return {(ms[x["s"]]["key"], x["type"], x["role"], ms[x["o"]]["key"]) for x in r["relations"]}

        old_r, new_r = rel_keys(rec), rel_keys(new)
        old_e = {(m["type"], m["key"]) for m in rec["mentions"]}
        new_e = {(m["type"], m["key"]) for m in new["mentions"]}
        base_rel += len(old_r)
        base_ent += len(old_e)
        add_rel += len(new_r - old_r)
        add_ent += len(new_e - old_e)
        add_alias += max(0, len(new["aliases"]) - len(rec["aliases"]))
        tokens["prompt"] += res.usage.prompt_tokens
        tokens["completion"] += res.usage.completion_tokens
        book = books[rec["book_id"]]
        for x in new["relations"]:
            ms = new["mentions"]
            key = (ms[x["s"]]["key"], x["type"], x["role"], ms[x["o"]]["key"])
            if key not in old_r and shown < args.show:
                shown += 1
                label = REL_BY_NAME[x["type"]].label + (f"/{KIN_ROLES[x['role']]}" if x["role"] else "")
                quote = book.text[x["evidence"][0] : x["evidence"][1]]
                print(f"[新增] {rec['chunk_id']} {ms[x['s']]['name']} -[{label}]-> {ms[x['o']]['name']}  「{quote}」")
    report = {
        "chunks": len(ok),
        "failed": len(results) - len(ok),
        "first_pass": {"entities": base_ent, "relations": base_rel},
        "gleaning_added": {"entities": add_ent, "relations": add_rel, "aliases": add_alias},
        "relative_gain": {
            "entities": round(add_ent / max(base_ent, 1), 3),
            "relations": round(add_rel / max(base_rel, 1), 3),
        },
        "gleaning_tokens": dict(tokens),
    }
    out = extract_dir(s.paths.data_dir, xver) / f"{args.tag}.gleaning_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
