"""同章事件之间的因果（CAUSES）：每章一次 LLM 调用，只收原文明确写出、引文能在本章定位到的因果。

跨章因果不做；BEFORE 也不从阅读顺序推断（卷四、卷五的时间线是并行的，阅读顺序不等于故事时间）。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from novel_rag.batch import run_jobs
from novel_rag.config import KGConfig
from novel_rag.corpus import BookText
from novel_rag.eval.qgen.locate import locate
from novel_rag.ingestion.summary import chapter_segments
from novel_rag.llm.base import ChatModel
from novel_rag.prompts import Prompt


class CauseOut(BaseModel):
    cause: int
    effect: int
    evidence: str = ""


class CausesOut(BaseModel):
    causes: list[CauseOut] = Field(default_factory=list)


async def extract_causes(
    summaries: Sequence[dict[str, Any]],
    books: dict[str, BookText],
    chunks: Sequence[dict[str, Any]],
    chat: ChatModel,
    prompt: Prompt,
    cfg: KGConfig,
    stats: Counter,
) -> list[dict[str, Any]]:
    by_chapter: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        by_chapter[(c["book_id"], c["chapter_idx"])].append(c)
    jobs = [s for s in summaries if s["section_type"] == "narrative" and len(s["events"]) >= 2]

    async def one(s: dict[str, Any]) -> list[dict[str, Any]]:
        book = books[s["book_id"]]
        cs = sorted(by_chapter[(s["book_id"], s["chapter_idx"])], key=lambda c: c["seq"])
        cited = {cid for e in s["events"] for cid in e["chunk_ids"]}
        segs = [seg for seg in chapter_segments(book, cs) if seg.chunk_id in cited]
        events = "\n".join(f"{i}. {e['description']}" for i, e in enumerate(s["events"], 1))
        msgs = prompt.render(
            chapter_title=s["chapter_title"], events=events, segments="\n\n".join(seg.text for seg in segs)
        )
        res = await chat.achat(msgs, schema=CausesOut, prompt_version=prompt.version_id)
        ch = book.chapter(s["chapter_idx"])
        out, seen = [], set()
        for c in res.parsed.causes:
            n = len(s["events"])
            if not (1 <= c.cause <= n and 1 <= c.effect <= n) or c.cause == c.effect:
                stats["cause_bad_index"] += 1
                continue
            loc = locate(
                book.text, c.evidence, ch.body_start, ch.char_end, fuzzy_threshold=cfg.evidence_fuzzy_threshold
            )
            if loc is None:
                stats["cause_evidence_not_found"] += 1
                continue
            key = (c.cause, c.effect)
            if key in seen:
                continue
            seen.add(key)
            chunk = next((x for x in cs if x["char_start"] <= loc.start < x["char_end"]), cs[0])
            a, b = s["events"][c.cause - 1]["event_id"], s["events"][c.effect - 1]["event_id"]
            ev = {
                "chunk_id": chunk["chunk_id"],
                "reading_order": s["reading_order"],
                "start": loc.start,
                "end": loc.end,
                "book_id": s["book_id"],
            }
            out.append(
                {
                    "subject": a,
                    "type": "CAUSES",
                    "role": None,
                    "object": b,
                    "first_reading_order": s["reading_order"],
                    "last_reading_order": s["reading_order"],
                    "n_evidence": 1,
                    "evidence": [ev],
                }
            )
        return out

    results = await run_jobs(jobs, one, concurrency=cfg.jobs_in_flight, label="kg_causes")
    rels = []
    for r in results:
        if r.ok:
            rels.extend(r.value)
        else:
            stats["cause_chapter_failed"] += 1
    stats["causes"] = len(rels)
    return rels
