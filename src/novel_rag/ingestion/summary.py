"""章节摘要：每章一次 LLM 调用，输出摘要、关键事件（带支撑 chunk）和主要人物。

- 输入：整章正文按 chunk 分段（相邻 chunk 的重叠部分只出现一次），每段标【编号】，编号映射回 chunk_id；
  附卷名、章节标题、POV、section_type。
- 只依据本章原文，所以天然不越过阅读进度；prompt 明确禁止模型补充自带的作品知识，忠实度抽查专门检查这一点。
- 校验：事件引用的段落编号必须属于本章，越界编号丢弃，一个有效引用都没有的事件整条作废（计数）。
- 每份摘要记录模型、prompt 版本、chunk 版本；摘要版本 = 这些因素的 hash，产物写到 data/summaries/{摘要版本}/。

摘要不进 chunk 索引：它在 text.txt 中没有字符位置，混进去会干扰按字符区间判定的评测。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from novel_rag.corpus import BookText, Chapter
from novel_rag.llm.base import ChatModel, Message
from novel_rag.prompts import Prompt

SUMMARY_SCHEMA_VERSION = "sum-1"
SECTION_NOTES = {"appendix": "\n（这是附录，不是叙事章节：摘要概括其内容与结构即可，关键事件可以为空列表。）"}


class SummaryEventOut(BaseModel):
    description: str = Field(description="一句话描述事件，不超过 60 字")
    participants: list[str] = Field(default_factory=list, description="参与的人物，按原文写法")
    location: str | None = Field(default=None, description="地点的原文写法；原文没有明确地点时为 null")
    segments: list[int] = Field(default_factory=list, description="支撑该事件的段落编号（1–3 个）")


class ChapterSummaryOut(BaseModel):
    summary: str = Field(description="300–500 字的本章摘要")
    events: list[SummaryEventOut] = Field(default_factory=list, description="3–8 条关键事件")
    characters: list[str] = Field(default_factory=list, description="本章主要人物，按原文写法，最多 15 个")


@dataclass(frozen=True)
class Segment:
    label: int  # 从 1 开始
    chunk_id: str
    text: str  # 去掉与前一个 chunk 重叠部分后的文本


def chapter_segments(book: BookText, chunks: Sequence[dict[str, Any]]) -> list[Segment]:
    """chunks 为同一章、按 seq 排好的 chunk；相邻 chunk 的重叠部分只保留在前一段中。"""
    segs: list[Segment] = []
    prev_end = -1
    for c in chunks:
        start = max(c["char_start"], prev_end)
        segs.append(Segment(len(segs) + 1, c["chunk_id"], book.text[start : c["char_end"]].strip()))
        prev_end = c["char_end"]
    return segs


def summary_version(prompt: Prompt, model_identity: dict[str, Any], chunk_version: str) -> str:
    payload = {
        "schema": SUMMARY_SCHEMA_VERSION,
        "prompt": prompt.version_id,
        "model": model_identity,
        "chunk_version": chunk_version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]


def build_messages(book: BookText, chapter: Chapter, segments: Sequence[Segment], prompt: Prompt) -> list[Message]:
    body = "\n\n".join(f"【{s.label}】{s.text}" for s in segments)
    return prompt.render(
        book_title=book.title or book.book_id,
        chapter_title=chapter.title,
        pov=chapter.pov or "（无）",
        section_type=chapter.section_type,
        section_note=SECTION_NOTES.get(chapter.section_type, ""),
        segments=body,
    )


def validate_summary(
    out: ChapterSummaryOut, chapter: Chapter, segments: Sequence[Segment]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """把段落编号映射为 chunk_id；越界编号丢弃，没有有效引用的事件作废。返回（事件列表, 计数）。"""
    by_label = {s.label: s.chunk_id for s in segments}
    events: list[dict[str, Any]] = []
    stats = {"invalid_refs": 0, "dropped_events": 0}
    for ev in out.events:
        refs = []
        for x in ev.segments:
            if x in by_label and by_label[x] not in refs:
                refs.append(by_label[x])
            elif x not in by_label:
                stats["invalid_refs"] += 1
        description = ev.description.strip()
        if not refs or not description:
            stats["dropped_events"] += 1
            continue
        events.append(
            {
                "event_id": f"{chapter.book_id}:{chapter.chapter_idx:04d}:e{len(events) + 1:02d}",
                "description": description,
                "participants": [p.strip() for p in ev.participants if p.strip()],
                "location": (ev.location or "").strip() or None,
                "chunk_ids": refs,
            }
        )
    return events, stats


async def summarize_chapter(
    book: BookText,
    chapter: Chapter,
    chunks: Sequence[dict[str, Any]],
    chat: ChatModel,
    prompt: Prompt,
    *,
    chunk_version: str,
) -> dict[str, Any]:
    segments = chapter_segments(book, chunks)
    if not segments:
        raise ValueError(f"{book.book_id} 第 {chapter.chapter_idx} 章没有 chunk")
    result = await chat.achat(
        build_messages(book, chapter, segments, prompt), schema=ChapterSummaryOut, prompt_version=prompt.version_id
    )
    out = result.parsed
    assert isinstance(out, ChapterSummaryOut)
    events, stats = validate_summary(out, chapter, segments)
    return {
        "chapter_key": f"{chapter.book_id}:{chapter.chapter_idx:04d}",
        "book_id": chapter.book_id,
        "book_title": book.title,
        "chapter_idx": chapter.chapter_idx,
        "chapter_title": chapter.title,
        "pov": chapter.pov,
        "section_type": chapter.section_type,
        "reading_order": chapter.reading_order,
        "summary": out.summary.strip(),
        "events": events,
        "characters": list(dict.fromkeys(c.strip() for c in out.characters if c.strip()))[:15],
        "n_segments": len(segments),
        "summary_chars": len(out.summary.strip()),
        **stats,
        "model": result.model,
        "prompt_version": prompt.version_id,
        "chunk_version": chunk_version,
        "cached": result.cached,
        "attempts": result.attempts,
        "usage": {"prompt_tokens": result.usage.prompt_tokens, "completion_tokens": result.usage.completion_tokens},
    }


def summaries_dir(data_dir: Path, version: str) -> Path:
    return data_dir / "summaries" / version


def load_summaries(directory: Path) -> list[dict[str, Any]]:
    path = directory / "summaries.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} 不存在，先运行 scripts/summarize.py")
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
