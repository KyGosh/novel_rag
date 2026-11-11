"""评测集格式：只存问题、简短答案和"卷 + 章 + 字符区间"形式的证据，不存原文。

目录结构：eval/datasets/{name}/items.jsonl + manifest.json。manifest 记录各卷 text.txt 的 sha256，
加载时与本地定稿文本比对，文本一变就拒绝使用（金标区间需要重新校验）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from novel_rag.corpus import BookText

QuestionType = Literal["fact", "relation", "multihop", "plot"]
QUESTION_TYPES: tuple[str, ...] = ("fact", "relation", "multihop", "plot")
Split = Literal["dev", "test"]


class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    book_id: str
    chapter_idx: int
    char_start: int
    char_end: int


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: QuestionType
    question: str
    answer: str
    evidence: list[Evidence] = Field(min_length=1)
    source: Literal["generated", "human"]
    split: Split
    max_reading_order: int | None = None  # 回答所需的最小阅读进度：最后一段证据所在章节的 reading_order
    review: Literal["accepted", "edited"] = "accepted"
    difficulty: int | None = None  # 校验时的人工难度标注 1–3（可选）
    origin: str | None = None  # 候选题 id，便于回溯到出题与校验记录
    answer_aliases: list[str] = Field(default_factory=list)  # 可接受的其他写法（别名、简称），供规则匹配判分（v1 起）


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    books: dict[str, str]  # book_id → 定稿 text.txt 的 sha256
    counts: dict[str, Any] = Field(default_factory=dict)


class DatasetError(RuntimeError):
    pass


def save_dataset(directory: Path, items: list[EvalItem], manifest: DatasetManifest) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = "".join(item.model_dump_json(exclude_none=False) + "\n" for item in items)
    (directory / "items.jsonl").write_text(lines, encoding="utf-8", newline="")
    (directory / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline=""
    )


def load_dataset(directory: Path, books_dir: Path, *, split: Split | None = None) -> list[EvalItem]:
    manifest = DatasetManifest.model_validate_json((directory / "manifest.json").read_text(encoding="utf-8"))
    for book_id, sha in manifest.books.items():
        book = BookText.load(books_dir / book_id)
        if book.text_sha256 != sha:
            raise DatasetError(
                f"{book_id} 的 text.txt 与评测集 {manifest.name} 标注时的版本不一致，金标区间需要重新校验"
            )
    items = [
        EvalItem.model_validate_json(line)
        for line in (directory / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    return [i for i in items if split is None or i.split == split]


def validate_items(
    items: list[EvalItem], books: dict[str, BookText], *, answer_max_chars: int, copyright_max_substring: int
) -> list[str]:
    """返回所有问题的描述；空列表表示通过。"""
    errors: list[str] = []
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        errors.append("存在重复的题目 id")
    for item in items:
        if len(item.answer) > answer_max_chars:
            errors.append(f"{item.id}：答案超过 {answer_max_chars} 字")
        for ev in item.evidence:
            book = books.get(ev.book_id)
            if book is None:
                errors.append(f"{item.id}：未知的 book_id {ev.book_id}")
                continue
            try:
                ch = book.chapter(ev.chapter_idx)
            except KeyError:
                errors.append(f"{item.id}：{ev.book_id} 没有第 {ev.chapter_idx} 章")
                continue
            if not (ch.char_start <= ev.char_start < ev.char_end <= ch.char_end):
                errors.append(f"{item.id}：证据区间 [{ev.char_start}, {ev.char_end}) 不在第 {ev.chapter_idx} 章内")
        if item.max_reading_order is not None:
            needed = max(
                books[e.book_id].chapter(e.chapter_idx).reading_order for e in item.evidence if e.book_id in books
            )
            if item.max_reading_order < needed:
                errors.append(f"{item.id}：max_reading_order 小于证据所在章节的 reading_order")
        for field_name in ("question", "answer"):
            leaked = copies_source(getattr(item, field_name), books.values(), copyright_max_substring)
            if leaked:
                errors.append(f"{item.id}：{field_name} 中出现原文连续 {copyright_max_substring} 字以上的片段")
    return errors


def copies_source(s: str, books, n: int) -> bool:  # noqa: ANN001
    """s 中是否有长度为 n 的子串原样出现在任一卷正文中。"""
    grams = {s[i : i + n] for i in range(len(s) - n + 1)}
    return any(g in b.text for g in grams for b in books)
