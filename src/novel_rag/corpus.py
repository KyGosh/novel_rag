"""读取一卷的导入产物（text.txt + chapters.jsonl + text.lock）。评测集构建、分块等下游步骤都经由这里访问正文。

字符偏移一律以 text.txt 为准；章节是偏移表。定稿（text.lock）后才允许在其上标注评测金标。
"""

from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class CorpusError(RuntimeError):
    pass


@dataclass(frozen=True)
class Chapter:
    book_id: str
    chapter_idx: int
    title: str
    pov: str | None
    section_type: str
    reading_order: int
    char_start: int
    char_end: int  # 不含章末换行
    title_end: int

    @property
    def body_start(self) -> int:
        return self.title_end + 1


@dataclass(frozen=True)
class BookText:
    book_id: str
    text: str
    text_sha256: str
    chapters: tuple[Chapter, ...]
    locked: bool

    @classmethod
    def load(cls, book_dir: Path, *, require_lock: bool = True) -> BookText:
        text_path, chapters_path = book_dir / "text.txt", book_dir / "chapters.jsonl"
        if not text_path.is_file() or not chapters_path.is_file():
            raise CorpusError(f"{book_dir} 下没有 text.txt / chapters.jsonl，先运行 scripts/ingest.py")
        text = text_path.read_text(encoding="utf-8")
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        lock_path = book_dir / "text.lock"
        locked = lock_path.is_file()
        if locked and json.loads(lock_path.read_text(encoding="utf-8"))["text_sha256"] != sha:
            raise CorpusError(f"{book_dir / 'text.txt'} 与 text.lock 中定稿的 hash 不一致")
        if require_lock and not locked:
            raise CorpusError(f"{book_dir.name} 的 text.txt 尚未定稿（缺少 text.lock）")
        rows = [json.loads(line) for line in chapters_path.read_text(encoding="utf-8").splitlines() if line]
        chapters = tuple(Chapter(**{k: r[k] for k in Chapter.__dataclass_fields__}) for r in rows)
        return cls(book_dir.name, text, sha, chapters, locked)

    def chapter(self, chapter_idx: int) -> Chapter:
        for c in self.chapters:
            if c.chapter_idx == chapter_idx:
                return c
        raise KeyError(f"{self.book_id} 没有第 {chapter_idx} 章")

    def chapter_at(self, offset: int) -> Chapter | None:
        starts = [c.char_start for c in self.chapters]
        i = bisect.bisect_right(starts, offset) - 1
        if i >= 0 and self.chapters[i].char_start <= offset <= self.chapters[i].char_end:
            return self.chapters[i]
        return None

    def paragraphs(self, chapter: Chapter) -> list[tuple[int, int]]:
        """章节正文（不含标题行）中每一行的 [start, end)。"""
        spans, pos = [], chapter.body_start
        while pos < chapter.char_end:
            nl = self.text.find("\n", pos, chapter.char_end)
            end = chapter.char_end if nl == -1 else nl
            if end > pos:
                spans.append((pos, end))
            pos = end + 1
        return spans

    def context(self, start: int, end: int, width: int = 150) -> tuple[str, str, str]:
        """（前文, 区间本身, 后文），前后文不越出所在章节。"""
        c = self.chapter_at(start)
        lo, hi = (c.char_start, c.char_end) if c else (0, len(self.text))
        return self.text[max(lo, start - width) : start], self.text[start:end], self.text[end : min(hi, end + width)]
