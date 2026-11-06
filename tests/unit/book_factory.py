"""测试用的合成书籍目录（text.txt + chapters.jsonl + text.lock），布局与 ingestion/assemble.py 一致。内容全部自拟。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def make_book(
    books_dir: Path,
    book_id: str,
    chapters: list[tuple[str, list[str]]],
    *,
    section_types: dict[int, str] | None = None,
    lock: bool = True,
    first_idx: int = 1,
) -> Path:
    """chapters：[(标题, [段落, ...]), ...]；章节号从 first_idx 起连续编号。"""
    section_types = section_types or {}
    book_dir = books_dir / book_id
    book_dir.mkdir(parents=True, exist_ok=True)
    pieces, rows, pos = [], [], 0
    for k, (title, paras) in enumerate(chapters):
        if k:
            pieces.append("\n")
            pos += 1
        start = pos
        for line in [title, *paras]:
            pieces.append(line + "\n")
            pos += len(line) + 1
        idx = first_idx + k
        rows.append(
            {
                "book_id": book_id,
                "chapter_idx": idx,
                "title": title,
                "pov": title if len(title) <= 4 else None,
                "section_type": section_types.get(idx, "narrative"),
                "reading_order": k,
                "char_start": start,
                "char_end": pos - 1,
                "title_end": start + len(title),
                "n_paragraphs": len(paras),
                "source_locator": "",
            }
        )
    text = "".join(pieces)
    (book_dir / "text.txt").write_text(text, encoding="utf-8", newline="")
    (book_dir / "chapters.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline=""
    )
    if lock:
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        (book_dir / "text.lock").write_text(json.dumps({"book_id": book_id, "text_sha256": sha}), encoding="utf-8")
    return book_dir


def para(seed: str, n: int) -> str:
    """长度约为 n 的自拟段落。"""
    base = f"{seed}在港口边等待船队归来，风从海面吹来，带着盐和鱼腥的味道。"
    return (base * (n // len(base) + 1))[: n - 1] + "。"
