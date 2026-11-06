"""把一卷的章节拼成规范化全文 text.txt，并计算章节、注释、插图锚点的字符偏移；负责落盘与定稿保护。

text.txt 格式：每章标题独占一行，其后每个段落（及章内小标题）一行；章与章之间空一行；文件以换行结尾。
注释和插图不进入正文，只记录锚点偏移。所有产物对同一输入逐字节确定。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from novel_rag.ingestion.ir import NOTE_MARK, Block, Document
from novel_rag.ingestion.normalize import NORMALIZE_VERSION
from novel_rag.ingestion.structure import Volume


@dataclass
class VolumeOutput:
    volume: Volume
    text: str
    text_sha256: str
    chapters: list[dict] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    image_anchors: list[dict] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)


class LockMismatchError(RuntimeError):
    """text.txt 已定稿，但本次生成的内容与锁定的 hash 不一致。"""


def assemble_volume(vol: Volume, doc: Document, series_id: str, text_section_types: tuple[str, ...]) -> VolumeOutput:
    pieces: list[str] = []
    pos = 0
    chapters, notes, images = [], [], []

    def add_line(block_text: str, block_notes: list, chapter_idx: int, locator: str) -> None:
        nonlocal pos
        clean, k = [], 0
        for ch in block_text:
            if ch == NOTE_MARK:
                note = block_notes[k]
                k += 1
                notes.append(
                    {
                        "note_id": f"{vol.book_id}:n{len(notes) + 1:04d}",
                        "book_id": vol.book_id,
                        "chapter_idx": chapter_idx,
                        "offset": pos + len(clean),
                        "label": note.label,
                        "text": note.text,
                        "marker_locator": note.marker_locator or locator,
                        "note_locator": note.note_locator,
                    }
                )
            else:
                clean.append(ch)
        line = "".join(clean)
        pieces.append(line + "\n")
        pos += len(line) + 1

    for ci, c in enumerate(vol.chapters):
        if ci > 0:
            pieces.append("\n")
            pos += 1
        blocks: list[Block] = c.section.blocks
        start = pos
        if blocks and blocks[0].kind == "heading":
            add_line(blocks[0].text, blocks[0].notes, c.chapter_idx, blocks[0].locator)
            body = blocks[1:]
        else:
            add_line(c.title, [], c.chapter_idx, c.section.locator)
            body = blocks
        title_end = pos - 1
        n_par = 0
        for b in body:
            if b.kind == "image":
                images.append(
                    {
                        "book_id": vol.book_id,
                        "chapter_idx": c.chapter_idx,
                        "offset": pos,
                        "src": b.image_src,
                        "locator": b.locator,
                    }
                )
                continue
            add_line(b.text, b.notes, c.chapter_idx, b.locator)
            n_par += b.kind == "paragraph"
        chapters.append(
            {
                "book_id": vol.book_id,
                "chapter_idx": c.chapter_idx,
                "title": c.title,
                "pov": c.pov,
                "section_type": c.section_type,
                "reading_order": c.reading_order,
                "char_start": start,
                "char_end": pos - 1,  # 不含章末换行
                "title_end": title_end,
                "n_paragraphs": n_par,
                "source_locator": c.section.locator,
            }
        )
    text = "".join(pieces)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest = {
        "book_id": vol.book_id,
        "series_id": series_id,
        "volume_number": vol.number,
        "volume_number_source": vol.number_source,
        "title": vol.title,
        "title_source": vol.title_source,
        "raw_part_titles": vol.raw_titles,
        "source": {"name": doc.source_name, "sha256": doc.source_sha256, "format": doc.format},
        "parser_version": doc.parser_version,
        "normalize_version": NORMALIZE_VERSION,
        "text_section_types": list(text_section_types),
        "text": {"sha256": sha, "chars": len(text)},
        "counts": {
            "chapters": len(chapters),
            "narrative": sum(c["section_type"] == "narrative" for c in chapters),
            "appendix": sum(c["section_type"] == "appendix" for c in chapters),
            "notes": len(notes),
            "image_anchors": len(images),
        },
        "reading_order": [chapters[0]["reading_order"], chapters[-1]["reading_order"]] if chapters else None,
    }
    return VolumeOutput(vol, text, sha, chapters, notes, images, manifest)


# ---------------------------------------------------------------- 落盘


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def read_lock(book_dir: Path) -> dict | None:
    p = book_dir / "text.lock"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def check_lock(book_dir: Path, out: VolumeOutput) -> str:
    """返回定稿状态：unlocked / match / mismatch。"""
    lock = read_lock(book_dir)
    if lock is None:
        return "unlocked"
    return "match" if lock.get("text_sha256") == out.text_sha256 else "mismatch"


def write_volume(book_dir: Path, out: VolumeOutput, *, force: bool = False, finalize: bool = False) -> None:
    if check_lock(book_dir, out) == "mismatch" and not force:
        raise LockMismatchError(
            f"{out.volume.book_id} 的 text.txt 已定稿，本次结果与锁定的 hash 不一致；"
            "已标注的评测金标需要重新定位校验。确认后用 --force 覆盖。"
        )
    book_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "text.txt": out.text,
        "chapters.jsonl": _jsonl(out.chapters),
        "notes.jsonl": _jsonl(out.notes),
        "manifest.json": json.dumps(out.manifest, ensure_ascii=False, indent=2) + "\n",
    }
    for name, content in files.items():
        (book_dir / name).write_text(content, encoding="utf-8", newline="")
    if finalize or (force and read_lock(book_dir) is not None):
        lock = {
            "book_id": out.volume.book_id,
            "text_sha256": out.text_sha256,
            "source_sha256": out.manifest["source"]["sha256"],
            "parser_version": out.manifest["parser_version"],
            "normalize_version": out.manifest["normalize_version"],
        }
        (book_dir / "text.lock").write_text(
            json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline=""
        )
