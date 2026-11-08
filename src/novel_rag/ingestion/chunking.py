"""分块：章内按整句贪心累积到目标长度，重叠只取整句，不跨章节，章末过短的尾块并入前一块。

chunk 文本就是 text.txt 的一个切片（可以跨段落，保留段落间的换行）。分块参数连同切句 / 分块规则版本的 hash
就是 chunk 版本号，产物写到 data/books/{book_id}/chunks/{chunk_version}/。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from novel_rag.config import ChunkConfig
from novel_rag.corpus import BookText, Chapter
from novel_rag.ingestion.sentences import SPLITTER_VERSION, split_sentences
from novel_rag.llm.base import TokenCounter

CHUNKER_VERSION = "chunk-1"


class ChunkingError(RuntimeError):
    pass


def chunk_version(cfg: ChunkConfig) -> str:
    payload = {
        "params": cfg.model_dump(exclude={"query_max_tokens"}),
        "splitter": SPLITTER_VERSION,
        "chunker": CHUNKER_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def chunk_spans(book: BookText, chapter: Chapter, cfg: ChunkConfig) -> list[tuple[int, int]]:
    """返回该章所有 chunk 的 [char_start, char_end)。"""
    sents: list[tuple[int, int]] = []
    for ps, pe in book.paragraphs(chapter):
        sents.extend(split_sentences(book.text, ps, pe, cfg))
    if not sents:
        return []
    ranges: list[tuple[int, int]] = []  # 句子下标区间 [i, j)
    i, n = 0, len(sents)
    while i < n:
        start, j = sents[i][0], i
        while j < n:
            new_len = sents[j][1] - start
            if j > i and new_len > cfg.max_chars:
                break
            j += 1
            if new_len >= cfg.target_chars:
                break
        ranges.append((i, j))
        if j >= n:
            break
        k = j  # 重叠：从块尾往回取整句，总长不超过 overlap_chars
        while k - 1 > i and sents[j - 1][1] - sents[k - 1][0] <= cfg.overlap_chars:
            k -= 1
        i = k
    if len(ranges) > 1 and sents[ranges[-1][1] - 1][1] - sents[ranges[-1][0]][0] < cfg.min_chars:
        last = ranges.pop()
        ranges[-1] = (ranges[-1][0], last[1])
    return [(sents[a][0], sents[b - 1][1]) for a, b in ranges]


def _pct(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values)
    return {
        "min": int(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "mean": round(float(arr.mean()), 1),
    }


def build_chunks(
    book: BookText, cfg: ChunkConfig, *, token_counter: TokenCounter | None = None, token_limit: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    version = chunk_version(cfg)
    chunks: list[dict[str, Any]] = []
    for ch in book.chapters:
        for seq, (s, e) in enumerate(chunk_spans(book, ch, cfg)):
            chunks.append(
                {
                    "chunk_id": f"{book.book_id}:{ch.chapter_idx:04d}:{seq:04d}",
                    "book_id": book.book_id,
                    "chapter_idx": ch.chapter_idx,
                    "seq": seq,
                    "char_start": s,
                    "char_end": e,
                    "reading_order": ch.reading_order,
                    "section_type": ch.section_type,
                    "chunk_type": "text",
                    "pov": ch.pov,
                    "chapter_title": ch.title,
                    "text": book.text[s:e],
                }
            )
    check_invariants(book, chunks)
    lengths = [c["char_end"] - c["char_start"] for c in chunks]
    manifest: dict[str, Any] = {
        "chunk_version": version,
        "book_id": book.book_id,
        "text_sha256": book.text_sha256,
        "splitter_version": SPLITTER_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "params": cfg.model_dump(),
        "n_chunks": len(chunks),
        "chars": _pct(lengths),
        "by_section_type": {
            t: sum(c["section_type"] == t for c in chunks) for t in sorted({c["section_type"] for c in chunks})
        },
    }
    if token_counter is not None:
        counts = token_counter.count([c["text"] for c in chunks])
        for c, n in zip(chunks, counts, strict=True):
            c["token_count"] = n
        stats = _pct(counts)
        manifest["tokens"] = {"tokenizer": token_counter.model_name, **stats}
        if token_limit is not None:
            budget = stats["p99"] + cfg.query_max_tokens
            manifest["token_budget"] = {"p99_plus_query": budget, "limit": token_limit, "ok": budget <= token_limit}
    return chunks, manifest


def check_invariants(book: BookText, chunks: list[dict[str, Any]]) -> None:
    """切片一致、不跨章节、正文（非换行）字符全部被覆盖、章内顺序单调。"""
    covered = np.zeros(len(book.text), dtype=bool)
    by_chapter: dict[int, list[dict[str, Any]]] = {}
    for c in chunks:
        if book.text[c["char_start"] : c["char_end"]] != c["text"]:
            raise ChunkingError(f"{c['chunk_id']} 的文本与 text.txt 切片不一致")
        ch = book.chapter(c["chapter_idx"])
        if not (ch.body_start <= c["char_start"] < c["char_end"] <= ch.char_end):
            raise ChunkingError(f"{c['chunk_id']} 越出了第 {ch.chapter_idx} 章正文")
        covered[c["char_start"] : c["char_end"]] = True
        by_chapter.setdefault(c["chapter_idx"], []).append(c)
    for idx, cs in by_chapter.items():
        starts = [c["char_start"] for c in cs]
        if starts != sorted(starts) or [c["seq"] for c in cs] != list(range(len(cs))):
            raise ChunkingError(f"第 {idx} 章的 chunk 顺序不单调")
    for ch in book.chapters:
        body = book.text[ch.body_start : ch.char_end]
        missing = [i for i, t in enumerate(body) if t != "\n" and not covered[ch.body_start + i]]
        if missing:
            raise ChunkingError(f"第 {ch.chapter_idx} 章有 {len(missing)} 个字符没有被任何 chunk 覆盖")


def chunks_dir(books_dir: Path, book_id: str, version: str) -> Path:
    return books_dir / book_id / "chunks" / version


def write_chunks(out_dir: Path, chunks: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chunks.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8", newline=""
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline=""
    )


def series_book_ids(books_dir: Path, version: str) -> list[str]:
    """该分块版本下已切分的全部卷，按 book_id 排序（{books_dir}/{book_id}/chunks/{version}/chunks.jsonl）。"""
    return sorted(p.parents[2].name for p in books_dir.glob(f"*/chunks/{version}/chunks.jsonl"))


def load_books(
    books_dir: Path, book_ids: Sequence[str], version: str
) -> tuple[dict[str, BookText], list[dict[str, Any]]]:
    """加载若干卷的定稿正文与 chunk，并检查 chunk 是在当前 text.txt 上切出来的。chunk 按卷序、章序、seq 排列。"""
    books: dict[str, BookText] = {}
    chunks: list[dict[str, Any]] = []
    for b in book_ids:
        book = BookText.load(books_dir / b)
        cs, manifest = load_chunks(chunks_dir(books_dir, b, version))
        if manifest["text_sha256"] != book.text_sha256:
            raise ChunkingError(f"{b} 的 chunks 与当前 text.txt 不一致，请重新运行 scripts/chunk.py")
        books[b] = book
        chunks.extend(cs)
    chunks.sort(key=lambda c: (c["reading_order"], c["seq"]))
    return books, chunks


def load_chunks(out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (out_dir / "chunks.jsonl").is_file():
        raise ChunkingError(f"{out_dir} 下没有 chunks.jsonl，先运行 scripts/chunk.py")
    chunks = [json.loads(x) for x in (out_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if x]
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    return chunks, manifest
