"""问答运行环境：加载检索范围内各卷的 chunk、卷名与金标用的字符区间，组装工具后端。脚本与集成测试共用。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from novel_rag.config import Settings
from novel_rag.corpus import BookText
from novel_rag.eval.metrics import Span
from novel_rag.ingestion.chunking import chunk_version, chunks_dir, load_chunks
from novel_rag.retrieval.expand import ChunkIndex


@dataclass
class QAEnv:
    settings: Settings
    scope: tuple[str, ...]
    chunk_version: str
    chunks: ChunkIndex
    titles: dict[str, str]
    spans: dict[str, Span]
    text_sha: dict[str, str]


def books_with_chunks(books_dir: Path, version: str) -> list[str]:
    return sorted(p.parents[2].name for p in books_dir.glob(f"*/chunks/{version}/chunks.jsonl"))


def load_env(settings: Settings, scope: tuple[str, ...] | None = None, version: str | None = None) -> QAEnv:
    version = version or chunk_version(settings.chunk)
    scope = scope or tuple(books_with_chunks(settings.paths.books_dir, version))
    all_chunks: list[dict[str, Any]] = []
    titles, spans, shas = {}, {}, {}
    for b in scope:
        chunks, cm = load_chunks(chunks_dir(settings.paths.books_dir, b, version))
        book = BookText.load(settings.paths.books_dir / b)
        if cm["text_sha256"] != book.text_sha256:
            raise RuntimeError(f"{b} 的 chunks 与当前定稿文本不一致")
        titles[b] = book.title
        shas[b] = book.text_sha256
        spans.update({c["chunk_id"]: Span(c["book_id"], c["char_start"], c["char_end"]) for c in chunks})
        all_chunks.extend(chunks)
    return QAEnv(settings, scope, version, ChunkIndex(all_chunks), titles, spans, shas)
