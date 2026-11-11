"""知识图谱各步骤共用的加载逻辑：系列级书籍配置 → schema 与名称规则；产物目录。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from novel_rag.ingestion.book_config import BookConfig, load_book_config
from novel_rag.kg.names import NameRules
from novel_rag.kg.schema import Schema


@dataclass(frozen=True)
class SeriesKG:
    series_id: str
    book_config: BookConfig
    schema: Schema
    rules: NameRules


def series_of(books_dir: Path, book_id: str) -> str:
    manifest = books_dir / book_id / "manifest.json"
    if manifest.is_file():
        sid = json.loads(manifest.read_text(encoding="utf-8")).get("series_id")
        if sid:
            return sid
    return book_id.rsplit("-", 1)[0]


def load_series_kg(books_dir: Path, book_ids: list[str]) -> SeriesKG:
    sids = {series_of(books_dir, b) for b in book_ids}
    if len(sids) != 1:
        raise ValueError(f"一次只能处理同一个系列的卷：{sorted(sids)}")
    sid = sids.pop()
    cfg = load_book_config(books_dir / sid / "book.toml")
    return SeriesKG(
        sid,
        cfg,
        Schema(extensions=cfg.kg.extensions),
        NameRules.with_extra(cfg.kg.honorifics, cfg.kg.generic_terms),
    )


def kg_root(data_dir: Path) -> Path:
    return data_dir / "kg"


def extract_dir(data_dir: Path, version: str) -> Path:
    return kg_root(data_dir) / "extract" / version
