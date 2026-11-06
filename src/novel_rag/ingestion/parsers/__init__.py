"""按扩展名选择解析器。所有解析器输出同一种中间表示（ir.Document）。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from novel_rag.config import IngestConfig
from novel_rag.ingestion.ir import Document, IngestError
from novel_rag.ingestion.parsers.epub import parse_epub

PARSERS: dict[str, Callable[[Path, IngestConfig], Document]] = {".epub": parse_epub}


def parse_file(path: Path, cfg: IngestConfig) -> Document:
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        raise IngestError(f"不支持的格式：{path.suffix}（支持 {sorted(PARSERS)}）")
    return parser(path, cfg)
