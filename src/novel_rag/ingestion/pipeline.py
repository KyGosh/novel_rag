"""导入流水线：解析 → 规范化 → 结构识别 → 拼接全文 → 质检 →（通过才）落盘。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from novel_rag.config import Settings, get_settings
from novel_rag.ingestion.assemble import assemble_volume, check_lock, write_volume
from novel_rag.ingestion.book_config import load_book_config
from novel_rag.ingestion.normalize import normalize_document
from novel_rag.ingestion.parsers import parse_file
from novel_rag.ingestion.qc import build_report, render_markdown
from novel_rag.ingestion.structure import build_volumes


@dataclass
class IngestResult:
    ok: bool
    report: dict[str, Any]
    report_json: Path
    report_md: Path
    written: list[str] = field(default_factory=list)


def default_series_id(path: Path, source_sha256: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    return slug[:40] if len(slug) >= 3 else f"book-{source_sha256[:8]}"


def run_ingest(
    path: Path,
    *,
    series_id: str | None = None,
    settings: Settings | None = None,
    volumes: set[int] | None = None,
    finalize: set[int] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> IngestResult:
    s = settings or get_settings()
    cfg = s.ingest
    doc = parse_file(path, cfg)
    sid = series_id or default_series_id(path, doc.source_sha256)
    book = load_book_config(s.paths.books_dir / sid / "book.toml")

    part_stats = normalize_document(
        doc,
        sample_size=cfg.qc_sample_size,
        qmark_sample_size=cfg.qc_qmark_sample_size,
        qmark_punct_ratio=cfg.qmark_punct_ratio,
    )
    vols = build_volumes(doc, sid, cfg, book)
    if volumes:
        vols = [v for v in vols if v.number in volumes]
    outputs = [assemble_volume(v, doc, sid, cfg.text_section_types) for v in vols]

    previous: dict[str, str | None] = {}
    locks: dict[str, str] = {}
    for out in outputs:
        book_dir = s.paths.books_dir / out.volume.book_id
        manifest = book_dir / "manifest.json"
        previous[out.volume.book_id] = (
            json.loads(manifest.read_text(encoding="utf-8"))["text"]["sha256"] if manifest.is_file() else None
        )
        locks[out.volume.book_id] = check_lock(book_dir, out)

    report = build_report(
        doc,
        sid,
        part_stats,
        outputs,
        previous,
        locks,
        book_config_source=book.source,
        sample_size=cfg.qc_sample_size,
        force=force,
    )
    written: list[str] = []
    ok = report["status"] == "pass"
    if ok and not dry_run:
        for out in outputs:
            write_volume(
                s.paths.books_dir / out.volume.book_id,
                out,
                force=force,
                finalize=bool(finalize) and out.volume.number in finalize,
            )
            written.append(out.volume.book_id)
    report["written"] = written
    report["dry_run"] = dry_run

    s.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rj = s.paths.reports_dir / f"ingest_{sid}_{stamp}.json"
    rm = rj.with_suffix(".md")
    rj.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rm.write_text(render_markdown(report), encoding="utf-8")
    return IngestResult(ok, report, rj, rm, written)
