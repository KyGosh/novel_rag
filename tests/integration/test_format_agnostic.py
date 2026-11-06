"""集成测试（只在本地数据上跑）：卷一 epub 导入结果导出成 txt，再用 txt 解析器导入，章节与正文应一致。

运行：uv run pytest -m integration tests/integration/test_format_agnostic.py
"""

import difflib
import json

import pytest

from novel_rag.config import PathsConfig, Settings, get_settings
from novel_rag.ingestion.pipeline import run_ingest

pytestmark = pytest.mark.integration


def _chapters(d):
    return [json.loads(x) for x in (d / "chapters.jsonl").read_text(encoding="utf-8").splitlines() if x]


def test_volume1_epub_vs_txt(tmp_path):
    src = get_settings().paths.books_dir / "asoiaf-1"
    if not (src / "text.txt").is_file():
        pytest.skip("本地没有卷一的导入产物")
    title = json.loads((src / "manifest.json").read_text(encoding="utf-8"))["title"]
    txt = tmp_path / f"{title}.txt"
    original = (src / "text.txt").read_text(encoding="utf-8")
    txt.write_text(original, encoding="utf-8")

    settings = Settings(_env_file=None, paths=PathsConfig(data_dir=tmp_path / "data"))
    res = run_ingest(txt, series_id="fmt", settings=settings)
    assert res.ok, res.report["errors"][:5]
    dst = tmp_path / "data/books/fmt-1"
    a, b = _chapters(src), _chapters(dst)
    assert len(a) == len(b)
    assert [(c["title"], c["section_type"], c["reading_order"]) for c in a] == [
        (c["title"], c["section_type"], c["reading_order"]) for c in b
    ]
    again = (dst / "text.txt").read_text(encoding="utf-8")
    ratio = difflib.SequenceMatcher(None, original, again, autojunk=False).quick_ratio()
    assert again == original or ratio > 0.999
