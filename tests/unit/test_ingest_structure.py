"""结构识别、拼接全文与落盘：卷合并、卷号、section_type、章节号、POV、reading_order、偏移不变量、定稿保护。"""

import pytest

from novel_rag.config import IngestConfig, PathsConfig, Settings
from novel_rag.ingestion.assemble import LockMismatchError, assemble_volume, read_lock, write_volume
from novel_rag.ingestion.book_config import BookConfig, BookConfigError, load_book_config
from novel_rag.ingestion.normalize import normalize_document
from novel_rag.ingestion.parsers.epub import parse_epub
from novel_rag.ingestion.pipeline import run_ingest
from novel_rag.ingestion.structure import build_volumes, parse_cn_number, strip_part_suffix
from tests.unit.epub_factory import three_part_book

CFG = IngestConfig()


def _volumes(tmp_path, book=None, cfg=CFG, **kw):
    doc = parse_epub(three_part_book(tmp_path / "b.epub", **kw), cfg)
    normalize_document(doc, sample_size=4, qmark_sample_size=4, qmark_punct_ratio=cfg.qmark_punct_ratio)
    return doc, build_volumes(doc, "demo", cfg, book or BookConfig())


@pytest.mark.parametrize(
    ("s", "n"), [("一", 1), ("十", 10), ("十二", 12), ("二十", 20), ("二十三", 23), ("一百零五", 105), ("7", 7)]
)
def test_parse_cn_number(s, n):
    assert parse_cn_number(s) == n


def test_strip_part_suffix():
    assert strip_part_suffix("星海纪事（上）") == "星海纪事"
    assert strip_part_suffix("星海纪事(下册)") == "星海纪事"
    assert strip_part_suffix("星海纪事上部") == "星海纪事"
    assert strip_part_suffix("上海") == "上海"


def test_parts_merged_into_volumes_with_numbers(tmp_path):
    doc, vols = _volumes(tmp_path)
    assert [(v.number, v.number_source, v.title, v.book_id, len(v.part_indices)) for v in vols] == [
        (1, "title_page", "星海纪事", "demo-1", 2),
        (2, "title_page", "远航之歌", "demo-2", 1),
    ]
    assert [i for i in doc.issues if i.level == "error"] == []


def test_volume_number_falls_back_to_order(tmp_path):
    _, vols = _volumes(tmp_path, volume_label="没有卷号")
    assert [(v.number, v.number_source) for v in vols] == [(1, "order"), (2, "title_page")]


def test_chapter_numbering_types_and_reading_order(tmp_path):
    _, vols = _volumes(tmp_path)
    v1, v2 = vols
    assert [(c.chapter_idx, c.title, c.section_type) for c in v1.chapters] == [
        (0, "序幕", "narrative"),  # 序幕记为 0
        (1, "阿岚", "narrative"),
        (2, "阿岚", "narrative"),
        (3, "附录", "appendix"),  # 附录排在正文之后
    ]
    assert [(c.chapter_idx, c.title) for c in v2.chapters] == [(1, "阿岚"), (2, "琪娜")]
    assert [c.reading_order for v in vols for c in v.chapters] == list(range(6))  # 全系列连续
    assert {o.section_type for o in v1.others} >= {"front_matter", "volume_title"}


def test_pov_heuristic_and_config(tmp_path):
    _, vols = _volumes(tmp_path)
    povs = {(c.title, c.pov) for v in vols for c in v.chapters}
    assert ("阿岚", "阿岚") in povs  # 作为章节标题出现 3 次
    assert ("琪娜", None) in povs and ("序幕", None) in povs
    _, vols = _volumes(tmp_path, book=BookConfig(pov={"琪娜": "琪娜", "阿岚": None}))
    povs = {(c.title, c.pov) for v in vols for c in v.chapters}
    assert ("琪娜", "琪娜") in povs and ("阿岚", None) in povs


def test_config_overrides_volume_title_and_section_type(tmp_path):
    book = BookConfig(volume_titles={1: "星海纪事·修订"}, section_types={(None, "附录"): "back_matter"})
    _, vols = _volumes(tmp_path, book=book)
    assert (vols[0].title, vols[0].title_source) == ("星海纪事·修订", "config")
    assert "附录" not in [c.title for c in vols[0].chapters]  # back_matter 不写入 text.txt


def test_load_book_config(tmp_path):
    assert load_book_config(tmp_path / "missing.toml") == BookConfig()
    p = tmp_path / "book.toml"
    p.write_text('[[volumes]]\nnumber = 1\ntitle = "甲"\n[pov]\n"别称" = "某人"\n"非人" = ""\n', encoding="utf-8")
    cfg = load_book_config(p)
    assert cfg.volume_titles == {1: "甲"} and cfg.pov == {"别称": "某人", "非人": None}
    p.write_text("[typo]\nx = 1\n", encoding="utf-8")
    with pytest.raises(BookConfigError):
        load_book_config(p)


# ---------------------------------------------------------------- 拼接与落盘


def _outputs(tmp_path):
    doc, vols = _volumes(tmp_path)
    return [assemble_volume(v, doc, "demo", CFG.text_section_types) for v in vols]


def test_text_layout_and_offsets(tmp_path):
    out = _outputs(tmp_path)[0]
    text = out.text
    assert text.startswith("序幕\n守望者站在高塔上，夜色渐浓。\n阿岚·星川望向远方。\n\n阿岚\n")
    assert text.endswith("\n")
    for c in out.chapters:
        assert text[c["char_start"] : c["title_end"]] == c["title"]
        assert text[c["char_end"]] == "\n"
    # 注释锚点：标记原本所在的位置
    note = out.notes[0]
    assert text[note["offset"] - 3 : note["offset"]] == "守望者" and note["text"] == "守望者：驻守边境的卫士。"
    # 插图锚点落在序幕末尾（下一行开头）
    img = out.image_anchors[0]
    assert img["chapter_idx"] == 0 and text[img["offset"] - 1] == "\n"
    # 卷内第二册的半角问号被统一为全角
    assert "我们去哪？”" in text
    assert "高塔 北境要塞" in text


def test_assemble_is_deterministic(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, b = _outputs(tmp_path / "a"), _outputs(tmp_path / "b")
    assert [(o.text, o.chapters, o.notes, o.manifest) for o in a] == [
        (o.text, o.chapters, o.notes, o.manifest) for o in b
    ]


def test_lock_protects_finalized_text(tmp_path):
    out = _outputs(tmp_path)[0]
    book_dir = tmp_path / "books" / out.volume.book_id
    write_volume(book_dir, out, finalize=True)
    assert read_lock(book_dir)["text_sha256"] == out.text_sha256
    write_volume(book_dir, out)  # 内容相同可以重写
    out.text_sha256 = "different"
    with pytest.raises(LockMismatchError):
        write_volume(book_dir, out)
    write_volume(book_dir, out, force=True)
    assert read_lock(book_dir)["text_sha256"] == "different"


def test_pipeline_end_to_end(tmp_path):
    settings = Settings(_env_file=None, paths=PathsConfig(data_dir=tmp_path / "data"))
    epub = three_part_book(tmp_path / "b.epub")
    result = run_ingest(epub, series_id="demo", settings=settings, finalize={1})
    assert result.ok and result.written == ["demo-1", "demo-2"]
    assert (tmp_path / "data/books/demo-1/text.txt").is_file()
    assert (tmp_path / "data/books/demo-1/text.lock").is_file()
    assert not (tmp_path / "data/books/demo-2/text.lock").exists()
    assert result.report_md.is_file()
    again = run_ingest(epub, series_id="demo", settings=settings)
    v1 = next(v for v in again.report["volumes"] if v["book_id"] == "demo-1")
    assert v1["text"]["lock"] == "match" and not v1["text"]["changed"]
