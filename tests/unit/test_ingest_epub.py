"""epub 解析器：目录（NCX / nav）、章节与目录交叉校验、注释配对、内联拼接、安全限制。样例均为自拟。"""

import zipfile

import pytest

from novel_rag.config import IngestConfig
from novel_rag.ingestion.ir import NOTE_MARK, IngestError
from novel_rag.ingestion.parsers.epub import load_xhtml, parse_epub
from tests.unit.epub_factory import Toc, build_epub, three_part_book

CFG = IngestConfig()


def _titles(doc):
    return [(p.title, [s.title for s in p.sections]) for p in doc.parts]


def _all_blocks(doc):
    return [b for p in doc.parts for s in p.sections for b in s.blocks]


def test_parts_and_sections_follow_toc(tmp_path):
    doc = parse_epub(three_part_book(tmp_path / "b.epub"), CFG)
    assert [i for i in doc.issues if i.level == "error"] == []
    assert _titles(doc) == [
        (None, [None]),  # 目录之前的作者简介
        ("星海纪事（上）", [None, "序幕", "阿岚"]),  # 序幕的 h2 没有锚点，按标题顺序对齐
        ("星海纪事（下）", [None, "阿岚", "附录"]),
        ("远航之歌", [None, "阿岚", "琪娜"]),
    ]


def test_ncx_and_nav_give_same_structure(tmp_path):
    a = parse_epub(three_part_book(tmp_path / "a.epub", toc_kind="ncx"), CFG)
    b = parse_epub(three_part_book(tmp_path / "b.epub", toc_kind="nav"), CFG)
    assert _titles(a) == _titles(b)
    assert [x.text for x in _all_blocks(a)] == [x.text for x in _all_blocks(b)]


def test_bidirectional_note_is_stripped_and_attached(tmp_path):
    doc = parse_epub(three_part_book(tmp_path / "b.epub"), CFG)
    blocks = _all_blocks(doc)
    marked = [b for b in blocks if b.notes]
    assert len(marked) == 1
    b = marked[0]
    assert b.text == f"守望者{NOTE_MARK}站在高塔上，夜色渐浓。"
    assert b.notes[0].label == "[1]" and b.notes[0].text == "守望者：驻守边境的卫士。"
    assert not any("驻守边境" in x.text for x in blocks)  # 注释段落已从正文移除
    assert doc.stats["notes_by_file"]["OEBPS/Text/c1.xhtml"]["paired"] == 1


def test_note_in_another_file_with_circled_label(tmp_path):
    files = {
        "Text/c.xhtml": '<h2>一</h2><p>他说了一句双关语<a id="cf1" href="notes.xhtml#f1">①</a>。</p>',
        "Text/notes.xhtml": '<p><a id="f1" href="c.xhtml#cf1">①</a><span>这里是双关。</span></p>',
    }
    doc = parse_epub(build_epub(tmp_path / "b.epub", files, [Toc("一", "Text/c.xhtml")]), CFG)
    blocks = _all_blocks(doc)
    assert [b.notes[0].text for b in blocks if b.notes] == ["这里是双关。"]
    assert not any("这里是双关" in b.text for b in blocks)


def test_unpaired_marker_is_kept_and_reported(tmp_path):
    files = {"Text/c.xhtml": '<h2>一</h2><p>文字<a href="c.xhtml#missing"><sup>[3]</sup></a>继续。</p>'}
    doc = parse_epub(build_epub(tmp_path / "b.epub", files, [Toc("一", "Text/c.xhtml")]), CFG)
    assert any(b.text == "文字[3]继续。" for b in _all_blocks(doc))
    assert [i.code for i in doc.issues] == ["unpaired_marker"]


def test_sup_marker_without_links_falls_back_to_label_order(tmp_path):
    files = {"Text/c.xhtml": "<h2>一</h2><p>前文<sup>[1]</sup>后文。</p><p>[1]无链接的注释。</p>"}
    doc = parse_epub(build_epub(tmp_path / "b.epub", files, [Toc("一", "Text/c.xhtml")]), CFG)
    blocks = _all_blocks(doc)
    assert [b.notes[0].text for b in blocks if b.notes] == ["无链接的注释。"]
    assert len([b for b in blocks if b.kind == "paragraph"]) == 1


def test_inline_spans_are_concatenated_and_self_closing_anchor_kept(tmp_path):
    doc = parse_epub(three_part_book(tmp_path / "b.epub"), CFG)
    texts = [b.text for b in _all_blocks(doc)]
    assert "阿岚•星川望向远方。" in texts  # 间隔号单独成 span，拼接后再由规范化统一
    # <a id="x"/> 自闭合锚点不能吞掉后续文字；&nbsp; 被识别
    assert "阿岚走进大厅 。他想起了往事。" in texts


def test_image_block_and_resource(tmp_path):
    doc = parse_epub(three_part_book(tmp_path / "b.epub"), CFG)
    images = [b for b in _all_blocks(doc) if b.kind == "image"]
    assert [b.image_src for b in images] == ["OEBPS/Images/pic.jpg"]
    assert "OEBPS/Images/pic.jpg" in doc.resources


def test_toc_count_mismatch_is_error(tmp_path):
    files = {"Text/c.xhtml": "<h2>一</h2><p>甲</p><h2>二</h2><p>乙</p>"}
    toc = [Toc("卷", "Text/c.xhtml", [Toc("一", "Text/c.xhtml"), Toc("二", "Text/c.xhtml"), Toc("三", "Text/c.xhtml")])]
    doc = parse_epub(build_epub(tmp_path / "b.epub", files, toc), CFG)
    assert "toc_count_mismatch" in [i.code for i in doc.issues if i.level == "error"]


def test_toc_title_mismatch_is_error(tmp_path):
    files = {"Text/c.xhtml": "<h2>一</h2><p>甲</p><h2>二</h2><p>乙</p>"}
    toc = [Toc("卷", "Text/c.xhtml", [Toc("一", "Text/c.xhtml"), Toc("完全不同", "Text/c.xhtml")])]
    doc = parse_epub(build_epub(tmp_path / "b.epub", files, toc), CFG)
    assert "toc_title_mismatch" in [i.code for i in doc.issues if i.level == "error"]


def test_part_start_snaps_to_file_start_before_short_lead(tmp_path):
    """目录指向标题页中间的 h1 时，h1 之前的"第X卷"仍属于本分册，而不是上一章的结尾。"""
    files = {
        "Text/c1.xhtml": "<h2>一</h2><p>第一卷的正文。</p>",
        "Text/t2.xhtml": '<div>【第二卷】</div><h1 id="h">乙书</h1>',
        "Text/c2.xhtml": "<h2>二</h2><p>第二卷的正文。</p>",
    }
    toc = [
        Toc("甲书", "Text/c1.xhtml", [Toc("一", "Text/c1.xhtml")]),
        Toc("乙书", "Text/t2.xhtml#h", [Toc("二", "Text/c2.xhtml")]),
    ]
    doc = parse_epub(build_epub(tmp_path / "b.epub", files, toc), CFG)
    first_chapter = doc.parts[0].sections[-1]
    assert [b.text for b in first_chapter.blocks] == ["一", "第一卷的正文。"]
    assert doc.parts[1].sections[0].blocks[0].text == "【第二卷】"


def test_flat_toc_is_single_part(tmp_path):
    files = {"Text/c.xhtml": "<h2>一</h2><p>甲</p><h2>二</h2><p>乙</p>"}
    toc = [Toc("一", "Text/c.xhtml"), Toc("二", "Text/c.xhtml")]
    doc = parse_epub(build_epub(tmp_path / "b.epub", files, toc, title="单册书"), CFG)
    assert _titles(doc) == [("单册书", ["一", "二"])]


def test_hidden_and_script_content_skipped(tmp_path):
    files = {"Text/c.xhtml": '<h2>一</h2><p>可见</p><p style="display:none">隐藏</p><script>var x=1;</script>'}
    doc = parse_epub(build_epub(tmp_path / "b.epub", files, [Toc("一", "Text/c.xhtml")]), CFG)
    assert [b.text for b in _all_blocks(doc)] == ["一", "可见"]


def test_external_entities_are_not_resolved():
    data = b'<?xml version="1.0"?><!DOCTYPE html [<!ENTITY x SYSTEM "file:///etc/passwd">]><html><body><p>a&x;b</p></body></html>'
    root, _ = load_xhtml(data)
    assert "root:" not in "".join(root.itertext())


# ---------------------------------------------------------------- 安全限制


def test_zip_limits(tmp_path):
    p = build_epub(tmp_path / "b.epub", {"Text/c.xhtml": "<h2>一</h2><p>甲</p>"}, [Toc("一", "Text/c.xhtml")])
    with pytest.raises(IngestError, match="文件数"):
        parse_epub(p, IngestConfig(max_archive_files=3))
    with pytest.raises(IngestError, match="总大小"):
        parse_epub(p, IngestConfig(max_uncompressed_bytes=100))


def test_path_traversal_rejected(tmp_path):
    p = build_epub(
        tmp_path / "b.epub", {"Text/c.xhtml": "<p>甲</p>"}, [Toc("一", "Text/c.xhtml")], extra={"../evil.txt": b"x"}
    )
    with pytest.raises(IngestError, match="非法路径"):
        parse_epub(p, CFG)


def test_drm_detected_but_font_obfuscation_allowed(tmp_path):
    def enc(algorithm: str) -> bytes:
        return (
            '<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
            'xmlns:enc="http://www.w3.org/2001/04/xmlenc#"><enc:EncryptedData>'
            f'<enc:EncryptionMethod Algorithm="{algorithm}"/></enc:EncryptedData></encryption>'
        ).encode()

    files, toc = {"Text/c.xhtml": "<h2>一</h2><p>甲</p>"}, [Toc("一", "Text/c.xhtml")]
    drm = build_epub(
        tmp_path / "drm.epub",
        files,
        toc,
        extra={"META-INF/encryption.xml": enc("http://www.w3.org/2001/04/xmlenc#aes128-cbc")},
    )
    with pytest.raises(IngestError, match="DRM"):
        parse_epub(drm, CFG)
    font = build_epub(
        tmp_path / "font.epub", files, toc, extra={"META-INF/encryption.xml": enc("http://www.idpf.org/2008/embedding")}
    )
    assert parse_epub(font, CFG).parts


def test_not_a_zip(tmp_path):
    p = tmp_path / "x.epub"
    p.write_bytes(b"not a zip")
    with pytest.raises(IngestError):
        parse_epub(p, CFG)
    assert not zipfile.is_zipfile(p)
