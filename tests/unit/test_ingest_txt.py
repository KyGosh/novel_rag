"""txt 解析器：编码探测、分段方式、卷 / 章标题识别（规则与启发式）；格式无关性（epub 导出成 txt 再导入，正文不变）。
全部用自拟样例。"""

import pytest

from novel_rag.config import IngestConfig, PathsConfig, Settings
from novel_rag.ingestion.ir import IngestError
from novel_rag.ingestion.parsers import parse_file
from novel_rag.ingestion.parsers.txt import decode, parse_txt
from novel_rag.ingestion.pipeline import run_ingest
from tests.unit.epub_factory import three_part_book

CFG = IngestConfig()
BODY = "她推开门，看见院子里落满了雪。“你来了。”老人说，"


def _para(i: int) -> str:
    return f"{BODY}这是第{i}段，写得足够长，用来充当正文。"


def _pov_book() -> str:
    """一行一段、章节标题是人名（无编号）、章与章之间空一行；含序幕与附录。"""
    out = ["序幕", *[_para(i) for i in range(3)]]
    for n, name in enumerate(["阿岚", "白舟", "阿岚", "白舟", "阿岚", "白舟", "孤灯"]):
        out += ["", name, *[_para(n * 10 + i) for i in range(4)]]
        out.append("雪")  # 短段落但前后没有空行、也不重复三次：仍是正文
    out += ["", "附录", "人物表", "阿岚家族", "白舟家族"]
    return "\n".join(out) + "\n"


def _titles(doc):
    return [(p.title, [s.title for s in p.sections]) for p in doc.parts]


def test_decode_variants():
    s = "第一章 雪夜"
    assert decode(s.encode("utf-8")) == (s, "utf-8")
    assert decode(b"\xef\xbb\xbf" + s.encode("utf-8")) == (s, "utf-8-sig")
    assert decode(s.encode("utf-16")) == (s, "utf-16")
    assert decode(s.encode("gb18030")) == (s, "gb18030")
    with pytest.raises(IngestError):
        decode(b"\xff\xff\xfa\xfb\x80\x81" * 3)


def test_pov_titles_by_blank_lines_and_repetition(tmp_path):
    p = tmp_path / "孤灯集.txt"
    p.write_text(_pov_book(), encoding="gb18030")
    doc = parse_file(p, CFG)
    assert doc.format == "txt" and doc.metadata["encoding"] == "gb18030"
    assert doc.stats["txt"]["paragraph_mode"] == "line"
    ((title, sections),) = _titles(doc)
    assert title == "孤灯集"
    assert sections == ["序幕", "阿岚", "白舟", "阿岚", "白舟", "阿岚", "白舟", "孤灯", "附录"]
    first = doc.parts[0].sections[1]
    assert first.blocks[0].kind == "heading" and first.blocks[-1].text == "雪"  # 短正文段没有被当成标题
    appendix = doc.parts[0].sections[-1]
    assert [b.text for b in appendix.blocks[1:]] == ["人物表", "阿岚家族", "白舟家族"]


def test_numbered_chapters_and_volumes_blank_separated(tmp_path):
    lines = []
    for v, vname in enumerate(["第一卷 风起", "第二卷 云涌"], 1):
        lines += [vname, ""]
        for c in range(1, 3):
            lines += [f"第{c}章 {vname[-2:]}{c}", ""]
            lines += [x for i in range(3) for x in (_para(v * 100 + c * 10 + i), "")]
    p = tmp_path / "风云.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    doc = parse_txt(p, CFG)
    assert doc.stats["txt"]["paragraph_mode"] == "blank"
    assert _titles(doc) == [
        ("第一卷 风起", [None, "第1章 风起1", "第2章 风起2"]),
        ("第二卷 云涌", [None, "第1章 云涌1", "第2章 云涌2"]),
    ]


def test_hard_wrapped_paragraphs_are_joined(tmp_path):
    para = "她推开门看见院子里落满了雪老人说你来了这是很长的一段话被硬换行切成了几行" * 2
    wrapped = [para[i : i + 12] for i in range(0, len(para), 12)]
    text = "第一章 雪\n\n" + "\n".join(wrapped) + "\n\n" + "\n".join(wrapped) + "\n"
    p = tmp_path / "w.txt"
    p.write_text(text, encoding="utf-8")
    doc = parse_txt(p, CFG)
    assert doc.stats["txt"]["paragraph_mode"] == "wrapped"
    sec = doc.parts[0].sections[0]
    assert sec.title == "第一章 雪" and [b.text for b in sec.blocks[1:]] == [para, para]


def test_no_headings_becomes_single_chapter(tmp_path):
    p = tmp_path / "短篇.txt"
    p.write_text("\n".join(_para(i) for i in range(5)), encoding="utf-8")
    doc = parse_txt(p, CFG)
    assert [s.title for s in doc.parts[0].sections] == ["短篇"]
    assert any(i.code == "txt_no_headings" for i in doc.issues)
    with pytest.raises(IngestError):
        (tmp_path / "空.txt").write_text("\n\n", encoding="utf-8")
        parse_txt(tmp_path / "空.txt", CFG)


def _export_txt(book_dir, dest):
    """把导入产物（text.txt：标题一行、段落一行、章间空一行）按"卷名 + 正文"导出成 txt。"""
    text = (book_dir / "text.txt").read_text(encoding="utf-8")
    dest.write_text(text, encoding="utf-8")
    return dest


def test_format_agnostic_roundtrip_epub_to_txt(tmp_path):
    """epub 导入后的正文导出成 txt 再导入：章节数、标题、正文逐字一致。"""
    settings = Settings(_env_file=None, paths=PathsConfig(data_dir=tmp_path / "a"))
    epub = three_part_book(tmp_path / "b.epub")
    assert run_ingest(epub, series_id="demo", settings=settings).ok
    src_dir = tmp_path / "a/books/demo-1"
    txt = _export_txt(src_dir, tmp_path / "星海纪事.txt")

    settings2 = Settings(_env_file=None, paths=PathsConfig(data_dir=tmp_path / "b"))
    res = run_ingest(txt, series_id="demo", settings=settings2)
    assert res.ok, res.report["errors"]
    dst_dir = tmp_path / "b/books/demo-1"
    assert (dst_dir / "text.txt").read_text(encoding="utf-8") == (src_dir / "text.txt").read_text(encoding="utf-8")
    import json

    def chapters(d):
        rows = [json.loads(x) for x in (d / "chapters.jsonl").read_text(encoding="utf-8").splitlines()]
        return [(r["title"], r["section_type"], r["char_start"], r["char_end"]) for r in rows]

    assert chapters(dst_dir) == chapters(src_dir)
