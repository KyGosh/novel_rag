"""测试用的迷你 epub 生成器。内容全部自拟，不含版权原文。"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

XHTML = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><head><title>t</title></head>
<body>{body}</body></html>"""


@dataclass
class Toc:
    title: str
    src: str  # 相对 OEBPS 的路径，可带 #片段
    children: list[Toc] = field(default_factory=list)


def _ncx(entries: list[Toc]) -> str:
    counter = iter(range(1, 10_000))

    def points(es: list[Toc]) -> str:
        return "".join(
            f'<navPoint id="np{(n := next(counter))}" playOrder="{n}"><navLabel><text>{e.title}</text></navLabel>'
            f'<content src="{e.src}"/>{points(e.children)}</navPoint>'
            for e in es
        )

    return (
        '<?xml version="1.0" encoding="utf-8"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
        f"<head/><docTitle><text>t</text></docTitle><navMap>{points(entries)}</navMap></ncx>"
    )


def _nav(entries: list[Toc]) -> str:
    def ol(es: list[Toc]) -> str:
        items = "".join(f'<li><a href="{e.src}">{e.title}</a>{ol(e.children) if e.children else ""}</li>' for e in es)
        return f"<ol>{items}</ol>"

    return XHTML.format(body=f'<nav epub:type="toc">{ol(entries)}</nav>')


def build_epub(
    path: Path,
    files: dict[str, str],
    toc: list[Toc],
    *,
    toc_kind: str = "ncx",
    title: str = "测试书",
    extra: dict[str, bytes] | None = None,
    raw_files: dict[str, str] | None = None,
) -> Path:
    """files：{"Text/a.xhtml": "<p>正文</p>"}，按插入顺序成为 spine；raw_files 为不套模板的完整 XHTML。"""
    raw_files = raw_files or {}
    all_names = list(files) + list(raw_files)
    items = "".join(
        f'<item id="f{i}" href="{name}" media-type="application/xhtml+xml"/>' for i, name in enumerate(all_names)
    )
    spine = "".join(f'<itemref idref="f{i}"/>' for i in range(len(all_names)))
    if toc_kind == "ncx":
        items += '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        spine_attr = ' toc="ncx"'
    else:
        items += '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
        spine_attr = ""
    opf = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="uid">'
        f'<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title>'
        "<dc:language>zh</dc:language></metadata>"
        f"<manifest>{items}</manifest><spine{spine_attr}>{spine}</spine></package>"
    )
    container = (
        '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
        "</rootfiles></container>"
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", container)
        z.writestr("OEBPS/content.opf", opf)
        if toc_kind == "ncx":
            z.writestr("OEBPS/toc.ncx", _ncx(toc))
        else:
            z.writestr("OEBPS/nav.xhtml", _nav(toc))
        for name, body in files.items():
            z.writestr(f"OEBPS/{name}", XHTML.format(body=body))
        for name, content in raw_files.items():
            z.writestr(f"OEBPS/{name}", content)
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return path


def three_part_book(path: Path, *, toc_kind: str = "ncx", volume_label: str = "【第一卷】") -> Path:
    """一卷分上下两册 + 第二卷一册的小书：带标题页、序幕（标题无锚点）、注释、插图和附录。"""
    files = {
        "Text/front.xhtml": "<p>作者简介：一位虚构的作者。</p>",
        "Text/t1.xhtml": f"<div>{volume_label}</div><h1>星海纪事（上）</h1>",
        "Text/c1.xhtml": (
            "<h2>序幕</h2>"
            '<p>守望者<a id="m1" href="c1.xhtml#n1"><sup>[1]</sup></a>站在高塔上，夜色渐浓。</p>'
            "<p>阿岚<span>•</span>星川望向远方。</p>"
            '<div><img src="../Images/pic.jpg"/></div>'
            '<h2 id="toc_1">阿岚</h2><p>阿岚走进大厅&nbsp;。<a id="x"/>他想起了往事。</p>'
            '<p><a id="n1" href="c1.xhtml#m1">[1]</a>守望者：驻守边境的卫士。</p>'
        ),
        "Text/t2.xhtml": "<h1>星海纪事（下）</h1>",
        "Text/c2.xhtml": (
            "<h2>阿岚</h2><p>第二天清晨，阿岚出发了。他问：“我们去哪?”</p>"
            "<h2>附录</h2><h3>附录一 地名表</h3><p>高塔 北境要塞</p>"
        ),
        "Text/t3.xhtml": "<div>【第二卷】</div><h1>远航之歌</h1>",
        "Text/c3.xhtml": "<h2>阿岚</h2><p>船队离开港口，驶向未知的海域。</p><h2>琪娜</h2><p>琪娜在甲板上数星星。</p>",
    }
    toc = [
        Toc("星海纪事（上）", "Text/t1.xhtml", [Toc("序幕", "Text/c1.xhtml"), Toc("阿岚", "Text/c1.xhtml#toc_1")]),
        Toc("星海纪事（下）", "Text/t2.xhtml", [Toc("阿岚", "Text/c2.xhtml"), Toc("附录", "Text/c2.xhtml")]),
        Toc("远航之歌", "Text/t3.xhtml", [Toc("阿岚", "Text/c3.xhtml"), Toc("琪娜", "Text/c3.xhtml")]),
    ]
    return build_epub(path, files, toc, toc_kind=toc_kind, extra={"OEBPS/Images/pic.jpg": _tiny_jpeg()})


def _tiny_jpeg() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="JPEG")
    return buf.getvalue()
