"""epub 解析器（EPUB 2 的 NCX 目录 / EPUB 3 的 nav 目录）。

流程：安全打开容器 → OPF（spine、manifest、元数据）→ 目录 → 每个 XHTML 转成块序列 →
全书范围配对注释（标记删除，内容挂到标记位置）→ 按目录顶层条目划分分册 → 分册内按章节标题切分章节，
并与目录第 1 层逐项交叉校验。解析器只反映文件结构，不做卷合并、section_type 等业务判定。
"""

from __future__ import annotations

import hashlib
import html.entities
import posixpath
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from lxml import etree
from lxml import html as lxml_html

from novel_rag.config import IngestConfig
from novel_rag.ingestion.ir import NOTE_MARK, Block, Document, IngestError, NoteRef, Part, Section
from novel_rag.ingestion.safe_zip import XML_PARSER, SafeZip

PARSER_VERSION = "epub-1"

_HEADINGS = {f"h{i}": i for i in range(1, 7)}
_BLOCK_TAGS = {
    "body",
    "div",
    "p",
    "section",
    "article",
    "aside",
    "nav",
    "header",
    "footer",
    "main",
    "blockquote",
    "ul",
    "ol",
    "li",
    "dl",
    "dt",
    "dd",
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "td",
    "th",
    "caption",
    "figure",
    "figcaption",
    "pre",
    "hr",
    "center",
    "address",
    *_HEADINGS,
}
_SKIP_TAGS = {"head", "script", "style", "noscript", "template", "title", "meta", "link"}
_NOTE_TYPES = {"footnote", "endnote", "rearnote", "note"}
_XHTML_TYPES = {"application/xhtml+xml", "text/html"}
_NCX_TYPE = "application/x-dtbncx+xml"

_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
_NUM = rf"(\d{{1,3}}|[{_CIRCLED}])"
_LABEL = re.compile(rf"^\s*[\[［【(（]?\s*{_NUM}\s*[\]］】)）]?\s*$")  # 整个链接文本就是编号
_LABEL_PREFIX = re.compile(rf"^\s*[\[［【(（]\s*{_NUM}\s*[\]］】)）]|^\s*([{_CIRCLED}])")  # 段首编号（无链接兜底）
_BR = " "  # <br> 在段落文本中的临时表示，flush 时据此拆分段落
_ENTITY = re.compile(rb"&([A-Za-z][A-Za-z0-9]{1,31});")
_XML_ENTITIES = {b"amp", b"lt", b"gt", b"quot", b"apos"}
_ENCODING_DECL = re.compile(rb"""^\s*<\?xml[^>]*encoding=["']([A-Za-z0-9._-]+)["']""")
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)
_WS = re.compile(r"\s+")


# ---------------------------------------------------------------- 通用小工具


def _tag(el: object) -> str | None:
    tag = getattr(el, "tag", None)
    if not isinstance(tag, str):  # 注释、处理指令
        return None
    return etree.QName(tag).localname.lower()


def _el_id(el: etree._Element) -> str | None:
    return el.get("id") or el.get("{http://www.w3.org/XML/1998/namespace}id")


def _epub_types(el: etree._Element) -> set[str]:
    for k, v in el.attrib.items():
        if k == "epub:type" or (k.startswith("{") and k.endswith("}type") and "idpf.org" in k):
            return set(v.split())
    return set()


def _href(el: etree._Element) -> str | None:
    for k, v in el.attrib.items():
        if k == "href" or k.endswith("}href"):
            return v
    return None


def _hidden(el: etree._Element) -> bool:
    return el.get("hidden") is not None or bool(_HIDDEN_STYLE.search(el.get("style") or ""))


def _label_key(label: str) -> str:
    m = _LABEL.match(label) or _LABEL_PREFIX.match(label)
    if not m:
        return label.strip()
    num = next(g for g in m.groups() if g)
    return str(_CIRCLED.index(num) + 1) if num in _CIRCLED else str(int(num))


def _canon_title(s: str) -> str:
    """目录标签与正文标题比对用：去空白，去掉各种间隔号及其被替换成的 '?'。"""
    return re.sub(r"[\s·•‧・･∙⋅?？]", "", s).lower()


def resolve_href(base_path: str, href: str) -> tuple[str, str | None] | None:
    """把相对链接解析成（容器内路径, 片段）；外部链接返回 None。"""
    href = href.strip()
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", href):
        return None
    path, _, frag = href.partition("#")
    target = base_path if not path else posixpath.normpath(posixpath.join(posixpath.dirname(base_path), unquote(path)))
    return target, (unquote(frag) or None)


def _replace_entity(m: re.Match[bytes]) -> bytes:
    name = m.group(1)
    if name in _XML_ENTITIES:
        return m.group(0)
    chars = html.entities.html5.get(name.decode("ascii") + ";")
    return "".join(f"&#{ord(c)};" for c in chars).encode("ascii") if chars else m.group(0)


def load_xhtml(data: bytes) -> tuple[etree._Element, bool]:
    """先按 XML 严格解析（HTML 命名实体先转成数字引用，因为不加载 DTD）；失败再退回 HTML 解析器。

    返回（根元素, 是否用了 HTML 兜底）。XHTML 中的 `<a id="x"/>` 这类自闭合标签只有 XML 解析才能正确处理。
    """
    try:
        return etree.fromstring(_ENTITY.sub(_replace_entity, data), XML_PARSER), False
    except etree.XMLSyntaxError:
        m = _ENCODING_DECL.match(data)
        parser = lxml_html.HTMLParser(
            encoding=m.group(1).decode("ascii") if m else "utf-8", no_network=True, remove_comments=True
        )
        return lxml_html.document_fromstring(data, parser=parser), True


# ---------------------------------------------------------------- XHTML → 块序列


@dataclass
class _Anchor:
    """可能是注释标记、也可能是注释段落开头编号的链接或上标。配对后确定角色。"""

    order: int  # 全书出现顺序
    path: str
    label: str
    key: str
    own_ids: tuple[str, ...]
    target: tuple[str, str | None] | None
    at_start: bool  # 位于段首：注释段落的编号通常如此
    noteref: bool
    block: int = -1
    role: str = "pending"  # marker / note / unpaired
    note: NoteRef | None = None


@dataclass
class _Buf:
    pieces: list[str] = field(default_factory=list)
    anchors: list[_Anchor] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)

    def current_line(self) -> str:
        return "".join(self.pieces).rsplit(_BR, 1)[-1]


class _XhtmlFile:
    def __init__(self, path: str, counter: Iterator[int], doc: Document):
        self.path = path
        self.blocks: list[Block | None] = []
        self.ids: dict[str, tuple[int, int]] = {}  # 元素 id → 它产生的块区间 [start, end)
        self.note_ids: set[str] = set()  # epub:type=footnote 等注释容器的 id
        self.anchors: list[_Anchor] = []
        self.block_anchors: dict[int, list[_Anchor]] = {}
        self._counter = counter
        self._doc = doc

    # -- 遍历

    def parse(self, root: etree._Element) -> None:
        body = next((e for e in root.iter() if _tag(e) == "body"), root)
        self._walk(body)

    def _walk(self, el: etree._Element) -> None:
        tag = _tag(el)
        if tag is None or tag in _SKIP_TAGS or _hidden(el):
            return
        start = len(self.blocks)
        if tag in _HEADINGS:
            self._leaf(el, "heading", _HEADINGS[tag])
        elif tag in ("img", "image"):
            self._emit_image(el)
        elif tag == "svg":
            for sub in el.iter():
                if _tag(sub) == "image":
                    self._emit_image(sub)
        elif self._has_block_desc(el):
            self._container(el)
        else:
            self._leaf(el, "paragraph", 0)
        el_id = _el_id(el)
        if el_id:
            self.ids.setdefault(el_id, (start, len(self.blocks)))
            if _epub_types(el) & _NOTE_TYPES:
                self.note_ids.add(el_id)

    @staticmethod
    def _has_block_desc(el: etree._Element) -> bool:
        # 不能按 id(el) 缓存：lxml 的元素代理对象被回收后 id 会被复用
        return any(_tag(d) in _BLOCK_TAGS or _tag(d) == "svg" for d in el.iterdescendants())

    def _container(self, el: etree._Element) -> None:
        buf = _Buf()
        self._text(buf, el.text)
        for child in el:
            ctag = _tag(child)
            if ctag is not None and ctag not in _SKIP_TAGS and not _hidden(child):
                if ctag in _BLOCK_TAGS or ctag == "svg" or self._has_block_desc(child):
                    self._flush(buf, "paragraph", 0, None)
                    buf = _Buf()
                    self._walk(child)
                else:
                    self._inline(child, buf, heading=False)
            self._text(buf, child.tail)
        self._flush(buf, "paragraph", 0, None)

    def _leaf(self, el: etree._Element, kind: str, level: int) -> None:
        buf = _Buf()
        self._text(buf, el.text)
        for child in el:
            self._inline(child, buf, heading=kind == "heading")
            self._text(buf, child.tail)
        self._flush(buf, kind, level, _el_id(el))

    def _inline(self, el: etree._Element, buf: _Buf, *, heading: bool) -> None:
        tag = _tag(el)
        if tag is None or tag in _SKIP_TAGS or _hidden(el):
            return
        el_id = _el_id(el)
        if el_id:
            buf.ids.append(el_id)
        if tag == "br":
            buf.pieces.append(" " if heading else _BR)
            return
        if tag in ("img", "image"):
            src = self._img_src(el)
            if src:
                buf.images.append(src)
            return
        if tag == "svg":
            buf.images.extend(s for sub in el.iter() if _tag(sub) == "image" and (s := self._img_src(sub)))
            return
        if tag in ("a", "sup") and (anchor := self._anchor(el, tag, buf)):
            buf.pieces.append(NOTE_MARK)
            buf.anchors.append(anchor)
            return
        self._text(buf, el.text)
        for child in el:
            self._inline(child, buf, heading=heading)
            self._text(buf, child.tail)

    def _text(self, buf: _Buf, text: str | None) -> None:
        if not text:
            return
        if NOTE_MARK in text or _BR in text:  # 原文中恰好出现占位符字符：去掉并计数，避免与标记混淆
            self._doc.stats["stray_placeholder_chars"] = int(self._doc.stats.get("stray_placeholder_chars", 0)) + (
                text.count(NOTE_MARK) + text.count(_BR)
            )
            text = text.replace(NOTE_MARK, "").replace(_BR, " ")
        buf.pieces.append(text)

    def _anchor(self, el: etree._Element, tag: str, buf: _Buf) -> _Anchor | None:
        label = "".join(el.itertext()).strip()
        types = _epub_types(el)
        noteref = "noteref" in types
        if not (_LABEL.match(label) or (noteref and label)):
            return None
        target = None
        if tag == "a":
            href = _href(el)
            if href:
                target = resolve_href(self.path, href)
                if target is None or (target[1] is None and not noteref):
                    return None  # 外部链接或指向整个文件的普通链接（目录页等），不是注释
        own = tuple(i for e in el.iter() if isinstance(e.tag, str) and (i := _el_id(e)))
        anchor = _Anchor(
            order=next(self._counter),
            path=self.path,
            label=label,
            key=_label_key(label),
            own_ids=own,
            target=target,
            at_start=not buf.current_line().strip(),
            noteref=noteref,
        )
        self.anchors.append(anchor)
        return anchor

    def _img_src(self, el: etree._Element) -> str | None:
        src = el.get("src") or _href(el)
        if not src:
            return None
        resolved = resolve_href(self.path, src)
        return resolved[0] if resolved else None

    def _emit_image(self, el: etree._Element) -> None:
        src = self._img_src(el)
        if src:
            idx = len(self.blocks)
            self.blocks.append(Block("image", locator=f"{self.path}@{idx}", image_src=src))

    def _flush(self, buf: _Buf, kind: str, level: int, el_id: str | None) -> None:
        text = "".join(buf.pieces)
        lines = text.split(_BR) if kind == "paragraph" else [text.replace(_BR, " ")]
        first = len(self.blocks)
        ai = 0
        for line in lines:
            n = line.count(NOTE_MARK)
            if not line.strip() and n == 0:
                if kind == "paragraph":
                    self._doc.stats["empty_paragraphs_dropped"] = (
                        int(self._doc.stats.get("empty_paragraphs_dropped", 0)) + 1
                    )
                continue
            idx = len(self.blocks)
            locator = f"{self.path}#{el_id}" if el_id and idx == first else f"{self.path}@{idx}"
            self.blocks.append(Block(kind, line, level, locator))  # type: ignore[arg-type]
            if n:
                group = buf.anchors[ai : ai + n]
                for a in group:
                    a.block = idx
                self.block_anchors[idx] = group
                ai += n
        for i in buf.ids:
            self.ids.setdefault(i, (first, max(first + 1, len(self.blocks)) if len(self.blocks) > first else first))
        for src in buf.images:
            idx = len(self.blocks)
            self.blocks.append(Block("image", locator=f"{self.path}@{idx}", image_src=src))


# ---------------------------------------------------------------- 注释配对


def _strip_mark(block: Block, anchors: list[_Anchor], drop: _Anchor) -> str:
    """注释段落的文本：删掉作为编号的那个占位符，其余占位符还原成原标签。"""
    out, k = [], 0
    for ch in block.text:
        if ch == NOTE_MARK:
            a = anchors[k]
            k += 1
            if a is not drop:
                out.append(a.label)
        else:
            out.append(ch)
    return "".join(out).strip()


def _pair_notes(files: dict[str, _XhtmlFile], doc: Document) -> None:
    by_id: dict[tuple[str, str], _Anchor] = {}
    all_anchors = sorted((a for f in files.values() for a in f.anchors), key=lambda a: a.order)
    for a in all_anchors:
        for i in a.own_ids:
            by_id[(a.path, i)] = a
    consumed: set[tuple[str, int]] = set()

    def note_text_from_blocks(f: _XhtmlFile, start: int, end: int) -> str:
        parts = []
        for i in range(start, end):
            b = f.blocks[i]
            if b is not None and b.kind != "image":
                anchors = f.block_anchors.get(i, [])
                parts.append(_strip_mark(b, anchors, drop=None))  # type: ignore[arg-type]
                for a in anchors:
                    a.role = "note"
                consumed.add((f.path, i))
        return " ".join(p for p in parts if p)

    for m in all_anchors:
        if m.role != "pending" or m.target is None:
            continue
        tpath, frag = m.target
        f = files.get(tpath)
        if f is None or frag is None:
            continue
        n = by_id.get((tpath, frag))
        if n is not None and n is not m and n.at_start and n.role == "pending":
            if m.at_start and n.order < m.order:
                continue  # 互指的两个段首编号：靠后的是注释，等处理 n 时再配对
            backlink_ok = n.target is not None and n.target[0] == m.path and n.target[1] in m.own_ids
            if backlink_ok or n.key == m.key:
                nb = files[n.path].blocks[n.block]
                assert nb is not None
                text = _strip_mark(nb, files[n.path].block_anchors[n.block], drop=n)
                m.role, n.role = "marker", "note"
                m.note = NoteRef(m.label, text, f"{m.path}@{m.block}", nb.locator)
                consumed.add((n.path, n.block))
                continue
        if frag in f.note_ids:  # EPUB 3：<aside epub:type="footnote" id=...>
            start, end = f.ids[frag]
            text = note_text_from_blocks(f, start, end)
            if text:
                m.role = "marker"
                m.note = NoteRef(m.label, text, f"{m.path}@{m.block}", f"{tpath}#{frag}")

    # 兜底：没有链接的上标标记，与同一文件中其后、以相同编号开头的段落按顺序配对
    for f in files.values():
        markers = [a for a in f.anchors if a.role == "pending" and a.target is None and not a.at_start]
        if not markers:
            continue
        used: set[int] = set()
        for m in markers:
            for i in range(m.block + 1, len(f.blocks)):
                b = f.blocks[i]
                if b is None or b.kind != "paragraph" or i in used or (f.path, i) in consumed:
                    continue
                starts = f.block_anchors.get(i, [])
                lead = starts[0] if starts and starts[0].at_start and starts[0].role == "pending" else None
                if lead is not None and lead.key == m.key:
                    text = _strip_mark(b, starts, drop=lead)
                    lead.role = "note"
                elif lead is None and (pm := _LABEL_PREFIX.match(b.text)) and _label_key(pm.group(0)) == m.key:
                    text = b.text[pm.end() :].strip()
                else:
                    continue
                m.role = "marker"
                m.note = NoteRef(m.label, text, f"{m.path}@{m.block}", b.locator)
                used.add(i)
                consumed.add((f.path, i))
                break

    # 统计、上报未配对项，并把配对结果写回块
    by_file: dict[str, dict[str, int]] = {}
    for a in all_anchors:
        s = by_file.setdefault(
            a.path, {"markers": 0, "notes": 0, "paired": 0, "unpaired_markers": 0, "orphan_notes": 0}
        )
        if a.role == "marker":
            s["markers"] += 1
            s["paired"] += 1
        elif a.role == "note":
            s["notes"] += 1
        elif a.at_start and a.target is not None:
            s["orphan_notes"] += 1
            a.role = "unpaired"
            doc.warn("orphan_note", f"注释段落 {a.label} 没有找到对应的正文标记，原文保留", f"{a.path}@{a.block}")
        else:
            s["markers"] += 1
            s["unpaired_markers"] += 1
            a.role = "unpaired"
            doc.warn("unpaired_marker", f"注释标记 {a.label} 没有找到对应的注释内容，原文保留", f"{a.path}@{a.block}")
    doc.stats["notes_by_file"] = by_file

    for f in files.values():
        for idx, anchors in f.block_anchors.items():
            b = f.blocks[idx]
            if b is None or (f.path, idx) in consumed:
                continue
            out, k = [], 0
            for ch in b.text:
                if ch == NOTE_MARK:
                    a = anchors[k]
                    k += 1
                    if a.role == "marker" and a.note is not None:
                        out.append(NOTE_MARK)
                        b.notes.append(a.note)
                    else:
                        out.append(a.label)
                else:
                    out.append(ch)
            b.text = "".join(out)
        for path, idx in consumed:
            if path == f.path:
                f.blocks[idx] = None


# ---------------------------------------------------------------- 目录


@dataclass
class TocEntry:
    title: str
    path: str
    frag: str | None
    children: list[TocEntry] = field(default_factory=list)


def _text_of(el: etree._Element | None) -> str:
    return _WS.sub(" ", "".join(el.itertext())).strip() if el is not None else ""


def _parse_ncx(root: etree._Element, ncx_path: str) -> list[TocEntry]:
    def walk(parent: etree._Element) -> list[TocEntry]:
        out = []
        for np in parent:
            if _tag(np) != "navpoint":
                continue
            label = next((e for e in np if _tag(e) == "navlabel"), None)
            content = next((e for e in np if _tag(e) == "content"), None)
            target = resolve_href(ncx_path, content.get("src", "")) if content is not None else None
            if target is None:
                out.extend(walk(np))
                continue
            out.append(TocEntry(_text_of(label), target[0], target[1], walk(np)))
        return out

    nav_map = next((e for e in root.iter() if _tag(e) == "navmap"), None)
    return walk(nav_map) if nav_map is not None else []


def _parse_nav(root: etree._Element, nav_path: str) -> list[TocEntry]:
    navs = [e for e in root.iter() if _tag(e) == "nav"]
    nav = next((n for n in navs if "toc" in _epub_types(n)), navs[0] if navs else None)
    if nav is None:
        return []

    def walk(ol: etree._Element) -> list[TocEntry]:
        out = []
        for li in ol:
            if _tag(li) != "li":
                continue
            link = next((e for e in li if _tag(e) in ("a", "span")), None)
            sub = next((e for e in li if _tag(e) == "ol"), None)
            children = walk(sub) if sub is not None else []
            href = _href(link) if link is not None else None
            target = resolve_href(nav_path, href) if href else None
            if target is None:
                out.extend(children)
                continue
            out.append(TocEntry(_text_of(link), target[0], target[1], children))
        return out

    ol = next((e for e in nav.iter() if _tag(e) == "ol"), None)
    return walk(ol) if ol is not None else []


# ---------------------------------------------------------------- 主流程


def parse_epub(path: Path, cfg: IngestConfig) -> Document:
    data = path.read_bytes()
    doc = Document(
        source_name=path.name,
        source_sha256=hashlib.sha256(data).hexdigest(),
        format="epub",
        parser_version=PARSER_VERSION,
    )
    with SafeZip(
        path,
        max_files=cfg.max_archive_files,
        max_total_bytes=cfg.max_uncompressed_bytes,
        max_member_bytes=cfg.max_member_bytes,
    ) as z:
        doc.stats["container"] = {"files": z.file_count, "uncompressed_bytes": z.total_bytes}
        _parse(z, doc)
    return doc


def _find_opf(z: SafeZip) -> str:
    if z.exists("META-INF/container.xml"):
        root = z.read_xml("META-INF/container.xml")
        rf = next((e for e in root.iter() if _tag(e) == "rootfile"), None)
        if rf is not None and rf.get("full-path") and z.exists(rf.get("full-path", "")):
            return posixpath.normpath(rf.get("full-path", ""))
    opfs = sorted(n for n in z.names() if n.lower().endswith(".opf"))
    if not opfs:
        raise IngestError("找不到 OPF 文件，不是有效的 epub")
    return opfs[0]


def _parse(z: SafeZip, doc: Document) -> None:
    opf_path = _find_opf(z)
    opf = z.read_xml(opf_path)
    doc.stats["container"]["epub_version"] = opf.get("version", "")  # type: ignore[index]
    for el in opf.iter():
        tag = _tag(el)
        if (
            tag in ("title", "creator", "language")
            and el.getparent() is not None
            and _tag(el.getparent()) == "metadata"
        ):
            doc.metadata.setdefault(tag, (el.text or "").strip())

    manifest: dict[str, tuple[str, str, str]] = {}
    for item in opf.iter():
        if _tag(item) == "item" and item.get("id") and item.get("href"):
            target = resolve_href(opf_path, item.get("href", ""))
            if target:
                manifest[item.get("id", "")] = (target[0], item.get("media-type", ""), item.get("properties", ""))
    spine_el = next((e for e in opf.iter() if _tag(e) == "spine"), None)
    if spine_el is None:
        raise IngestError("OPF 中没有 spine")
    spine: list[str] = []
    nonlinear = 0
    for ref in spine_el:
        if _tag(ref) != "itemref" or ref.get("idref") not in manifest:
            continue
        p, media, _ = manifest[ref.get("idref", "")]
        if media not in _XHTML_TYPES:
            doc.warn("spine_non_xhtml", f"spine 中的非 XHTML 文件已跳过：{p}（{media}）", p)
            continue
        if ref.get("linear") == "no":
            nonlinear += 1
        spine.append(p)

    # 目录：优先 nav（EPUB 3），否则 NCX
    toc: list[TocEntry] = []
    nav_item = next((v for v in manifest.values() if "nav" in v[2].split()), None)
    ncx_item = manifest.get(spine_el.get("toc", "")) or next((v for v in manifest.values() if v[1] == _NCX_TYPE), None)
    toc_type = "none"
    if nav_item and z.exists(nav_item[0]):
        toc, toc_type = _parse_nav(load_xhtml(z.read(nav_item[0]))[0], nav_item[0]), "nav"
    if not toc and ncx_item and z.exists(ncx_item[0]):
        toc, toc_type = _parse_ncx(z.read_xml(ncx_item[0]), ncx_item[0]), "ncx"

    # XHTML → 块
    counter = iter(range(10**9))
    files: dict[str, _XhtmlFile] = {}
    html_fallback = []
    for p in spine:
        if p in files:
            continue
        if not z.exists(p):
            doc.warn("spine_missing", f"spine 引用的文件不存在：{p}", p)
            continue
        root, fallback = load_xhtml(z.read(p))
        if fallback:
            html_fallback.append(p)
        f = _XhtmlFile(p, counter, doc)
        f.parse(root)
        files[p] = f
    if html_fallback:
        doc.warn("html_fallback", f"{len(html_fallback)} 个文件不是合法 XML，已改用 HTML 解析器：{html_fallback[:5]}")
    _pair_notes(files, doc)

    # 拼成全书块流，记录每个文件、每个 id 在流中的位置
    stream: list[Block] = []
    file_start: dict[str, int] = {}
    new_index: dict[str, list[int]] = {}
    for p, f in files.items():
        file_start[p] = len(stream)
        mapping = []
        for b in f.blocks:
            mapping.append(len(stream) - file_start[p])
            if b is not None:
                stream.append(b)
        new_index[p] = mapping

    def position(path: str, frag: str | None) -> int | None:
        if path not in file_start:
            return None
        f = files[path]
        if frag and frag in f.ids:
            old = f.ids[frag][0]
            m = new_index[path]
            kept = sum(1 for b in f.blocks if b is not None)
            return file_start[path] + (m[old] if old < len(m) else kept)
        return file_start[path]

    doc.parts = _build_parts(doc, stream, toc, position)
    for b in stream:
        if b.kind == "image" and b.image_src and b.image_src not in doc.resources:
            if z.exists(b.image_src):
                doc.resources[b.image_src] = z.read(b.image_src)
            else:
                doc.warn("image_missing", f"插图资源不存在：{b.image_src}", b.locator)

    # 每个 spine 文件的归类，供质检报告
    part_of_block = {id(b): i for i, part in enumerate(doc.parts) for s in part.sections for b in s.blocks}
    doc.stats["spine"] = [
        {
            "path": p,
            "blocks": sum(1 for b in f.blocks if b is not None),
            "chars": sum(len(b.text) for b in f.blocks if b is not None),
            "part": next((part_of_block.get(id(b)) for b in f.blocks if b is not None), None),
        }
        for p, f in files.items()
    ]
    doc.stats["toc"] = {"type": toc_type, "top_level": len(toc), "nonlinear_spine": nonlinear}


def _build_parts(doc: Document, stream: list[Block], toc: list[TocEntry], position) -> list[Part]:  # noqa: ANN001
    if not stream:
        doc.error("empty", "没有解析出任何正文内容")
        return []
    if not toc:
        doc.warn("no_toc", "没有目录，整份文件作为一个分册，按最常见的标题层级切分章节")
        return [_part_without_toc(doc, stream)]
    # 目录只有一层：顶层条目就是章节，整份文件是一个分册
    tops = (
        toc
        if any(t.children for t in toc)
        else [TocEntry(doc.metadata.get("title", ""), toc[0].path, toc[0].frag, toc)]
    )
    starts = []
    for t in tops:
        pos = position(t.path, t.frag)
        if pos is None:
            doc.warn("toc_target_missing", f"目录条目「{t.title}」指向的文件不在 spine 中，已忽略", t.path)
            continue
        # 分册通常从文件边界开始：锚点之前只有几个字（标题页上的系列名、卷号）且没有标题时，起点前移到文件开头
        file_pos = position(t.path, None)
        if file_pos is not None and file_pos < pos:
            lead = stream[file_pos:pos]
            if not any(b.kind == "heading" for b in lead) and sum(len(b.text.strip()) for b in lead) < 50:
                pos = file_pos
        starts.append((pos, t))
    if not starts:
        doc.error("toc_unusable", "目录条目都无法定位到正文")
        return [_part_without_toc(doc, stream)]
    if any(a[0] > b[0] for a, b in zip(starts, starts[1:], strict=False)):
        doc.error("toc_order", "目录顶层条目的顺序与正文顺序不一致")
    parts: list[Part] = []
    if starts[0][0] > 0:
        parts.append(Part(None, stream[0].locator, [Section(None, stream[0].locator, stream[: starts[0][0]])]))
    for i, (pos, top) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(stream)
        parts.append(_split_part(doc, top, stream[pos:end], position, pos))
    return parts


def _split_part(doc: Document, top: TocEntry, blocks: list[Block], position, offset: int) -> Part:  # noqa: ANN001
    locator = blocks[0].locator if blocks else top.path
    part = Part(top.title, locator)
    labels = [c.title for c in top.children]
    if not blocks:
        doc.warn("empty_part", f"分册「{top.title}」没有内容", top.path)
        return part
    if not labels:
        part.sections.append(Section(None, locator, blocks))
        return part

    heading_idx = _align_headings(doc, top, blocks, labels)
    if heading_idx is None:  # 无法对齐：退回到按目录条目位置切分
        heading_idx = []
        for c in top.children:
            pos = position(c.path, c.frag)
            if pos is not None and offset <= pos < offset + len(blocks):
                heading_idx.append(pos - offset)
        heading_idx = sorted(set(heading_idx))
    if heading_idx and heading_idx[0] > 0:
        part.sections.append(Section(None, locator, blocks[: heading_idx[0]]))
    for j, start in enumerate(heading_idx):
        end = heading_idx[j + 1] if j + 1 < len(heading_idx) else len(blocks)
        first = blocks[start]
        title = first.text.strip() if first.kind == "heading" else labels[min(j, len(labels) - 1)]
        part.sections.append(Section(title, first.locator, blocks[start:end]))
    return part


def _align_headings(doc: Document, top: TocEntry, blocks: list[Block], labels: list[str]) -> list[int] | None:
    """在分册内找出与目录第 1 层一一对应的那一级标题，返回这些标题块的下标；对不上返回 None 并记 error。"""
    n = len(labels)
    want = [_canon_title(t) for t in labels]
    by_level: dict[int, list[int]] = {}
    for i, b in enumerate(blocks):
        if b.kind == "heading":
            by_level.setdefault(b.level, []).append(i)
    counts = {lv: len(ix) for lv, ix in by_level.items()}
    mismatch_detail = None
    for level in sorted(by_level):
        idx = by_level[level]
        # 分册自己的标题与章节标题同级时，多出来的第一个标题不算章节
        if len(idx) == n + 1 and _canon_title(blocks[idx[0]].text) == _canon_title(top.title or ""):
            idx = idx[1:]
        if len(idx) != n:
            continue
        got = [_canon_title(blocks[i].text) for i in idx]
        bad = [(k, labels[k], blocks[idx[k]].text) for k in range(n) if got[k] != want[k]]
        partial = [x for x in bad if want[x[0]].startswith(got[x[0]]) or got[x[0]].startswith(want[x[0]])]
        if len(partial) == len(bad):
            for k, label, text in partial:
                doc.warn("toc_title_partial", f"目录「{label}」与正文标题「{text}」只部分一致", blocks[idx[k]].locator)
            return idx
        mismatch_detail = bad
    if mismatch_detail:
        doc.error(
            "toc_title_mismatch",
            f"分册「{top.title}」中目录与正文标题不一致："
            + "；".join(f"#{k}「{a}」≠「{b}」" for k, a, b in mismatch_detail[:10]),
            blocks[0].locator,
        )
    else:
        doc.error(
            "toc_count_mismatch",
            f"分册「{top.title}」目录第 1 层有 {n} 项，正文各级标题数为 {counts}，没有一级能对上",
            blocks[0].locator,
        )
    return None


def _part_without_toc(doc: Document, stream: list[Block]) -> Part:
    counts: dict[int, int] = {}
    for b in stream:
        if b.kind == "heading":
            counts[b.level] = counts.get(b.level, 0) + 1
    level = min((lv for lv, c in counts.items() if c >= 2), default=None)
    part = Part(doc.metadata.get("title") or None, stream[0].locator)
    cuts = [i for i, b in enumerate(stream) if b.kind == "heading" and b.level == level] if level else []
    if not cuts or cuts[0] > 0:
        part.sections.append(Section(None, stream[0].locator, stream[: cuts[0] if cuts else len(stream)]))
    for j, start in enumerate(cuts):
        end = cuts[j + 1] if j + 1 < len(cuts) else len(stream)
        part.sections.append(Section(stream[start].text.strip(), stream[start].locator, stream[start:end]))
    return part
