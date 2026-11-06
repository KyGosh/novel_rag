"""纯文本（.txt）解析器：编码探测 → 段落切分 → 卷 / 章标题识别 → 统一中间表示。

txt 没有结构标记，章节全靠标题行识别（不含任何书籍专属知识）：
1. 规则：卷标题"第X卷 / 第X部"；章标题"第X章 / 回 / 节"、"Chapter N"，以及序幕、尾声、附录等通用关键词开头的短行。
2. 启发式（针对以人物名为章节标题、没有编号的书）：短行、不含句读与引号，并且
   - 在"一行一段"的文本中前面有空行（正文段落之间一般不空行，章与章之间的空行就是分章信号；
     只看前面，因为章末最后一段后面也是空行）；
   - 在"空行分段 / 硬换行"的文本中空行没有区分度，改为要求同样的短行在全书中作为候选出现至少
     pov_min_occurrences 次（POV 人名反复做标题）。
   "一行一段"时只靠关键词（序幕、附录……）匹配的短行也要求前面有空行，避免把附录里的"附录一 ……"小标题切成新章。
没有被识别为标题的短行仍然是普通段落。标题之间正文过短（如目录页）的章节由 structure.py 判为 blank，不进入正文。

编码依次尝试：BOM（UTF-8 / UTF-16）、UTF-8、GB18030（兼容 GBK / GB2312）、Big5，全部失败才报错。
硬换行（每行固定宽度、段落间空一行）的文本，同一段内的行会先拼接起来；判定要求多数行等宽且不以句末标点结尾。
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

from novel_rag.config import IngestConfig
from novel_rag.ingestion.ir import Block, Document, IngestError, Part, Section

PARSER_VERSION = "txt-1"

_NUM = r"[零〇一二两三四五六七八九十百千\d]+"
VOLUME_RE = re.compile(rf"^第\s*{_NUM}\s*[卷部](?:\s|$|[:：·])")
CHAPTER_RE = re.compile(rf"^(?:第\s*{_NUM}\s*[章回节](?:\s|$|[:：·])|chapter\s+\d+\b)", re.I)
# 句读、引号、括号：正文句子几乎总带这些，标题几乎从不带（间隔号 · 与空格允许）
_PUNCT = re.compile(r"[，。！？；：、“”‘’「」『』《》（）()…—,.!?;:\"']")
_ENCODINGS = ("utf-8", "gb18030", "big5")


def decode(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8"), "utf-8-sig"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16"), "utf-16"
    for enc in _ENCODINGS:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise IngestError("无法识别文本编码（已尝试 UTF-8 / UTF-16 / GB18030 / Big5）")


def _is_short_title(line: str, max_len: int) -> bool:
    t = line.strip()
    return 0 < len(t) <= max_len and not _PUNCT.search(t) and not t.isdigit()


def _keyword_title(line: str, cfg: IngestConfig) -> bool:
    kw = cfg.keywords
    t = line.strip().lower()
    words = (*kw.prologue, *kw.epilogue, *kw.appendix, "番外", "后记", "尾声")
    return any(t.startswith(w.lower()) for w in words)


def _paragraphs(lines: list[str]) -> tuple[list[tuple[str, bool, bool]], str]:
    """返回 [(段落文本, 前面是否有空行, 后面是否有空行)] 与分段方式（line / blank / wrapped）。"""
    nonempty = [i for i, x in enumerate(lines) if x.strip()]
    if not nonempty:
        return [], "line"
    followed_by_text = sum(1 for i in nonempty if i + 1 < len(lines) and lines[i + 1].strip())
    no_end = sum(1 for i in nonempty if not re.search(r"[。！？…”」』.!?]\s*$", lines[i]))
    has_blank = len(nonempty) < len(lines) - 1
    inner = [len(lines[i].strip()) for i in nonempty if i + 1 < len(lines) and lines[i + 1].strip()]
    # 最常见的行宽（至少 10 字才算换行宽度，短标题等宽不算）
    width, same_width = Counter(inner).most_common(1)[0] if inner else (0, 0)
    # 有空行分段，过半的非空行紧跟非空行、不以句末标点结尾，且这些行多数等宽：硬换行
    if (
        has_blank
        and followed_by_text > 0.5 * len(nonempty)
        and no_end > 0.5 * len(nonempty)
        and width >= 10
        and same_width >= 0.5 * len(inner)
    ):
        mode = "wrapped"
    elif followed_by_text < 0.3 * len(nonempty):
        mode = "blank"  # 段落之间空一行
    else:
        mode = "line"  # 一行一段
    out: list[tuple[str, bool, bool]] = []
    if mode == "wrapped":
        buf: list[str] = []
        for x in [*lines, ""]:
            if x.strip():
                buf.append(x.strip())
            elif buf:
                out.append(("".join(buf), True, True))
                buf = []
        return out, mode
    for i in nonempty:
        before = i == 0 or not lines[i - 1].strip()
        after = i + 1 >= len(lines) or not lines[i + 1].strip()
        out.append((lines[i].strip(), before, after))
    return out, mode


def parse_txt(path: Path, cfg: IngestConfig) -> Document:
    data = path.read_bytes()
    if len(data) > cfg.max_uncompressed_bytes:
        raise IngestError(f"文件过大：{len(data)} 字节，超过 {cfg.max_uncompressed_bytes}")
    text, encoding = decode(data)
    doc = Document(
        source_name=path.name,
        source_sha256=hashlib.sha256(data).hexdigest(),
        format="txt",
        parser_version=PARSER_VERSION,
        metadata={"title": path.stem, "encoding": encoding},
    )
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    paras, mode = _paragraphs(lines)
    if not paras:
        raise IngestError("文件中没有任何文字")

    max_len = cfg.txt_title_max_len
    candidates = [p for p, _, _ in paras if _is_short_title(p, max_len)]
    repeated = {t for t in set(candidates) if candidates.count(t) >= cfg.pov_min_occurrences}

    def kind(p: str, before: bool, after: bool) -> str:
        isolated = before
        if VOLUME_RE.match(p) and len(p) <= max_len * 2:
            return "volume"
        short_enough = len(p) <= max_len * 2 and not _PUNCT.search(p[-1])
        if CHAPTER_RE.match(p) and short_enough:
            return "chapter"
        if _keyword_title(p, cfg) and short_enough and (mode != "line" or isolated):
            return "chapter"
        if _is_short_title(p, max_len):
            if mode == "line" and isolated:
                return "chapter"
            if mode != "line" and p in repeated:
                return "chapter"
        return "paragraph"

    parts: list[Part] = []
    part = Part(doc.metadata["title"], f"{path.name}#L0")
    section = Section(None, f"{path.name}#L0")
    counts = {"volume": 0, "chapter": 0, "paragraph": 0}
    for n, (p, before, after) in enumerate(paras):
        k = kind(p, before, after)
        counts[k] += 1
        loc = f"{path.name}#P{n}"
        if k == "volume":
            if section.blocks or section.title:
                part.sections.append(section)
            if part.sections:
                parts.append(part)
            part = Part(p, loc)
            section = Section(None, loc, [Block("heading", p, 1, loc)])
        elif k == "chapter":
            if section.blocks or section.title:
                part.sections.append(section)
            section = Section(p, loc, [Block("heading", p, 2, loc)])
        else:
            section.blocks.append(Block("paragraph", p, 0, loc))
    if section.blocks or section.title:
        part.sections.append(section)
    if part.sections:
        parts.append(part)
    doc.parts = parts
    doc.stats["txt"] = {"encoding": encoding, "paragraph_mode": mode, "lines": len(lines), **counts}
    if counts["chapter"] == 0:
        doc.warn("txt_no_headings", "没有识别出任何章节标题，全文将作为一章")
        # 没有标题时整篇作为一个章节，标题取文件名
        for pt in doc.parts:
            for s in pt.sections:
                if s.title is None:
                    s.title = pt.title or path.stem
    return doc
