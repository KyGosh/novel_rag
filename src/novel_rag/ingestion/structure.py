"""业务结构：分册合并成卷、卷号、section_type、章节编号、POV、全系列 reading_order。与原始格式无关。"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from novel_rag.config import IngestConfig, SectionKeywords
from novel_rag.ingestion.book_config import BookConfig
from novel_rag.ingestion.ir import Document, Section

# 分册后缀：（上）（中）（下）、（上册）、上部 / 下部 等
_PART_SUFFIX = re.compile(r"\s*(?:[（(]\s*[上中下](?:册|部|卷)?\s*[)）]|[上中下][册部])\s*$")
_VOLUME_NO = re.compile(r"第\s*([零〇一二两三四五六七八九十百\d]+)\s*[卷部]")
_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def parse_cn_number(s: str) -> int | None:
    """解析"第X卷"中的数字：阿拉伯数字或不超过 999 的中文数字。"""
    if s.isdigit():
        return int(s)
    total, cur = 0, 0
    for ch in s:
        if ch in _CN_DIGITS:
            cur = _CN_DIGITS[ch]
        elif ch == "十":
            total += (cur or 1) * 10
            cur = 0
        elif ch == "百":
            total += (cur or 1) * 100
            cur = 0
        else:
            return None
    return total + cur or None


def strip_part_suffix(title: str) -> str:
    return _PART_SUFFIX.sub("", title).strip()


def _starts_with(title: str, words: tuple[str, ...]) -> bool:
    t = title.strip().lower()
    return any(t.startswith(w.lower()) for w in words)


@dataclass
class Chapter:
    section: Section
    title: str
    section_type: str
    is_prologue: bool = False
    chapter_idx: int = -1
    pov: str | None = None
    reading_order: int = -1


@dataclass
class OtherSection:
    """不写入 text.txt 的内容（标题页、献词、前后辅文等），只出现在质检报告中。"""

    title: str | None
    section_type: str
    chars: int
    images: int
    locator: str
    part_title: str | None


@dataclass
class Volume:
    number: int
    number_source: str  # title_page / order
    title: str
    title_source: str  # config / toc / metadata
    raw_titles: list[str]
    part_indices: list[int]
    book_id: str = ""
    chapters: list[Chapter] = field(default_factory=list)
    others: list[OtherSection] = field(default_factory=list)


def _section_chars(s: Section, skip_first_heading: bool) -> tuple[int, int]:
    blocks = s.blocks[1:] if skip_first_heading and s.blocks and s.blocks[0].kind == "heading" else s.blocks
    return sum(len(b.text) for b in blocks if b.kind != "image"), sum(1 for b in blocks if b.kind == "image")


def _classify(title: str, s: Section, kw: SectionKeywords, blank_max: int) -> tuple[str, bool]:
    """返回（section_type, 是否序幕）。"""
    if _starts_with(title, kw.appendix):
        return "appendix", False
    if _starts_with(title, kw.front_matter):
        return "front_matter", False
    if _starts_with(title, kw.back_matter):
        return "back_matter", False
    if _starts_with(title, kw.dedication):
        return "dedication", False
    chars, _ = _section_chars(s, skip_first_heading=True)
    if chars < blank_max:  # 只有标题或一张图
        return "blank", False
    return "narrative", _starts_with(title, kw.prologue)


def _preamble_type(s: Section, part_title: str | None, kw: SectionKeywords, blank_max: int) -> str:
    text = " ".join(b.text for b in s.blocks if b.kind != "image")
    if part_title and any(
        b.kind == "heading" and strip_part_suffix(b.text) == strip_part_suffix(part_title) for b in s.blocks
    ):
        return "volume_title"
    if any(w in text.lower() for w in (x.lower() for x in kw.dedication)):
        return "dedication"
    chars, _ = _section_chars(s, skip_first_heading=False)
    return "blank" if chars < blank_max else "front_matter"


def build_volumes(doc: Document, series_id: str, cfg: IngestConfig, book: BookConfig) -> list[Volume]:
    kw = cfg.keywords

    # 1. 相邻且去掉分册后缀后同名的分册合并成一卷；目录之前的内容归入第一卷的辅文
    groups: list[tuple[str, list[int]]] = []
    leading: list[int] = []
    for i, part in enumerate(doc.parts):
        if part.title is None:
            leading.append(i)
            continue
        base = strip_part_suffix(part.title)
        if groups and groups[-1][0] == base and _PART_SUFFIX.search(part.title):
            groups[-1][1].append(i)
        else:
            groups.append((base, [i]))
    # 没有任何章节的分组（例如独立的"版权页"条目）不成卷，并入相邻卷的辅文
    real = [(b, ix) for b, ix in groups if any(s.title for i in ix for s in doc.parts[i].sections)]
    if not real:
        doc.error("no_chapters", "没有识别出任何章节")
        return []

    # 2. 卷号：优先取分册标题页上的"第X卷"，否则按出现顺序
    numbers: list[int | None] = []
    for _base, ix in real:
        first = doc.parts[ix[0]]
        texts = [first.title or ""] + [b.text for s in first.sections if s.title is None for b in s.blocks]
        found = next((m for t in texts if (m := _VOLUME_NO.search(t))), None)
        numbers.append(parse_cn_number(found.group(1)) if found else None)
    from_page = all(n is not None for n in numbers) and len(set(numbers)) == len(numbers)
    consistent = all(n is None or n == k + 1 for k, n in enumerate(numbers))  # 缺了几个，但与出现顺序一致
    if not from_page and not consistent:
        doc.warn("volume_number_fallback", f"标题页上的卷号与出现顺序冲突或有重复（{numbers}），改为按出现顺序编号")

    volumes: list[Volume] = []
    for k, (base, ix) in enumerate(real):
        number = numbers[k] if from_page else k + 1
        assert number is not None
        number_source = "title_page" if numbers[k] is not None and (from_page or consistent) else "order"
        title, source = base, "toc"
        if number in book.volume_titles:
            title, source = book.volume_titles[number], "config"
        volumes.append(
            Volume(
                number=number,
                number_source=number_source,
                title=title,
                title_source=source,
                raw_titles=[doc.parts[i].title or "" for i in ix],
                part_indices=list(ix),
                book_id=f"{series_id}-{number}",
            )
        )
    real_parts = {i for v in volumes for i in v.part_indices}
    for i, part in enumerate(doc.parts):
        if i in real_parts:
            continue
        # 目录之前的内容 / 无章节的分组：并入位置上最近的后一卷，没有则并入最后一卷
        target = next((v for v in volumes if v.part_indices[0] > i), volumes[-1])
        for s in part.sections:
            chars, images = _section_chars(s, skip_first_heading=False)
            st = "front_matter" if i in leading or target.part_indices[0] > i else "back_matter"
            target.others.append(OtherSection(s.title, st, chars, images, s.locator, part.title))

    # 3. 每卷的 section_type 与章节编号
    text_types = set(cfg.text_section_types)
    for v in volumes:
        chapters: list[Chapter] = []
        for i in v.part_indices:
            part = doc.parts[i]
            for s in part.sections:
                if s.title is None:
                    chars, images = _section_chars(s, skip_first_heading=False)
                    v.others.append(
                        OtherSection(
                            None,
                            _preamble_type(s, part.title, kw, cfg.blank_section_max_chars),
                            chars,
                            images,
                            s.locator,
                            part.title,
                        )
                    )
                    continue
                st, prologue = _classify(s.title, s, kw, cfg.blank_section_max_chars)
                override = book.section_type_for(v.number, s.title)
                if override:
                    st, prologue = override, prologue and override == "narrative"
                if st in text_types:
                    chapters.append(Chapter(s, s.title, st, prologue))
                else:
                    chars, images = _section_chars(s, skip_first_heading=True)
                    v.others.append(OtherSection(s.title, st, chars, images, s.locator, part.title))
        # 附录（以及其他非正文类型）排在正文之后，本身含剧透
        chapters.sort(key=lambda c: c.section_type != "narrative")
        start = 0 if chapters and chapters[0].section_type == "narrative" and chapters[0].is_prologue else 1
        for j, c in enumerate(chapters):
            c.chapter_idx = start + j
        v.chapters = chapters

    # 4. POV：全系列中反复出现的短标题视为人名；书籍配置可映射或取消
    counts = Counter(c.title for v in volumes for c in v.chapters if c.section_type == "narrative")
    keyword_sets = (kw.prologue, kw.epilogue, kw.appendix, kw.front_matter, kw.back_matter, kw.dedication)
    for v in volumes:
        for c in v.chapters:
            if c.title in book.pov:
                c.pov = book.pov[c.title]
            elif (
                c.section_type == "narrative"
                and len(c.title) <= cfg.pov_title_max_len
                and counts[c.title] >= cfg.pov_min_occurrences
                and not any(_starts_with(c.title, ws) for ws in keyword_sets)
            ):
                c.pov = c.title

    # 5. 全系列 reading_order：按卷号、卷内章节号连续编号
    volumes.sort(key=lambda v: v.number)
    order = 0
    for v in volumes:
        for c in v.chapters:
            c.reading_order = order
            order += 1
    return volumes
