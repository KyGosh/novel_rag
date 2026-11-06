"""文本规范化。规则按固定顺序逐块执行，每条规则统计命中次数并采样上下文，供质检报告核对是否误伤正文。

不做 NFKC（保持全角标点）。规则集一改就要升 NORMALIZE_VERSION：text.txt 随之变化，已标注的金标需要重新校验。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from novel_rag.ingestion.ir import Document

NORMALIZE_VERSION = "norm-1"

_CJK = r"㐀-䶿一-鿿豈-﫿"
_CJK_OR_FULLWIDTH_PUNCT = _CJK + r"　-〿＀-￯“”‘’…—"
_INTERPUNCT_VARIANTS = "•‧・･∙⋅"
INTERPUNCT = "·"  # U+00B7


@dataclass
class _Rule:
    name: str
    pattern: re.Pattern[str]
    repl: str | Callable[[re.Match[str]], str]


# 顺序有意义：先去零宽字符和多余空白，再统一间隔号，最后修复丢失的间隔号
_RULES = [
    _Rule("zero_width", re.compile("[​-‍⁠﻿]"), ""),
    _Rule("nbsp", re.compile(" "), " "),
    _Rule("whitespace_run", re.compile(r"[ \t\r\n\f\v]{2,}|[\t\r\n\f\v]"), " "),
    _Rule("edge_space", re.compile(r"^[\s　]+|[\s　]+$"), ""),
    _Rule("interpunct", re.compile(rf"(?<=\S)[{_INTERPUNCT_VARIANTS}](?=\S)"), INTERPUNCT),
]
# 半角 '?' 有两种来源，按分册判定后只启用其中一条（见 detect_qmark_mode）：
# - 间隔号在转换中丢失：两个汉字之间的 '?' 修复为 '·'
# - 排版用半角问号：紧跟在中文字符或中文标点之后的 '?' 统一为全角 '？'
_QMARK_REPAIR = _Rule("qmark_repair", re.compile(rf"(?<=[{_CJK}])\?(?=[{_CJK}])"), INTERPUNCT)
_QMARK_FULLWIDTH = _Rule("qmark_fullwidth", re.compile(rf"(?<=[{_CJK_OR_FULLWIDTH_PUNCT}?])\?"), "？")
_QMARK_BETWEEN_CJK = _QMARK_REPAIR.pattern
RULE_NAMES = (*(r.name for r in _RULES), _QMARK_REPAIR.name, _QMARK_FULLWIDTH.name)
QmarkMode = str  # "repair" / "fullwidth" / "none"
# 只计数、不修改：独立出现的间隔号变体（可能是列表圆点）、段内全角空格、汉字之间的空格
# （附录表格用空格分隔各列，删掉会把列粘在一起）
_WATCH = {
    "cjk_space": re.compile(rf"(?<=[{_CJK_OR_FULLWIDTH_PUNCT}]) (?=[{_CJK_OR_FULLWIDTH_PUNCT}])"),
    "interpunct_isolated": re.compile(rf"(?<!\S)[{_INTERPUNCT_VARIANTS}]|[{_INTERPUNCT_VARIANTS}](?!\S)"),
    "inner_fullwidth_space": re.compile(r"(?<=\S)　+(?=\S)"),
}


@dataclass
class NormalizeStats:
    sample_size: int = 8
    sample_sizes: dict[str, int] = field(default_factory=dict)  # 个别规则需要更多样例
    hits: dict[str, int] = field(default_factory=dict)
    samples: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    decisions: list[str] = field(default_factory=list)  # 按分册做出的规则取舍，写进质检报告

    def record(self, name: str, text: str, m: re.Match[str], locator: str) -> None:
        self.hits[name] = self.hits.get(name, 0) + 1
        bucket = self.samples.setdefault(name, [])
        if len(bucket) < self.sample_sizes.get(name, self.sample_size):
            bucket.append(
                {
                    "before": text[max(0, m.start() - 10) : m.start()],
                    "match": m.group(0),
                    "after": text[m.end() : m.end() + 10],
                    "locator": locator,
                }
            )

    def merge(self, other: NormalizeStats) -> None:
        for k, v in other.hits.items():
            self.hits[k] = self.hits.get(k, 0) + v
        for k, v in other.samples.items():
            bucket = self.samples.setdefault(k, [])
            bucket.extend(v[: max(0, self.sample_sizes.get(k, self.sample_size) - len(bucket))])
        self.decisions.extend(other.decisions)


def detect_qmark_mode(texts: list[str], punct_ratio: float) -> tuple[QmarkMode, int, int]:
    """判定一个分册里半角 '?' 的用途，返回（模式, '?' 总数, 夹在两个汉字之间的个数）。

    如果相当比例的 '?' 出现在引号前、行尾等"问号位置"，说明这个分册用半角问号排版，此时
    "汉字?汉字"多半是问号后紧跟下一句，不能当成丢失的间隔号修复。
    """
    total = sum(t.count("?") for t in texts)
    between = sum(len(_QMARK_BETWEEN_CJK.findall(t)) for t in texts)
    if total == 0:
        return "none", 0, 0
    return ("fullwidth" if (total - between) / total >= punct_ratio else "repair"), total, between


def normalize_text(text: str, stats: NormalizeStats, locator: str = "", qmark_mode: QmarkMode = "repair") -> str:
    rules = list(_RULES)
    if qmark_mode == "repair":
        rules.append(_QMARK_REPAIR)
    elif qmark_mode == "fullwidth":
        rules.append(_QMARK_FULLWIDTH)
    for rule in rules:
        matches = list(rule.pattern.finditer(text))
        if not matches:
            continue
        for m in matches:
            stats.record(rule.name, text, m, locator)
        text = rule.pattern.sub(rule.repl, text)
    for name, pattern in _WATCH.items():
        for m in pattern.finditer(text):
            stats.record(name, text, m, locator)
    return text


def normalize_document(
    doc: Document, *, sample_size: int, qmark_sample_size: int, qmark_punct_ratio: float
) -> list[NormalizeStats]:
    """原地规范化全部块文本、标题和注释内容；返回每个分册（与 doc.parts 同序）的统计。

    规范化后为空的段落直接删除（计入 empty_para）；带注释标记的段落不会为空，因为占位符不是空白。
    """
    per_part = []
    for part in doc.parts:
        stats = NormalizeStats(
            sample_size=sample_size,
            sample_sizes={"qmark_repair": qmark_sample_size, "qmark_fullwidth": qmark_sample_size},
        )
        texts = [b.text for s in part.sections for b in s.blocks] + [
            n.text for s in part.sections for b in s.blocks for n in b.notes
        ]
        mode, total, between = detect_qmark_mode(texts, qmark_punct_ratio)
        if total:
            action = {"repair": "两个汉字之间的按丢失的间隔号修复", "fullwidth": "视为半角问号，统一为全角"}[mode]
            stats.decisions.append(
                f"分册「{part.title or '目录前'}」：半角 ? 共 {total} 处，其中 {between} 处夹在两个汉字之间 → {action}"
            )
        if part.title:
            part.title = normalize_text(part.title, stats, part.locator, mode)
        for section in part.sections:
            first = section.blocks[0] if section.blocks else None
            title_is_heading = first is not None and first.kind == "heading" and first.text.strip() == section.title
            if section.title and not title_is_heading:  # 标题就是首个标题块时随块一起规范化，避免重复计数
                section.title = normalize_text(section.title, stats, section.locator, mode)
            kept = []
            for b in section.blocks:
                if b.kind != "image":
                    b.text = normalize_text(b.text, stats, b.locator, mode)
                    for note in b.notes:
                        note.text = normalize_text(note.text, stats, note.note_locator, mode)
                    if not b.text:
                        stats.hits["empty_para"] = stats.hits.get("empty_para", 0) + 1
                        continue
                kept.append(b)
            section.blocks = kept
            if title_is_heading and first is not None:
                section.title = first.text
        per_part.append(stats)
    return per_part
