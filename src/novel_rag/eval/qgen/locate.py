"""证据原文片段 → text.txt 中的字符区间。三级依次尝试：精确 → 宽松（统一引号、标点宽度、间隔号，去空白）→ 模糊对齐。

宽松与模糊匹配都在"归一化后的字符串"上进行，再通过偏移映射还原成原文区间，保证结果是 text.txt 的真实偏移。
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

_CANON = str.maketrans(
    {
        "“": '"', "”": '"', "„": '"', "「": '"', "」": '"', "『": '"', "』": '"', "＂": '"',
        "‘": "'", "’": "'", "＇": "'",
        "，": ",", "。": ".", "？": "?", "！": "!", "：": ":", "；": ";", "（": "(", "）": ")",
        "•": "·", "‧": "·", "・": "·", "･": "·", "∙": "·", "⋅": "·",
        "⋯": "…",
    }
)  # fmt: skip


@dataclass(frozen=True)
class Located:
    start: int
    end: int
    method: str  # exact / loose / fuzzy
    score: float = 100.0


def _loose(s: str, offset: int = 0) -> tuple[str, list[int]]:
    """归一化字符串，以及每个归一化字符对应的原文偏移。"""
    chars, mapping = [], []
    for i, ch in enumerate(s):
        if ch.isspace():
            continue
        chars.append(ch.translate(_CANON))
        mapping.append(offset + i)
    return "".join(chars), mapping


def locate(text: str, quote: str, lo: int, hi: int, *, fuzzy_threshold: float) -> Located | None:
    """在 text[lo:hi] 中定位 quote；找不到返回 None。"""
    quote = quote.strip()
    if not quote:
        return None
    i = text.find(quote, lo, hi)
    if i != -1:
        return Located(i, i + len(quote), "exact")

    hay, mapping = _loose(text[lo:hi], lo)
    needle, _ = _loose(quote)
    if not needle or not hay:
        return None
    j = hay.find(needle)
    if j != -1:
        return Located(mapping[j], mapping[j + len(needle) - 1] + 1, "loose")

    al = fuzz.partial_ratio_alignment(needle, hay, score_cutoff=fuzzy_threshold)
    if al is None or al.dest_end <= al.dest_start:
        return None
    if (al.dest_end - al.dest_start) < 0.8 * len(needle):  # 只对上了片段的一小部分
        return None
    return Located(mapping[al.dest_start], mapping[al.dest_end - 1] + 1, "fuzzy", round(al.score, 1))
