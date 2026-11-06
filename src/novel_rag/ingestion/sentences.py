"""中文切句：只输出字符偏移，不拷贝字符串。段落（text.txt 中的一行）边界必然是句子边界。

规则：
- 句末标点 `。！？!?` 处断句；紧随其后的句末标点和后引号 / 后括号一并归入当前句。
- 引语以句末标点收尾、后面紧跟归属语（"他说"）时不断开，对话与归属语算同一句。
- 省略号 `……` 只有后面紧跟后引号或位于段尾时才算句末。
- 引号内不断句；但引号内已累积超过 long_quote_chars 字时，允许在句末标点处断开（长篇对话）。
- 超过 sentence_hard_max 字的句子在逗号 / 分号 / 顿号处硬切，找不到就按长度硬切。
"""

from __future__ import annotations

from novel_rag.config import ChunkConfig

SPLITTER_VERSION = "sent-1"

_END = frozenset("。！？!?")
_ELLIPSIS = "…"
_QUOTES = {"“": "”", "「": "」", "『": "』", "‘": "’"}
_CLOSE_QUOTES = frozenset(_QUOTES.values())
_CLOSERS = _CLOSE_QUOTES | frozenset("）)】》〉")
_SOFT_BREAKS = frozenset("，；、,;：:")


def split_sentences(text: str, start: int, end: int, cfg: ChunkConfig) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    sent_start = start
    depth = 0
    quote_seg_start = start  # 当前引号段（或上次在引号内断开处）的起点
    i = start
    while i < end:
        ch = text[i]
        if ch in _QUOTES:
            depth += 1
            if depth == 1:
                quote_seg_start = i
            i += 1
            continue
        if ch in _CLOSE_QUOTES:
            depth = max(0, depth - 1)
            i += 1
            continue
        is_end = ch in _END
        if ch == _ELLIPSIS:
            j = i
            while j < end and text[j] == _ELLIPSIS:
                j += 1
            is_end = j >= end or text[j] in _CLOSERS
            if not is_end:
                i = j
                continue
        if not is_end:
            i += 1
            continue
        # 吞掉连续的句末标点、省略号，以及其后的后引号 / 后括号
        j = i + 1
        while j < end and (text[j] in _END or text[j] == _ELLIPSIS):
            j += 1
        closed_quote = False
        while j < end and text[j] in _CLOSERS:
            if text[j] in _CLOSE_QUOTES:
                depth = max(0, depth - 1)
                closed_quote = True
            j += 1
        # 引语以句末标点收尾、后面紧跟"他说"之类的归属语时，不断开，让对话与归属语留在同一句
        if closed_quote and depth == 0 and j < end and text[j] not in _QUOTES:
            i = j
            continue
        if depth == 0 or i - quote_seg_start >= cfg.long_quote_chars:
            spans.append((sent_start, j))
            sent_start = j
            if depth > 0:
                quote_seg_start = j
        i = j
    if sent_start < end:
        spans.append((sent_start, end))

    out: list[tuple[int, int]] = []
    for s, e in spans:
        out.extend(_hard_cut(text, s, e, cfg.sentence_hard_max))
    return out


def _hard_cut(text: str, s: int, e: int, limit: int) -> list[tuple[int, int]]:
    out = []
    while e - s > limit:
        cut = next((k + 1 for k in range(s + limit - 1, s + limit // 2, -1) if text[k] in _SOFT_BREAKS), s + limit)
        out.append((s, cut))
        s = cut
    out.append((s, e))
    return out
