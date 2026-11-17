"""结果入库前的版权处理（P6）：把字符串里与原文连续重合达到 N 字的片段替换成"[…]"。

用途：评测结果（per_question.jsonl 等）只存 chunk_id 和指标，但 Agent 自己写的检索语句偶尔会照抄原文短句。
P6 的版权自查（scripts/copyright_scan.py）以 30 字为界，这里用同一规则处理：
- run_qa 写结果之前，对每题记录里的所有字符串做一遍（RedactIndex.redact_obj）；
- copyright_scan --fix 对已入库的 JSON / JSONL 结果文件做一遍。
比较时忽略空白（原文与字符串都去掉空白后再找重合），替换的是原字符串里对应的那一段。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_WS = re.compile(r"\s+")
MARK = "[…]"


def strip_ws(text: str) -> tuple[str, list[int]]:
    """去掉空白，返回 (去空白后的文本, 每个字符在原文本中的位置)。"""
    chars, pos = [], []
    for i, ch in enumerate(text):
        if not ch.isspace():
            chars.append(ch)
            pos.append(i)
    return "".join(chars), pos


def find_overlaps(t: str, idx: set[int], n: int, full: str) -> list[tuple[int, int]]:
    """t（已去空白）中与原文连续重合 ≥ n 字的区间 [(起点, 长度)]，每个命中延伸到最长。"""
    out, i = [], 0
    while i <= len(t) - n:
        w = t[i : i + n]
        if hash(w) in idx and w in full:
            j = i + n
            while j < len(t) and t[i : j + 1] in full:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


class RedactIndex:
    def __init__(self, texts: Iterable[str], n: int = 30):
        parts = [_WS.sub("", t) for t in texts]
        self.n = n
        self.full = "\n".join(parts)
        self.idx = {hash(p[i : i + n]) for p in parts for i in range(len(p) - n + 1)}
        self.count = 0  # 替换过的片段数

    def redact(self, s: str) -> str:
        if len(s) < self.n:
            return s
        t, pos = strip_ws(s)
        hits = find_overlaps(t, self.idx, self.n, self.full)
        if not hits:
            return s
        self.count += len(hits)
        out, last = [], 0
        for off, length in hits:
            a, b = pos[off], pos[off + length - 1] + 1
            out += [s[last:a], MARK]
            last = b
        out.append(s[last:])
        return "".join(out)

    def redact_obj(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.redact(obj)
        if isinstance(obj, list):
            return [self.redact_obj(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.redact_obj(v) for k, v in obj.items()}
        return obj
