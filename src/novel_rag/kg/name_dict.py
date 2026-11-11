"""从图谱导出 IK 扩展词典（人名、组织名、地名、头衔及其别名），回灌 ES 的分词。

收录：实体规范名与可见别名，带与不带间隔号两种写法；间隔号名字的各片段（≥ 2 字，"琼恩·雪诺" → "琼恩"、"雪诺"）。
排除：泛称（names.GENERIC_TERMS + 书籍配置）、单字、超过 max_len 的长串、含"的"的短语、非中文写法、提及过少的实体，
以及 IK 默认词典已经整体成词的写法（加了也没用）。
词典是全局的：所有用 IK 的索引在 ES 重启后查询端分词都会变，已有索引要重建（新建 es2，es1 只用于回灌前的对比）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from novel_rag.kg.names import NameRules, normalize_name, segments


def candidate_words(
    entities: Sequence[dict[str, Any]],
    rules: NameRules,
    *,
    min_mentions: int = 2,
    max_len: int = 10,
    types: Iterable[str] | None = None,
) -> dict[str, int]:
    """词 → 提及数（同一个词来自多个实体时取最大值，用于排序与统计）。"""
    allowed = set(types) if types else {"Person", "Organization", "Location", "Title"}
    words: dict[str, int] = {}

    def add(w: str, n: int) -> None:
        w = normalize_name(w)
        if len(w) < 2 or len(w) > max_len or rules.is_generic(w) or any(ch.isspace() for ch in w):
            return
        if "的" in w or not any("一" <= ch <= "鿿" for ch in w):  # 短语（"某某的某某"）与非中文写法不收
            return
        words[w] = max(words.get(w, 0), n)

    for e in entities:
        if e["type"] not in allowed or e.get("n_mentions", 0) < min_mentions:
            continue
        n = int(e.get("n_mentions", 0))
        forms = [e["name"], *(a["alias"] for a in e.get("aliases", []))]
        for f in forms:
            f = normalize_name(f)
            add(f, n)
            if "·" in f:
                add(f.replace("·", ""), n)
                if e["type"] == "Person":
                    for s in segments(f):
                        add(s, n)
    return words


def filter_known(words: dict[str, int], is_single_token: Callable[[list[str]], list[bool]]) -> dict[str, int]:
    """去掉 IK 默认词典已经整体成词的写法。is_single_token 批量返回每个词在当前分词器下是否只切出一个 token。"""
    items = sorted(words)
    flags = is_single_token(items)
    return {w: words[w] for w, single in zip(items, flags, strict=True) if not single}


def render_dic(words: dict[str, int]) -> str:
    """每行一个词，按提及数降序、再按字典序；UTF-8 无 BOM。"""
    return "".join(w + "\n" for w in sorted(words, key=lambda w: (-words[w], w)))
