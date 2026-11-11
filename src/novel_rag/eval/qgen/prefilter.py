"""候选题自动预筛。硬规则直接丢弃（记录原因）；软规则只打标，人工校验时优先查看。"""

from __future__ import annotations

from typing import Any

from novel_rag.config import EvalGenConfig
from novel_rag.corpus import BookText

# 必须跨越多个章节的题型
_MULTI_CHAPTER = {"multihop", "plot"}


def longest_common_substring(a: str, b: str) -> int:
    if not a or not b:
        return 0
    best, prev = 0, [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def _bigrams(s: str) -> set[str]:
    s = "".join(ch for ch in s if not ch.isspace())
    return {s[i : i + 2] for i in range(len(s) - 1)}


def jaccard(a: str, b: str) -> float:
    x, y = _bigrams(a), _bigrams(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def hard_reasons(cand: dict[str, Any]) -> list[str]:
    reasons = []
    answer = cand["answer"].strip()
    if len(answer) >= 2 and answer in cand["question"]:
        reasons.append("answer_in_question")
    if cand["type"] in _MULTI_CHAPTER:
        if len({e["window_id"] for e in cand["evidence"]}) < 2:
            reasons.append("single_window")
        elif len({(e.get("book_id"), e["chapter_idx"]) for e in cand["evidence"]}) < 2:  # 跨卷时章号会重
            reasons.append("single_chapter")
    return reasons


def apply_prefilter(
    cands: list[dict[str, Any]],
    book: BookText | dict[str, BookText],
    cfg: EvalGenConfig,
    *,
    earlier: list[dict[str, Any]],
) -> None:
    """原地设置 status / drop_reasons / flags。earlier 是之前轮次已通过定位的候选，用于跨轮查重。
    book 可以是单卷（v0）或 book_id → BookText（多卷，证据按各自的 book_id 取原文）。"""
    books = book if isinstance(book, dict) else None
    seen = [c for c in earlier if c["status"] == "ok"]
    for c in cands:
        if c["status"] != "ok":
            continue
        reasons = hard_reasons(c)
        if reasons:
            c["status"] = "dropped"
            c["drop_reasons"].extend(reasons)
            continue
        evid_text = "".join(
            (books[e["book_id"]] if books is not None else book).text[e["char_start"] : e["char_end"]]  # type: ignore[union-attr]
            for e in c["evidence"]
        )
        lcs = longest_common_substring(c["question"], evid_text)
        if lcs >= cfg.copy_lcs_chars:
            c["flags"].append(f"copy:{lcs}")
        if len(c["answer"]) > cfg.answer_max_chars:
            c["flags"].append("answer_long")
        if len(c["question"]) > cfg.question_max_chars:
            c["flags"].append("question_long")
        if any(e["method"] == "fuzzy" for e in c["evidence"]):
            c["flags"].append("fuzzy_evidence")
        dup = next((o["cand_id"] for o in seen if jaccard(o["question"], c["question"]) >= cfg.dup_jaccard), None)
        if dup:
            c["flags"].append(f"dup:{dup}")
        seen.append(c)
