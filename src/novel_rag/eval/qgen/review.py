"""人工校验的状态机。校验日志 review.jsonl 只追加不修改；当前状态由重放日志得到，因此可以随时中断、撤销、续做。

日志记录（不含原文）：
    {"cand_id", "action": "accept" | "edit" | "drop" | "respan" | "undo", ...}
    accept：可带 difficulty（1–3）
    edit：question / answer（未改的字段不出现），可带 difficulty
    drop：reason（见 DROP_REASONS）
    respan：evidence（替换后的完整证据列表）
    undo：撤销同一题的上一条有效记录
"""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any

from novel_rag.corpus import BookText
from novel_rag.eval.dataset import QUESTION_TYPES
from novel_rag.eval.qgen.locate import locate

DROP_REASONS = {
    "1": "too_easy",  # 太简单
    "2": "ambiguous",  # 有歧义 / 答案不唯一
    "3": "weak_evidence",  # 证据不足以支撑答案
    "4": "wrong_type",  # 题型不符
    "5": "wrong_answer",  # 答案错误
    "6": "duplicate",  # 与其他题重复
    "7": "other",
}


@dataclass
class ReviewState:
    cand: dict[str, Any]  # 导入时的候选（located_r*.jsonl 中的一行）
    status: str = "pending"  # pending / accepted / edited / dropped
    question: str = ""
    answer: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    difficulty: int | None = None
    drop_reason: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)  # 生效中的记录，供 undo 回退
    answer_aliases: list[str] = field(default_factory=list)

    @classmethod
    def fresh(cls, cand: dict[str, Any]) -> ReviewState:
        return cls(
            cand,
            question=cand["question"],
            answer=cand["answer"],
            evidence=deepcopy(cand["evidence"]),
            answer_aliases=list(cand.get("answer_aliases", [])),
        )

    @property
    def changed(self) -> bool:
        return (
            self.question != self.cand["question"]
            or self.answer != self.cand["answer"]
            or self.answer_aliases != list(self.cand.get("answer_aliases", []))
            or [(e["chapter_idx"], e["char_start"], e["char_end"]) for e in self.evidence]
            != [(e["chapter_idx"], e["char_start"], e["char_end"]) for e in self.cand["evidence"]]
        )


def _apply(state: ReviewState, rec: dict[str, Any]) -> None:
    action = rec["action"]
    if action == "accept":
        state.status = "edited" if state.changed else "accepted"
        state.difficulty = rec.get("difficulty", state.difficulty)
    elif action == "edit":
        state.question = rec.get("question", state.question)
        state.answer = rec.get("answer", state.answer)
        state.answer_aliases = list(rec.get("answer_aliases", state.answer_aliases))
        state.difficulty = rec.get("difficulty", state.difficulty)
        state.status = "edited" if state.changed else "accepted"
    elif action == "respan":
        state.evidence = deepcopy(rec["evidence"])
        state.status = "pending"  # 改了证据要再确认一次
    elif action == "drop":
        state.status = "dropped"
        state.drop_reason = rec.get("reason", "other")
    else:
        raise ValueError(f"未知的校验动作：{action}")


def replay(cands: list[dict[str, Any]], log: list[dict[str, Any]]) -> dict[str, ReviewState]:
    states = {c["cand_id"]: ReviewState.fresh(c) for c in cands}
    for rec in log:
        st = states.get(rec["cand_id"])
        if st is None:
            continue
        if rec["action"] == "undo":
            if st.history:
                st.history.pop()
            fresh = ReviewState.fresh(st.cand)
            for r in st.history:
                _apply(fresh, r)
            fresh.history = st.history
            states[rec["cand_id"]] = fresh
            continue
        _apply(st, rec)
        st.history.append(rec)
    return states


def review_order(cands: list[dict[str, Any]]) -> list[str]:
    """按题型轮流排列（同一题型内保持原顺序），避免长时间只看一种题。"""
    by_type: dict[str, list[str]] = {t: [] for t in QUESTION_TYPES}
    for c in cands:
        by_type.setdefault(c["type"], []).append(c["cand_id"])
    return [cid for group in zip_longest(*by_type.values()) for cid in group if cid is not None]


class ReviewLog:
    def __init__(self, path: Path):
        self.path = path

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, rec: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = {**rec, "ts": datetime.now().isoformat(timespec="seconds")}
        with self.path.open("a", encoding="utf-8", newline="") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def review_stats(states: dict[str, ReviewState]) -> dict[str, Any]:
    """按题型与来源统计通过率、修改率和丢弃原因。通过率 = (通过 + 修改后通过) / 已校验。"""

    def summarize(group: list[ReviewState]) -> dict[str, Any]:
        c = Counter(s.status for s in group)
        reviewed = c["accepted"] + c["edited"] + c["dropped"]
        kept = c["accepted"] + c["edited"]
        return {
            "total": len(group),
            "reviewed": reviewed,
            "accepted": c["accepted"],
            "edited": c["edited"],
            "dropped": c["dropped"],
            "pending": c["pending"],
            "pass_rate": round(kept / reviewed, 3) if reviewed else None,
            "edit_rate": round(c["edited"] / kept, 3) if kept else None,
            "drop_reasons": dict(Counter(s.drop_reason for s in group if s.status == "dropped")),
        }

    all_states = list(states.values())
    out: dict[str, Any] = {"overall": summarize(all_states), "by_type": {}, "by_source": {}}
    for t in sorted({s.cand["type"] for s in all_states}):
        out["by_type"][t] = summarize([s for s in all_states if s.cand["type"] == t])
    for src in sorted({s.cand.get("source", "generated") for s in all_states}):
        out["by_source"][src] = summarize([s for s in all_states if s.cand.get("source", "generated") == src])
    return out


def respan_evidence(
    book: BookText,
    evidence: list[dict[str, Any]],
    index: int | None,
    quote: str,
    chapter_idx: int,
    *,
    fuzzy_threshold: float,
) -> list[dict[str, Any]] | None:
    """在指定章节内定位新的原文片段，替换第 index 段证据（index 为 None 时追加）。定位失败返回 None。"""
    ch = book.chapter(chapter_idx)
    loc = locate(book.text, quote, ch.char_start, ch.char_end, fuzzy_threshold=fuzzy_threshold)
    if loc is None:
        return None
    new = {
        "window_id": f"ch{chapter_idx}",
        "book_id": book.book_id,
        "chapter_idx": chapter_idx,
        "char_start": loc.start,
        "char_end": loc.end,
        "method": loc.method,
        "score": loc.score,
    }
    out = deepcopy(evidence)
    if index is None:
        out.append(new)
    else:
        out[index] = new
    return out


def load_candidates(work: Path) -> list[dict[str, Any]]:
    """所有通过定位与预筛的候选：第一轮、第二轮、人工题。"""
    cands = []
    for name in ("located_r1.jsonl", "located_r2.jsonl", "located_human.jsonl"):
        p = work / name
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    c = json.loads(line)
                    if c["status"] == "ok":
                        cands.append(c)
    return cands
