"""Agent 结果分析：失败自动分类、工具使用统计。输入是 per_question.jsonl 的行（不含原文），不依赖 agent/。

失败分类按环节定位（P4 方案第七节），只对没答对（label != correct）的题：
- 阶段（互斥，按流水线顺序判定）：
  - bridge_missing   多跳题第一跳证据（evidence[0]）从未出现在观察中
  - retrieval_miss   金标证据一段都没出现在观察中
  - retrieval_partial 部分金标出现在观察中，笔记覆盖与观察相同（检索不全）
  - found_not_noted  出现在观察中，但没进笔记（笔记召回 < 过程覆盖率）
  - noted_but_wrong  笔记覆盖了观察到的全部金标，答案仍然错
- 附加标记（可多选）：
  - premature_finish 笔记没有覆盖全部金标，未触及上限就结束了
  - hit_limit        终止原因为 limit_*
  - looping          重复相同或几乎相同的检索
  - entity_ambiguous 实体查找返回多个候选（实体链接可能选错，需人工确认）
  - abstained        弃答
答对的题另外标记 correct_bad_citation：答对了，但引用没有一段与金标重叠。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from novel_rag.eval.dataset import EvalItem
from novel_rag.eval.metrics import Span, is_relevant

STAGES = ("bridge_missing", "retrieval_miss", "retrieval_partial", "found_not_noted", "noted_but_wrong")
FLAGS = ("premature_finish", "hit_limit", "looping", "entity_ambiguous", "abstained")
STAGE_NAMES = {
    "bridge_missing": "桥接没找到",
    "retrieval_miss": "没检索到",
    "retrieval_partial": "检索不全",
    "found_not_noted": "找到没记",
    "noted_but_wrong": "记了答错",
    "premature_finish": "过早结束",
    "hit_limit": "触及上限",
    "looping": "原地打转",
    "entity_ambiguous": "实体歧义（待人工确认）",
    "abstained": "弃答",
    "correct_bad_citation": "答对但引用错",
}


def _bigrams(s: str) -> set[str]:
    s = "".join(s.split())
    return {s[i : i + 2] for i in range(len(s) - 1)} or {s}


def looping(trace: list[dict[str, Any]], threshold: float = 0.8) -> bool:
    """同一工具的两次调用参数相同，或检索语句的字符 bigram Jaccard ≥ threshold。"""
    seen: list[tuple[str, str, set[str]]] = []
    for st in trace:
        if st.get("step") != "tool":
            continue
        args = st.get("args") or {}
        key = str(sorted((k, str(v)) for k, v in args.items() if k != "query"))
        q = _bigrams(str(args.get("query", "")))
        for tool, k2, q2 in seen:
            if tool == st["tool"] and k2 == key and (not q or len(q & q2) / max(len(q | q2), 1) >= threshold):
                return True
        seen.append((st["tool"], key, q))
    return False


def _covered(ids: list[str], ev, spans: dict[str, Span], rel: dict[str, Any]) -> bool:  # noqa: ANN001
    return any(c in spans and is_relevant(spans[c], ev, **rel) for c in ids)


def classify(row: dict[str, Any], item: EvalItem, spans: dict[str, Span], rel: dict[str, Any]) -> dict[str, Any]:
    sets = row.get("evidence_sets", {})
    shown, notes = sets.get("shown", []), sets.get("notes", [])
    ev_shown = [_covered(shown, ev, spans, rel) for ev in item.evidence]
    ev_noted = [_covered(notes, ev, spans, rel) for ev in item.evidence]
    agent = row.get("agent", {})
    flags = []
    term = row.get("termination") or ""
    if row["label"] == "correct":
        cites = row.get("citations", [])
        bad = bool(cites) and not any(_covered([c], ev, spans, rel) for c in cites for ev in item.evidence)
        return {"stage": None, "flags": ["correct_bad_citation"] if bad else []}
    if item.type == "multihop" and not ev_shown[0]:
        stage = "bridge_missing"
    elif not any(ev_shown):
        stage = "retrieval_miss"
    elif sum(ev_noted) < sum(ev_shown):
        stage = "found_not_noted"
    elif not all(ev_shown):
        stage = "retrieval_partial"
    else:
        stage = "noted_but_wrong"
    if not all(ev_noted) and term.startswith("finish"):
        flags.append("premature_finish")
    if term.startswith("limit"):
        flags.append("hit_limit")
    if looping(agent.get("trace", [])):
        flags.append("looping")
    if any(
        st.get("tool") == "find_entity" and len((st.get("meta") or {}).get("candidates", [])) > 1
        for st in agent.get("trace", [])
    ):
        flags.append("entity_ambiguous")
    if row["label"] == "abstain":
        flags.append("abstained")
    return {"stage": stage, "flags": flags}


def failure_table(
    rows: list[dict[str, Any]], items: dict[str, EvalItem], spans: dict[str, Span], rel: dict[str, Any]
) -> dict[str, Any]:
    by_type: dict[str, Counter] = defaultdict(Counter)
    overall: Counter = Counter()
    per_q = {}
    for r in rows:
        c = classify(r, items[r["id"]], spans, rel)
        per_q[r["id"]] = c
        keys = ([c["stage"]] if c["stage"] else []) + c["flags"]
        for k in keys:
            by_type[r["type"]][k] += 1
            overall[k] += 1
        if r["label"] != "correct":
            by_type[r["type"]]["_failed"] += 1
            overall["_failed"] += 1
    return {"overall": dict(overall), "by_type": {t: dict(v) for t, v in by_type.items()}, "per_question": per_q}


def tool_usage(rows: list[dict[str, Any]], items: dict[str, EvalItem], spans: dict[str, Span],
               rel: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    """各工具调用频率、使用率、出错数；正确答案所引（且与金标重叠的）证据最早来自哪个工具。"""
    calls: Counter = Counter()
    used_by: Counter = Counter()
    errors: Counter = Counter()
    by_type: dict[str, Counter] = defaultdict(Counter)
    cite_src: Counter = Counter()
    cite_src_type: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        a = r.get("agent") or {}
        tc = a.get("tool_counts", {})
        calls.update(tc)
        used_by.update(tc.keys())
        by_type[r["type"]].update(tc)
        for st in a.get("trace", []):
            if st.get("step") == "tool" and not st.get("ok"):
                errors[st["tool"]] += 1
        if r["label"] != "correct":
            continue
        srcs = a.get("note_sources", {})
        for c in r.get("citations", []):
            if not any(_covered([c], ev, spans, rel) for ev in items[r["id"]].evidence):
                continue
            first = (srcs.get(c) or ["unknown"])[0]
            cite_src[first] += 1
            cite_src_type[r["type"]][first] += 1
    n = len(rows)
    return {
        "n_questions": n,
        "calls": dict(calls),
        "calls_per_question": {k: round(v / n, 3) for k, v in calls.items()} if n else {},
        "questions_using": {k: round(v / n, 3) for k, v in used_by.items()} if n else {},
        "errors": dict(errors),
        "calls_by_type": {t: dict(v) for t, v in by_type.items()},
        "correct_cited_gold_source": dict(cite_src),
        "correct_cited_gold_source_by_type": {t: dict(v) for t, v in cite_src_type.items()},
    }
