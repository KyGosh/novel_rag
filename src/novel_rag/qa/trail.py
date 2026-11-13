"""P5 失败模式与收益来源分析的纯函数：题目分组、金标证据在 Agent 过程中的轨迹、收益机制归类。

输入都是 per_question.jsonl 的行（不含原文）和评测集条目；相关性沿用字符区间重叠（eval.metrics.is_relevant）。
不依赖 agent/，不调用任何模型。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from novel_rag.eval.dataset import EvalItem
from novel_rag.eval.metrics import Span, is_relevant

# 金标证据第一次以原文形式出现的方式
MODES = ("search_plain", "search_entity", "search_scoped", "read_via_ref", "read_adjacent", "read_direct")
MODE_NAMES = {
    "search_plain": "原文检索（换检索语句，不带限定）",
    "search_entity": "原文检索（按实体限定）",
    "search_scoped": "原文检索（限定卷 / 章节范围）",
    "read_via_ref": "读原文（编号来自摘要 / 图谱 / 检索的未展示段）",
    "read_adjacent": "读原文（已看过段落的同章邻近段）",
    "read_direct": "读原文（直接给编号）",
}


def is_correct(row: dict[str, Any]) -> bool:
    return row["label"] == "correct"


def question_groups(
    rag: dict[str, dict[str, Any]], a0: dict[str, dict[str, Any]], a1: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
    """四个题组（P5 方案第三节）。回退 / 收益按第一次运行 a0 定义，第二次运行的另列（*_s1）。"""
    ids = sorted(set(rag) & set(a0) & set(a1))
    return {
        "stable_fail": [i for i in ids if not is_correct(a0[i]) and not is_correct(a1[i])],
        "unstable": [i for i in ids if is_correct(a0[i]) != is_correct(a1[i])],
        "regress": [i for i in ids if is_correct(rag[i]) and not is_correct(a0[i])],
        "regress_s1": [i for i in ids if is_correct(rag[i]) and not is_correct(a1[i])],
        "gain": [i for i in ids if not is_correct(rag[i]) and is_correct(a0[i])],
        "gain_s1": [i for i in ids if not is_correct(rag[i]) and is_correct(a1[i])],
        "stable_correct": [i for i in ids if is_correct(a0[i]) and is_correct(a1[i])],
    }


def _hit(ids: Sequence[str], ev: Any, spans: dict[str, Span], rel: dict[str, Any]) -> str | None:
    return next((c for c in ids if c in spans and is_relevant(spans[c], ev, **rel)), None)


def _chapter(chunk_id: str) -> str:
    return chunk_id.rsplit(":", 1)[0]


def _search_mode(st: dict[str, Any]) -> str:
    args, meta = st.get("args") or {}, st.get("meta") or {}
    if args.get("entity_id") and "entity_chunks" in meta:  # 限定确实生效了（太宽时会被忽略）
        return "search_entity"
    if args.get("volume") or args.get("chapter_from") or args.get("chapter_to"):
        return "search_scoped"
    return "search_plain"


def evidence_trail(
    row: dict[str, Any], item: EvalItem, spans: dict[str, Span], rel: dict[str, Any]
) -> list[dict[str, Any]]:
    """每段金标证据在一次 Agent 运行中的轨迹：第一次只给出编号（ref）、第一次展示原文（shown）、是否进笔记、是否被引用。

    shown 的 mode 见 MODES；read_text 的来源按"这个编号此前在哪里出现过"判断。
    """
    steps = [st for st in (row.get("agent") or {}).get("trace", []) if st.get("step") == "tool" and st.get("ok")]
    sets = row.get("evidence_sets", {})
    notes, cited = sets.get("notes", []), row.get("citations", [])
    out = []
    for ev in item.evidence:
        rec: dict[str, Any] = {"ref": None, "shown": None}
        seen_refs: dict[str, str] = {}  # chunk → 最早给出它编号的工具
        seen_chapters: set[str] = set()
        for st in steps:
            tool = st["tool"]
            if rec["ref"] is None and (c := _hit(st.get("refs", []), ev, spans, rel)):
                rec["ref"] = {"round": st["round"], "tool": tool, "chunk_id": c}
            if rec["shown"] is None and (c := _hit(st.get("shown", []), ev, spans, rel)):
                if tool == "search_text":
                    mode = _search_mode(st)
                elif c in seen_refs:
                    mode = "read_via_ref"
                elif _chapter(c) in seen_chapters:
                    mode = "read_adjacent"
                else:
                    mode = "read_direct"
                rec["shown"] = {"round": st["round"], "tool": tool, "mode": mode, "chunk_id": c,
                                "ref_from": seen_refs.get(c)}  # fmt: skip
            for c in st.get("refs", []):
                seen_refs.setdefault(c, tool)
            seen_chapters.update(_chapter(c) for c in st.get("shown", []))
        rec["noted"] = _hit(notes, ev, spans, rel) is not None
        rec["cited"] = _hit(cited, ev, spans, rel) is not None
        out.append(rec)
    return out


def rag_evidence(
    row: dict[str, Any], item: EvalItem, spans: dict[str, Span], rel: dict[str, Any]
) -> list[dict[str, Any]]:
    """固定 RAG：每段金标证据是否进了作答上下文、在结果池里的排名（没进池为 None）。"""
    ctx = row.get("evidence_sets", {}).get("context", [])
    hits = row.get("hits", [])
    out = []
    for ev in item.evidence:
        rank = next((r for r, c in enumerate(hits, 1) if c in spans and is_relevant(spans[c], ev, **rel)), None)
        out.append({"in_context": _hit(ctx, ev, spans, rel) is not None, "rank": rank})
    return out


def gain_source(
    rag_row: dict[str, Any], agent_row: dict[str, Any], item: EvalItem, spans: dict[str, Span], rel: dict[str, Any]
) -> dict[str, Any]:
    """一道收益题（固定 RAG 没答对、Agent 答对）赢在哪里。

    kind：
    - rag_had_evidence   固定 RAG 的上下文里金标证据已经齐全，却没答对 → 收益来自作答前的证据整理（笔记），不是检索
    - retrieval          Agent 以原文形式找到了固定 RAG 上下文里没有的金标证据（mechanisms 给出找到的方式）
    - other_evidence     Agent 没有比固定 RAG 多找到金标证据也答对了（靠金标之外的段落，或图谱 / 摘要线索）
    """
    trail = evidence_trail(agent_row, item, spans, rel)
    rag = rag_evidence(rag_row, item, spans, rel)
    missed = [i for i, r in enumerate(rag) if not r["in_context"]]
    found = [i for i in missed if trail[i]["shown"] is not None]
    mechanisms = sorted({trail[i]["shown"]["mode"] for i in found})
    rounds = [trail[i]["shown"]["round"] for i in found]
    if not missed:
        kind = "rag_had_evidence"
    elif found:
        kind = "retrieval"
    else:
        kind = "other_evidence"
    tools = (agent_row.get("agent") or {}).get("tool_counts", {})
    return {
        "kind": kind,
        "n_evidence": len(item.evidence),
        "rag_missed": len(missed),
        "agent_found_missed": len(found),
        "mechanisms": mechanisms,
        "first_found_round": min(rounds) if rounds else None,
        "last_found_round": max(rounds) if rounds else None,
        "rag_missed_ranks": [rag[i]["rank"] for i in missed],
        "used_find_entity": bool(tools.get("find_entity")),
        "used_graph_query": bool(tools.get("graph_query")),
        "used_summaries": bool(tools.get("search_summaries") or tools.get("read_summary")),
        "rounds": (agent_row.get("agent") or {}).get("rounds"),
        "trail": trail,
        "rag": rag,
    }


def evidence_state(trail: list[dict[str, Any]]) -> str:
    """一次运行的证据到位情况（跨章节题专项）：all_noted / all_shown（看到了没记全）/ partial / none。"""
    shown = [t["shown"] is not None for t in trail]
    noted = [t["noted"] for t in trail]
    if all(noted):
        return "all_noted"
    if all(shown):
        return "all_shown"
    if any(shown) or any(noted):
        return "partial"
    return "none"
