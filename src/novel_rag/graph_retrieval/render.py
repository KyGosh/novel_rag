"""把图谱查询结果转成给模型 / 人看的文本：每条事实带证据 chunk_id 和"卷名 · 章节标题"，总长度有上限。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from novel_rag.graph_retrieval.templates import QueryResult
from novel_rag.kg.schema import KIN_ROLES, REL_BY_NAME

TRUNC_NOTE = "……（结果过长，已截断）"


def relation_label(rel_type: str, role: str | None) -> str:
    label = REL_BY_NAME[rel_type].label if rel_type in REL_BY_NAME else rel_type
    return f"{label}（{KIN_ROLES[role]}）" if role and role in KIN_ROLES else label


def fact_text(subject: str, rel_type: str, role: str | None, obj: str, outgoing: bool) -> str:
    s, o = (subject, obj) if outgoing else (obj, subject)
    return f"{s} —{relation_label(rel_type, role)}→ {o}"


def _ev(evidence: list[dict[str, Any]], where: Callable[[str], str]) -> str:
    return "；".join(f"{e['chunk_id']}（{where(e['chunk_id'])}）" for e in evidence)


def render(result: QueryResult, where: Callable[[str], str], max_chars: int) -> str:
    lines: list[str] = []
    op, rows = result.op, result.rows
    if not rows:
        return f"[{op}] 没有找到（在当前阅读进度内）。"
    if op == "resolve":
        for r in rows:
            q = f"（{r['qualifier']}）" if r.get("qualifier") else ""
            facts = "；".join(fact_text(r["name"], f["type"], f["role"], f["other"], f["outgoing"]) for f in r["facts"])
            alias = "、".join(a for a in r["aliases"][:6] if a != r["name"])
            lines.append(
                f"- {r['name']}{q} [{r['entity_id']}] {r['type']}；别名：{alias or '无'}；{facts or '暂无已知关系'}"
            )
    elif op == "profile":
        r = rows[0]
        lines.append(
            f"{r['name']} [{r['entity_id']}] {r['type']}，"
            f"首次 / 末次出现 reading_order {r['first_seen']}–{r['last_seen']}"
        )
        lines.append(f"别名：{'、'.join(r['aliases']) or '无'}；所属：{'、'.join(r['groups']) or '不详'}")
        for x in r["relations"]:
            lines.append(
                f"- {fact_text(r['name'], x['type'], x['role'], x['other'], x['outgoing'])}"
                f"（证据 {x['n_evidence']} 条，ro {x['first_ro']}–{x['last_ro']}）证据：{_ev(x['evidence'], where)}"
            )
    elif op in ("neighbors",):
        for x in rows:
            lines.append(
                f"- {fact_text('该实体', x['type'], x['role'], x['other'], x['outgoing'])} [{x['other_id']}]"
                f"（证据 {x['n_evidence']} 条）证据：{_ev(x['evidence'], where)}"
            )
    elif op == "between":
        for x in rows:
            lines.append(
                f"- {fact_text('A', x['type'], x['role'], 'B', x['a_to_b'])}（证据 {x['n_evidence']} 条，"
                f"ro {x['first_ro']}–{x['last_ro']}）证据：{_ev(x['evidence'], where)}"
            )
    elif op == "paths":
        for k, p in enumerate(rows, 1):
            names = {n["id"]: n["name"] for n in p["nodes"]}
            hops = [
                f"{names.get(r['from'], r['from'])} —{relation_label(r['type'], r['role'])}→ "
                f"{names.get(r['to'], r['to'])}"
                f"（{_ev(r['evidence'], where)}）"
                for r in p["rels"]
            ]
            lines.append(f"路径 {k}：" + " ／ ".join(hops))
    elif op == "timeline":
        lines.append("（按阅读顺序排列；阅读顺序不等于故事内的时间顺序）")
        for x in rows:
            if x["kind"] == "event":
                lines.append(
                    f"- ro {x['ro']}：{x['description']}（{_ev([{'chunk_id': c} for c in x['chunk_ids'][:2]], where)}）"
                )
            else:
                lines.append(
                    f"- ro {x['ro']}：首次出现关系 "
                    f"{fact_text('该实体', x['type'], x['role'], x['other'], x['outgoing'])}"
                    f"（{_ev(x['evidence'], where)}）"
                )
    elif op == "mentions":
        for x in rows:
            lines.append(f"- {x['chunk_id']}（{where(x['chunk_id'])}）提及 {x['count']} 次")
    elif op == "title_holders":
        for x in rows:
            lines.append(
                f"- {x['holder']} [{x['holder_id']}]：ro {x['first_ro']}–{x['last_ro']}，证据 {x['n_evidence']} 条："
                f"{_ev(x['evidence'], where)}"
            )
    text = f"[{op}]\n" + "\n".join(lines)
    if result.truncated:
        text += "\n（结果已按上限截断，可缩小范围或指定关系类型）"
    if len(text) > max_chars:
        text = text[: max_chars - len(TRUNC_NOTE)] + TRUNC_NOTE
    return text
