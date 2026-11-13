"""图谱工具：实体查找（名字 → 候选实体）与图谱查询（一个工具、多种操作）。

底座是 graph_retrieval.templates 的参数化只读 Cypher 模板（强制 LIMIT、带阅读进度、结果复核），不做 Text2Cypher。
每条关系带证据条数和证据 chunk_id；单条证据的关系注明"建议回原文核实"（P3：单证据关系可信度低）。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from novel_rag.graph_retrieval.render import render
from novel_rag.graph_retrieval.templates import STRUCTURAL as STRUCTURAL_RELS
from novel_rag.graph_retrieval.templates import QueryError
from novel_rag.kg.schema import ENTITY_LABELS, EVENT, REL_BY_NAME
from novel_rag.tools.backends import ToolBackends
from novel_rag.tools.base import ToolContext, ToolInputError, ToolNotFound, ToolResult, truncate
from novel_rag.tools.text import _Args

RO_KEYS = ("ro", "first_ro", "last_ro", "first_seen", "last_seen")


def collect(rows: Any) -> tuple[list[str], list[int]]:
    """从查询结果中收集证据 chunk_id 与所有位置字段（防剧透断言 + 笔记可引用范围）。"""
    ids: list[str] = []
    ros: list[int] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "chunk_id" and isinstance(v, str):
                    ids.append(v)
                elif k == "chunk_ids" and isinstance(v, list):
                    ids.extend(c for c in v if isinstance(c, str))
                elif k in RO_KEYS and isinstance(v, int):
                    ros.append(v)
                elif k == "ros" and isinstance(v, list):
                    ros.extend(r for r in v if isinstance(r, int))
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(rows)
    return list(dict.fromkeys(ids)), ros


def _graph(b: ToolBackends) -> Any:
    if b.graph is None:
        raise ToolInputError("当前环境没有知识图谱")
    return b.graph


def _in_scope(g: Any, entity_id: str, ctx: ToolContext) -> None:
    """图谱范围限定（运行环境注入）：实体类型不在范围内时按"找不到"处理，不向模型暴露范围之外的实体。"""
    scope = ctx.graph_entity_types
    if not scope:
        return
    t = g.entity_type(entity_id, ctx.max_reading_order)
    if t is not None and t not in scope:
        raise ToolNotFound(f"图谱里没有实体 {entity_id}（当前图谱只包含 {'、'.join(scope)} 类实体）")


def _scope_rows(op: str, rows: list[dict[str, Any]], scope: tuple[str, ...]) -> list[dict[str, Any]]:
    """把查询结果裁剪到范围内的实体类型（neighbors / paths 已在查询参数里限定，这里处理其余操作）。"""
    if op == "profile":
        for r in rows:
            r["relations"] = [x for x in r.get("relations", []) if x.get("other_type") in scope]
            r["groups"] = [x["other"] for x in r["relations"] if x["type"] == "MEMBER_OF" and x["outgoing"]]
        return rows
    if op == "timeline":  # 事件节点不在范围内时不列事件，只留范围内实体之间的关系
        return [x for x in rows if (x.get("kind") == "event" and EVENT in scope)
                or (x.get("kind") == "relation" and x.get("other_type") in scope)]  # fmt: skip
    return rows


class FindEntityArgs(_Args):
    name: str = Field(min_length=1, max_length=40, description="人名、地名、组织名、头衔或别名")


FIND_ENTITY_DESC = (
    "实体查找：名字 → 图谱中的候选实体（同名时返回多个），附实体 ID、类型、可见别名和几条主要关系。"
    "什么时候用：要用图谱查询或按实体限定检索之前，先拿到实体 ID；也用来确认别名、称号指的是谁。"
    "写法也是常用词时会提示，需要自己判断问题里指的是不是它。"
)


def find_entity(args: FindEntityArgs, ctx: ToolContext, b: ToolBackends) -> ToolResult:
    from novel_rag.config import get_settings

    cfg = get_settings().tools
    res = _graph(b).resolve(args.name, ctx.max_reading_order, limit=cfg.entity_max_candidates)
    if scope := ctx.graph_entity_types:
        res.rows = [r for r in res.rows if r["type"] in scope]
        for r in res.rows:
            r["facts"] = [f for f in r.get("facts", []) if f.get("other_type") in scope]
    if not res.rows:
        raise ToolNotFound(f"图谱里没有找到“{args.name}”（在当前阅读进度内）；可以换个写法，或直接用 search_text")
    text = render(res, b.where_chunk, cfg.graph_max_chars)
    _, ros = collect(res.rows)
    return ToolResult(
        "find_entity",
        True,
        text,
        reading_orders=ros,
        meta={
            "candidates": [r["entity_id"] for r in res.rows],
            "names": [f"{r['name']}[{r['entity_id']}]（{r['type']}）" for r in res.rows],
        },
    )


GraphOp = Literal["profile", "between", "paths", "neighbors", "timeline", "title_holders", "mentions"]
_REL_NAMES = [r for r in REL_BY_NAME if r not in STRUCTURAL_RELS]
_ENTITY_TYPES = [t for t in ENTITY_LABELS if t != EVENT]


class GraphQueryArgs(_Args):
    op: GraphOp = Field(
        description="profile=实体档案（别名、所属、主要关系）；between=两个实体之间的直接关系；"
        "paths=两者之间不超过 3 跳的关系路径；neighbors=邻居（可按关系类型、对方类型筛选）；"
        "timeline=按阅读顺序的经历；title_holders=头衔的历任持有者（entity 填头衔实体）；"
        "mentions=实体被提及的段落位置"
    )
    entity: str = Field(description="实体 ID（来自 find_entity）")
    other: str | None = Field(None, description="另一个实体 ID（between / paths 必填）")
    relation_types: list[str] | None = Field(
        None, description=f"关系类型筛选（neighbors / paths）：{', '.join(_REL_NAMES)}"
    )
    other_types: list[str] | None = Field(
        None, description=f"对方实体类型筛选（neighbors）：{', '.join(_ENTITY_TYPES)}"
    )
    max_hops: int | None = Field(None, ge=1, le=3, description="paths 的最大跳数，默认 3")
    limit: int | None = Field(None, ge=1, le=30, description="返回条数上限")


GRAPH_QUERY_DESC = (
    "图谱查询：在知识图谱上做固定操作（人物档案、两者关系、关系路径、邻居、时间线、头衔持有者、提及位置）。"
    "什么时候用：人物关系、亲属、效忠、谁杀了谁、某头衔是谁，以及多跳问题里找桥接实体。"
    "每条关系带证据条数和证据 chunk_id；图谱由模型自动抽取，可能有错，单条证据的关系务必用 read_text 回原文核实。"
)


def graph_query(args: GraphQueryArgs, ctx: ToolContext, b: ToolBackends) -> ToolResult:
    from novel_rag.config import get_settings

    cfg = get_settings().tools
    g = _graph(b)
    ro = ctx.max_reading_order
    lim = args.limit or cfg.graph_default_limit
    if args.op in ("between", "paths") and not args.other:
        raise ToolInputError(f"{args.op} 需要 other（另一个实体 ID）")
    if args.other_types and (bad := [t for t in args.other_types if t not in _ENTITY_TYPES]):
        raise ToolInputError(f"未知的实体类型 {bad}；可选：{', '.join(_ENTITY_TYPES)}")
    scope = ctx.graph_entity_types
    other_types = args.other_types
    if scope:
        other_types = [t for t in (args.other_types or scope) if t in scope]
        if not other_types:
            raise ToolNotFound(f"[{args.op}] 没有找到（当前图谱只包含 {'、'.join(scope)} 类实体）")
    try:
        _in_scope(g, args.entity, ctx)
        if args.other:
            _in_scope(g, args.other, ctx)
        if args.op == "profile":
            res = g.profile(args.entity, ro, limit=lim)
        elif args.op == "between":
            res = g.between(args.entity, args.other, ro, limit=lim)
        elif args.op == "paths":
            res = g.paths(
                args.entity, args.other, ro, max_hops=args.max_hops, types=args.relation_types,
                node_types=scope, limit=min(lim, 5),
            )  # fmt: skip
        elif args.op == "neighbors":
            res = g.neighbors(args.entity, ro, types=args.relation_types, other_types=other_types, limit=lim)
        elif args.op == "timeline":
            res = g.timeline(args.entity, ro, limit=lim)
        elif args.op == "title_holders":
            res = g.title_holders(args.entity, ro, limit=lim)
        else:
            res = g.mentions(args.entity, ro, limit=lim)
    except QueryError as e:
        raise ToolInputError(str(e)) from e
    if scope:
        res.rows = _scope_rows(args.op, res.rows, scope)
    if not res.rows:
        raise ToolNotFound(f"[{args.op}] 没有找到（实体 ID 不存在、超出阅读进度，或两者之间没有记录的关系）")
    text, cut = truncate(render(res, b.where_chunk, cfg.graph_max_chars), cfg.graph_max_chars)
    refs, ros = collect(res.rows)
    ros += [r for r in (b.chunk_ro(c) for c in refs) if r is not None]
    return ToolResult(
        "graph_query", True, text, ref_chunk_ids=refs, reading_orders=ros, truncated=cut or res.truncated,
        meta={"op": args.op, "rows": len(res.rows)},
    )  # fmt: skip
