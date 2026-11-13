"""工具箱：按名字执行工具，统一处理参数校验、重试、出错返回、长度上限与防剧透断言。

- 工具名不存在、参数不是合法 JSON / 不符合 schema：不调用，返回结构化错误（error_kind 区分）。
- 服务超时 / 连接失败：按次数上限重试（指数退避），仍失败返回 transient 错误。
- 参数错误、找不到：不重试，直接返回错误说明。
- 无论哪种情况都不向外抛异常——**防剧透断言除外**：观察中出现超过阅读进度的 chunk / 位置时抛 SpoilerLeak，
  由调用方把该题记为严重失败。这是兜底，正常情况下各后端自己就会过滤。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from novel_rag.llm.base import ToolSchema
from novel_rag.tools.backends import ToolBackends
from novel_rag.tools.base import (
    SpoilerLeak,
    Tool,
    ToolContext,
    ToolInputError,
    ToolResult,
    is_transient,
    truncate,
)


def _validation_message(e: ValidationError) -> str:
    parts = []
    for err in e.errors(include_url=False):
        loc = ".".join(str(x) for x in err["loc"]) or "参数"
        parts.append(f"{loc}：{err['msg']}")
    return "；".join(parts)


def default_tools() -> list[Tool]:
    from novel_rag.tools import graph, summary, text

    return [
        Tool("search_text", text.SEARCH_TEXT_DESC, text.SearchTextArgs, text.search_text),
        Tool("read_text", text.READ_TEXT_DESC, text.ReadTextArgs, text.read_text),
        Tool("search_summaries", summary.SEARCH_SUMMARIES_DESC, summary.SearchSummariesArgs, summary.search_summaries),
        Tool("read_summary", summary.READ_SUMMARY_DESC, summary.ReadSummaryArgs, summary.read_summary),
        Tool("find_entity", graph.FIND_ENTITY_DESC, graph.FindEntityArgs, graph.find_entity),
        Tool("graph_query", graph.GRAPH_QUERY_DESC, graph.GraphQueryArgs, graph.graph_query),
    ]


class Toolbox:
    def __init__(
        self,
        backends: ToolBackends,
        tools: Sequence[Tool] | None = None,
        *,
        max_retries: int | None = None,
        retry_backoff_s: float | None = None,
        hard_max_chars: int = 6000,
        sleep=time.sleep,  # noqa: ANN001 —— 单测替换
    ):
        from novel_rag.config import get_settings

        cfg = get_settings().tools
        self.backends = backends
        self.tools = {t.name: t for t in (tools if tools is not None else default_tools())}
        self.max_retries = cfg.max_retries if max_retries is None else max_retries
        self.retry_backoff_s = cfg.retry_backoff_s if retry_backoff_s is None else retry_backoff_s
        self.hard_max_chars = hard_max_chars
        self._sleep = sleep

    def without(self, *names: str) -> Toolbox:
        """去掉某些工具（P5 消融：去掉图谱工具、去掉摘要）。"""
        tb = Toolbox.__new__(Toolbox)
        tb.__dict__.update(self.__dict__)
        tb.tools = {n: t for n, t in self.tools.items() if n not in names}
        return tb

    def without_graph(self) -> Toolbox:
        """去掉全部图谱（P5 消融）：去掉实体查找与图谱查询，原文检索也不再支持按实体限定。"""
        from novel_rag.tools import text

        tb = self.without("find_entity", "graph_query")
        if "search_text" in tb.tools:
            tb.tools = {
                n: Tool(n, text.SEARCH_TEXT_DESC_NO_ENTITY, text.SearchTextArgsNoEntity, text.search_text)
                if n == "search_text" else t
                for n, t in tb.tools.items()
            }  # fmt: skip
        return tb

    def schemas(self) -> list[ToolSchema]:
        return [t.schema() for t in self.tools.values()]

    def _error(self, name: str, kind: str, msg: str, attempts: int = 1, t0: float | None = None) -> ToolResult:
        lat = (time.perf_counter() - t0) * 1000 if t0 is not None else 0.0
        return ToolResult(name, False, f"[错误] {msg}", error=msg, error_kind=kind, attempts=attempts, latency_ms=lat)

    def call(
        self, name: str, args: dict[str, Any] | None, ctx: ToolContext, *, args_error: str | None = None
    ) -> ToolResult:
        t0 = time.perf_counter()
        tool = self.tools.get(name)
        if tool is None:
            return self._error(name, "unknown_tool", f"没有工具 {name!r}；可用：{', '.join(self.tools)}", t0=t0)
        if args_error is not None or args is None:
            return self._error(name, "invalid_args", args_error or "缺少参数", t0=t0)
        try:
            parsed = tool.args_model.model_validate(args)
        except ValidationError as e:
            return self._error(name, "invalid_args", f"参数不合法：{_validation_message(e)}", t0=t0)

        attempt = 0
        while True:
            attempt += 1
            try:
                result = tool.fn(parsed, ctx, self.backends)
                break
            except ToolInputError as e:
                return self._error(name, e.kind, str(e), attempt, t0)
            except SpoilerLeak:
                raise
            except Exception as e:  # noqa: BLE001 —— 工具失败不能让整题失败
                if is_transient(e) and attempt <= self.max_retries:
                    self._sleep(self.retry_backoff_s * (2 ** (attempt - 1)))
                    continue
                kind = "transient" if is_transient(e) else "backend"
                return self._error(name, kind, f"{type(e).__name__}: {str(e)[:200]}", attempt, t0)

        result.attempts = attempt
        result.latency_ms = (time.perf_counter() - t0) * 1000
        self.assert_no_spoiler(result, ctx)
        result.text, cut = truncate(result.text, self.hard_max_chars)
        result.truncated = result.truncated or cut
        return result

    def assert_no_spoiler(self, result: ToolResult, ctx: ToolContext) -> None:
        """兜底断言：观察中出现的每个 chunk、证据和位置都不超过阅读进度、不出检索范围。"""
        max_ro = ctx.max_reading_order
        bad: list[str] = []
        for cid in result.chunk_ids:
            try:
                c = self.backends.chunks.by_id(cid)
            except KeyError:
                continue  # 图谱证据里有、当前 chunk 索引里没有（不同分块版本）：没有原文，不构成泄露
            if c["book_id"] not in ctx.book_ids:
                bad.append(f"{cid}（不在检索范围）")
            elif max_ro is not None and int(c["reading_order"]) > max_ro:
                bad.append(f"{cid}（reading_order {c['reading_order']}）")
        if max_ro is not None:
            bad += [f"位置 {r}" for r in result.reading_orders if r > max_ro]
        if bad:
            raise SpoilerLeak(f"{result.tool} 的观察超出阅读进度 {max_ro}：{bad[:5]}")
