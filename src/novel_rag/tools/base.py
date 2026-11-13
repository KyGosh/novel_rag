"""工具层的公共类型：工具定义、运行环境（注入）、结构化结果与错误分类。

工具是纯函数 `fn(args, ctx, backends) -> ToolResult`：
- args：经 pydantic 校验的模型可见参数；
- ctx：运行环境注入的阅读进度与检索范围，**不出现在模型可见的参数 schema 里**，模型无法改写；
- backends：检索器、图谱、摘要等后端（测试时注入假后端）。
LangGraph Agent、固定流程 RAG 和评测脚本都通过同一个 Toolbox 调用同一套工具。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from novel_rag.llm.base import ToolSchema

if TYPE_CHECKING:
    from novel_rag.tools.backends import ToolBackends


@dataclass(frozen=True)
class ToolContext:
    """运行环境：每题由评测 / 调用方设置，工具内部据此过滤。"""

    max_reading_order: int | None
    book_ids: tuple[str, ...]
    graph_entity_types: tuple[str, ...] | None = None  # 图谱范围：只暴露这些类型的实体（None = 全量）


@dataclass
class ToolResult:
    """一次工具调用的结果。text 是模型（反思环节）看到的观察；其余字段供循环、trace 和评测使用。"""

    tool: str
    ok: bool
    text: str
    shown_chunk_ids: list[str] = field(default_factory=list)  # 展示了原文的 chunk（过程覆盖率按这个算）
    ref_chunk_ids: list[str] = field(default_factory=list)  # 只给出编号的 chunk（图谱 / 摘要的证据）
    reading_orders: list[int] = field(default_factory=list)  # 观察中出现的全部位置（防剧透断言用）
    error: str | None = None
    error_kind: str | None = None  # invalid_args / not_found / unknown_tool / backend / transient / limit
    truncated: bool = False
    attempts: int = 1
    latency_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)  # 不含原文的结构化信息（检索延迟、实体 ID 等）

    @property
    def chunk_ids(self) -> list[str]:
        """观察中出现的全部 chunk（展示原文的在前）。反思环节的笔记只能引用这些。"""
        return list(dict.fromkeys([*self.shown_chunk_ids, *self.ref_chunk_ids]))

    def summary(self) -> dict[str, Any]:
        """不含原文的结构化摘要（写进 eval/results 的 trace 摘要）。"""
        return {
            "tool": self.tool,
            "ok": self.ok,
            "error_kind": self.error_kind,
            "shown": self.shown_chunk_ids,
            "refs": self.ref_chunk_ids,
            "chars": len(self.text),
            "truncated": self.truncated,
            "attempts": self.attempts,
            "latency_ms": round(self.latency_ms, 1),
            **({"meta": self.meta} if self.meta else {}),
        }


class ToolInputError(ValueError):
    """参数不合法（格式、范围）：不重试，作为结构化错误返回给模型。"""

    kind = "invalid_args"


class ToolNotFound(ToolInputError):
    """要找的东西不存在（或超出阅读进度，不区分以免泄露）：不重试。"""

    kind = "not_found"


class TransientToolError(RuntimeError):
    """服务超时 / 连接失败：按次数上限重试。"""


class SpoilerLeak(RuntimeError):
    """防剧透兜底断言失败：观察中出现了超过阅读进度的内容。严重错误，该题记为失败。"""


_TRANSIENT_NAMES = ("timeout", "connection", "unavailable", "transient", "sessionexpired")


def is_transient(exc: BaseException) -> bool:
    """超时、连接失败类错误（跨 ES / Milvus / Neo4j 客户端按类名判断，避免在这里 import 各家 SDK）。"""
    if isinstance(exc, TransientToolError | TimeoutError | ConnectionError):
        return True
    names = [c.__name__.lower() for c in type(exc).__mro__]
    return any(t in n for n in names for t in _TRANSIENT_NAMES)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str  # 写清"什么时候用"
    args_model: type[BaseModel]
    fn: Callable[[Any, ToolContext, ToolBackends], ToolResult]

    def schema(self) -> ToolSchema:
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        for p in params.get("properties", {}).values():
            p.pop("title", None)
        return {"name": self.name, "description": self.description, "parameters": params}


def truncate(text: str, max_chars: int, note: str = "……（输出过长，已截断）") -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[: max(0, max_chars - len(note))] + note, True
