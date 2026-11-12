"""P4 工具层：与编排框架无关的纯函数工具。Agent（LangGraph）、固定流程 RAG 与评测脚本共用。"""

from novel_rag.tools.backends import ToolBackends, build_backends
from novel_rag.tools.base import SpoilerLeak, Tool, ToolContext, ToolResult
from novel_rag.tools.registry import Toolbox, default_tools

__all__ = [
    "SpoilerLeak",
    "Tool",
    "ToolBackends",
    "ToolContext",
    "ToolResult",
    "Toolbox",
    "build_backends",
    "default_tools",
]
