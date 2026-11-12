"""硬性约束：厂商 SDK 只能在 llm/ 中 import，业务代码不能直接依赖。"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "novel_rag"
VENDOR = re.compile(r"^\s*(?:import|from)\s+(openai|FlagEmbedding|torch|transformers|sentence_transformers)\b", re.M)


def test_vendor_sdks_only_imported_inside_llm():
    offenders = [
        str(p.relative_to(SRC))
        for p in SRC.rglob("*.py")
        if "llm" not in p.relative_to(SRC).parts and VENDOR.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


LANGGRAPH = re.compile(r"^\s*(?:import|from)\s+(langgraph|langchain\w*)\b", re.M)


def test_langgraph_only_imported_inside_agent():
    """P4：LangGraph 只在 agent/ 中 import；不用 LangChain 的模型封装（模型调用一律走 llm/）。"""
    offenders = []
    for p in SRC.rglob("*.py"):
        m = LANGGRAPH.search(p.read_text(encoding="utf-8"))
        if m and (m.group(1).startswith("langchain") or "agent" not in p.relative_to(SRC).parts):
            offenders.append(str(p.relative_to(SRC)))
    assert offenders == []


def test_tools_do_not_depend_on_agent():
    """工具层独立于编排框架：tools/ 与 qa/ 不 import agent/。"""
    pat = re.compile(r"^\s*(?:import|from)\s+novel_rag\.agent\b", re.M)
    offenders = [
        str(p.relative_to(SRC))
        for sub in ("tools", "qa")
        for p in (SRC / sub).rglob("*.py")
        if pat.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []
