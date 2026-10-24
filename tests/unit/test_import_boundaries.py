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
