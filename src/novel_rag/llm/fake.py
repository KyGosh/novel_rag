"""单测用的 Fake 实现：确定性、无网络、无 GPU。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from novel_rag.llm.base import (
    ChatModel,
    ChatResult,
    Embedder,
    Message,
    Reranker,
    TokenUsage,
    VisionModel,
    l2_normalize,
)
from novel_rag.llm.structured import parse_structured


class FakeChatModel(ChatModel):
    """按 responder(messages) 返回固定文本；传 schema 时按结构化输出解析。记录调用次数便于断言缓存。"""

    def __init__(
        self,
        responder: Callable[[Sequence[Message]], str] | None = None,
        *,
        task: str = "fake",
        model_name: str = "fake-chat",
    ):
        self.responder = responder or (lambda msgs: f"echo:{msgs[-1]['content']}")
        self.task = task
        self.model_name = model_name
        self.calls = 0

    def chat(self, messages, *, schema=None, prompt_version=None) -> ChatResult:
        self.calls += 1
        text = self.responder(messages)
        parsed: BaseModel | None = parse_structured(text, schema) if schema is not None else None
        usage = TokenUsage(sum(len(str(m.get("content", ""))) for m in messages), len(text))
        return ChatResult(text=text, model=self.model_name, usage=usage, parsed=parsed)

    async def achat(self, messages, *, schema=None, prompt_version=None) -> ChatResult:
        return self.chat(messages, schema=schema, prompt_version=prompt_version)


class FakeEmbedder(Embedder):
    """由文本 hash 生成的确定性单位向量：相同文本得到相同向量。"""

    def __init__(self, dim: int = 8, model_name: str = "fake-embed"):
        self.dim = dim
        self.model_name = model_name

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
        return np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._vec(text))


class FakeReranker(Reranker):
    """分数 = 查询与候选的共同字符数占查询字符数的比例。"""

    model_name = "fake-rerank"

    def score(self, query: str, candidates: Sequence[str]) -> list[float]:
        q = set(query)
        return [len(q & set(c)) / max(len(q), 1) for c in candidates]


class FakeVisionModel(VisionModel):
    model_name = "fake-vision"

    def describe(self, image: bytes | Path, prompt: str) -> str:
        return "fake image description"
