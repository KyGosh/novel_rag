"""llm/ 抽象层的接口与数据类型。业务代码只依赖这里的类型，不直接 import 任何厂商 SDK。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel

Message = dict[str, Any]  # OpenAI 风格：{"role": "system" | "user" | "assistant", "content": "..."}
SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(self.prompt_tokens + other.prompt_tokens, self.completion_tokens + other.completion_tokens)


@dataclass(frozen=True)
class ChatResult:
    text: str  # 最终回答（思考类模型的推理内容已丢弃）
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_s: float = 0.0
    parsed: BaseModel | None = None  # 传入 schema 时为通过校验的对象
    cached: bool = False
    attempts: int = 1  # 结构化输出校验失败重试时 > 1


class StructuredOutputError(RuntimeError):
    """多次请求后模型输出仍无法通过 pydantic 校验。"""

    def __init__(self, message: str, last_text: str):
        super().__init__(message)
        self.last_text = last_text


class ChatModel(ABC):
    """对话模型。一个实例绑定一个任务（抽取 / 生成 / judge ……）及其模型参数。"""

    task: str
    model_name: str

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        *,
        schema: type[BaseModel] | None = None,
        prompt_version: str | None = None,
    ) -> ChatResult: ...

    @abstractmethod
    async def achat(
        self,
        messages: Sequence[Message],
        *,
        schema: type[BaseModel] | None = None,
        prompt_version: str | None = None,
    ) -> ChatResult: ...

    def cache_identity(self) -> dict[str, Any]:
        """决定输出的全部请求参数（模型名、温度等），参与缓存键计算。"""
        return {"model": self.model_name}


class Embedder(ABC):
    """文本向量化。文档与查询分开编码，便于以后换成需要查询前缀的模型。输出 L2 归一化的 float32 向量。"""

    model_name: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """返回形状 (len(texts), dim)。"""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """返回形状 (dim,)。"""


class Reranker(ABC):
    model_name: str

    @abstractmethod
    def score(self, query: str, candidates: Sequence[str]) -> list[float]:
        """返回与 candidates 一一对应的相关性分数，越大越相关。"""


class VisionModel(ABC):
    """插图理解（P5 再实现，届时选 Kimi 视觉模型或本地小模型）。"""

    model_name: str

    @abstractmethod
    def describe(self, image: bytes | Path, prompt: str) -> str: ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)
