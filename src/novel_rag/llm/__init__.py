"""LLM / Embedding / Reranker / 视觉模型的统一抽象层。业务代码只从这里 import。"""

from novel_rag.llm.base import (
    ChatModel,
    ChatResult,
    Embedder,
    Message,
    Reranker,
    StructuredOutputError,
    TokenUsage,
    VisionModel,
)
from novel_rag.llm.factory import get_chat_model, get_embedder, get_reranker, get_vision_model
from novel_rag.llm.usage import UsageRecord, UsageTracker, get_usage_tracker

__all__ = [
    "ChatModel",
    "ChatResult",
    "Embedder",
    "Message",
    "Reranker",
    "StructuredOutputError",
    "TokenUsage",
    "UsageRecord",
    "UsageTracker",
    "VisionModel",
    "get_chat_model",
    "get_embedder",
    "get_reranker",
    "get_usage_tracker",
    "get_vision_model",
]
