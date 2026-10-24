"""按配置创建模型实例。业务代码只调用这里的工厂函数，测试中可以直接构造 Fake 实现注入。"""

from __future__ import annotations

from functools import cache, lru_cache

from novel_rag.config import Settings, get_settings
from novel_rag.llm.base import ChatModel, Embedder, Reranker, VisionModel
from novel_rag.llm.usage import get_usage_tracker


@lru_cache(maxsize=1)
def _limiter():
    from novel_rag.llm.openai_chat import ConcurrencyLimiter

    return ConcurrencyLimiter(get_settings().llm.max_concurrency)


@lru_cache(maxsize=1)
def _llm_cache_store():
    from novel_rag.llm.cache import open_disk_cache

    return open_disk_cache(get_settings().paths.llm_cache_dir)


@cache
def get_chat_model(task: str) -> ChatModel:
    """task 对应 config 中 llm.tasks 的键，例如 extraction / generation / judge。"""
    from novel_rag.llm.cache import CachedChatModel
    from novel_rag.llm.openai_chat import OpenAICompatChatModel

    settings: Settings = get_settings()
    if task not in settings.llm.tasks:
        raise KeyError(f"config 中没有 LLM 任务 {task!r}，可选：{sorted(settings.llm.tasks)}")
    tracker = get_usage_tracker()
    model: ChatModel = OpenAICompatChatModel(task, settings.llm.tasks[task], settings.llm, tracker, _limiter())
    if settings.llm.cache_enabled:
        model = CachedChatModel(model, _llm_cache_store(), tracker)
    return model


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    from novel_rag.llm.bge import BgeM3Embedder

    s = get_settings()
    return BgeM3Embedder(s.embedding, load_from=s.paths.resolve_model(s.embedding.model_name))


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    from novel_rag.llm.bge import BgeReranker

    s = get_settings()
    return BgeReranker(s.reranker, load_from=s.paths.resolve_model(s.reranker.model_name))


def get_vision_model() -> VisionModel:
    raise NotImplementedError("VisionModel 在 P5 插图理解时再实现")
