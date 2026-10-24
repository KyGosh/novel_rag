"""OpenAI 兼容接口的对话模型（Kimi / Moonshot 走这里）。整个项目只有本文件 import openai。

横切能力：429 / 5xx / 网络错误指数退避重试；全局并发上限；结构化输出校验失败重试；
思考类模型只取 content，reasoning_content 丢弃；每次调用记录用量。
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Sequence
from typing import Any

import openai
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from novel_rag.config import LLMConfig, LLMTaskConfig
from novel_rag.llm.base import ChatModel, ChatResult, Message, StructuredOutputError, TokenUsage
from novel_rag.llm.structured import (
    StructuredMode,
    parse_structured,
    response_format_for,
    validation_feedback,
    with_schema_instruction,
)
from novel_rag.llm.usage import UsageRecord, UsageTracker


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, openai.RateLimitError | openai.APIConnectionError):  # 后者包含超时
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


class ConcurrencyLimiter:
    """同一账户下所有任务共享的并发上限，同步和异步调用各自计数。"""

    def __init__(self, limit: int):
        self.limit = limit
        self._sync = threading.BoundedSemaphore(limit)
        self._async: dict[int, asyncio.Semaphore] = {}

    @property
    def sync(self) -> threading.BoundedSemaphore:
        return self._sync

    def async_sem(self) -> asyncio.Semaphore:
        loop_id = id(asyncio.get_running_loop())
        if loop_id not in self._async:
            self._async[loop_id] = asyncio.Semaphore(self.limit)
        return self._async[loop_id]


class OpenAICompatChatModel(ChatModel):
    def __init__(
        self,
        task: str,
        task_cfg: LLMTaskConfig,
        llm_cfg: LLMConfig,
        tracker: UsageTracker,
        limiter: ConcurrencyLimiter,
        *,
        client: Any | None = None,
        async_client: Any | None = None,
    ):
        self.task = task
        self.model_name = task_cfg.model
        self.task_cfg = task_cfg
        self.structured_mode: StructuredMode = llm_cfg.structured_mode
        self.structured_max_attempts = llm_cfg.structured_max_attempts
        self.max_retries = llm_cfg.max_retries
        self.retry_wait = wait_exponential_jitter(initial=1, max=30)  # 单测里替换为 wait_none()
        self.tracker = tracker
        self.limiter = limiter
        client_kwargs = {
            "api_key": llm_cfg.api_key.get_secret_value() or "missing-key",
            "base_url": llm_cfg.base_url,
            "timeout": llm_cfg.timeout,
            "max_retries": 0,  # 重试由 tenacity 统一负责
        }
        self._client = client or openai.OpenAI(**client_kwargs)
        self._async_client = async_client or openai.AsyncOpenAI(**client_kwargs)

    def cache_identity(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "temperature": self.task_cfg.temperature,
            "max_tokens": self.task_cfg.max_tokens,
            "extra_body": self.task_cfg.extra_body,
            "structured_mode": self.structured_mode,
        }

    # ---- 请求构造与响应解析 ----

    def _request_kwargs(self, messages: Sequence[Message], schema: type[BaseModel] | None) -> dict[str, Any]:
        msgs = list(messages)
        kwargs: dict[str, Any] = {"model": self.model_name}
        if schema is not None:
            kwargs["response_format"] = response_format_for(schema, self.structured_mode)
            if self.structured_mode == "json_object":
                msgs = with_schema_instruction(msgs, schema)
        kwargs["messages"] = msgs
        if self.task_cfg.temperature is not None:
            kwargs["temperature"] = self.task_cfg.temperature
        if self.task_cfg.max_tokens is not None:
            kwargs["max_tokens"] = self.task_cfg.max_tokens
        if self.task_cfg.extra_body:
            kwargs["extra_body"] = self.task_cfg.extra_body
        return kwargs

    @staticmethod
    def _extract(resp: Any) -> tuple[str, TokenUsage, str]:
        # 只取最终回答；思考模型的 reasoning_content 不读取、不返回
        text = resp.choices[0].message.content or ""
        u = resp.usage
        usage = TokenUsage(u.prompt_tokens or 0, u.completion_tokens or 0) if u is not None else TokenUsage()
        return text, usage, resp.model or ""

    def _retrying(self) -> dict[str, Any]:
        return {
            "stop": stop_after_attempt(self.max_retries + 1),
            "wait": self.retry_wait,
            "retry": retry_if_exception(is_retryable),
            "reraise": True,
        }

    def _finish(
        self, text: str, usage: TokenUsage, model: str, t0: float, parsed: BaseModel | None, attempts: int
    ) -> ChatResult:
        result = ChatResult(
            text=text,
            model=model or self.model_name,
            usage=usage,
            latency_s=time.perf_counter() - t0,
            parsed=parsed,
            attempts=attempts,
        )
        self.tracker.record(
            UsageRecord(
                self.task,
                result.model,
                usage.prompt_tokens,
                usage.completion_tokens,
                result.latency_s,
                cached=False,
                attempts=attempts,
            )
        )
        return result

    # ---- 同步 ----

    def _call(self, kwargs: dict[str, Any]) -> Any:
        for attempt in Retrying(**self._retrying()):
            with attempt, self.limiter.sync:
                return self._client.chat.completions.create(**kwargs)

    def chat(self, messages, *, schema=None, prompt_version=None) -> ChatResult:
        t0 = time.perf_counter()
        msgs = list(messages)
        total = TokenUsage()
        attempts = self.structured_max_attempts if schema is not None else 1
        text, model = "", ""
        for i in range(1, attempts + 1):
            text, usage, model = self._extract(self._call(self._request_kwargs(msgs, schema)))
            total += usage
            if schema is None:
                return self._finish(text, total, model, t0, None, i)
            try:
                parsed = parse_structured(text, schema)
            except ValidationError as e:
                msgs = list(messages) + validation_feedback(text, e)
                continue
            return self._finish(text, total, model, t0, parsed, i)
        self._finish(text, total, model, t0, None, attempts)
        raise StructuredOutputError(f"{attempts} 次请求后输出仍不符合 {schema.__name__}", text)

    # ---- 异步 ----

    async def _acall(self, kwargs: dict[str, Any]) -> Any:
        async for attempt in AsyncRetrying(**self._retrying()):
            with attempt:
                async with self.limiter.async_sem():
                    return await self._async_client.chat.completions.create(**kwargs)

    async def achat(self, messages, *, schema=None, prompt_version=None) -> ChatResult:
        t0 = time.perf_counter()
        msgs = list(messages)
        total = TokenUsage()
        attempts = self.structured_max_attempts if schema is not None else 1
        text, model = "", ""
        for i in range(1, attempts + 1):
            text, usage, model = self._extract(await self._acall(self._request_kwargs(msgs, schema)))
            total += usage
            if schema is None:
                return self._finish(text, total, model, t0, None, i)
            try:
                parsed = parse_structured(text, schema)
            except ValidationError as e:
                msgs = list(messages) + validation_feedback(text, e)
                continue
            return self._finish(text, total, model, t0, parsed, i)
        self._finish(text, total, model, t0, None, attempts)
        raise StructuredOutputError(f"{attempts} 次请求后输出仍不符合 {schema.__name__}", text)
