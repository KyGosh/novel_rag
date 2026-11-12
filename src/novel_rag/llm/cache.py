"""LLM 调用结果缓存（包装器）。

缓存键 = hash(模型及请求参数, 消息, schema, prompt 版本[, 工具 schema][, 采样编号])。只缓存成功（含结构化校验通过）的
结果，所以 P3 全量抽取中断后重跑，已完成的部分直接命中缓存，相当于断点续跑。

工具 schema 与采样编号（P4）只在非默认值时才写进键：不带工具、sample=0 的调用与旧缓存的键完全一致，
P3 的抽取 / 摘要缓存不失效。消息历史本身（含 assistant 的 tool_calls 与 tool 结果消息）整体参与键计算。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import MutableMapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from novel_rag.llm.base import ChatModel, ChatResult, Message, TokenUsage, ToolCall, ToolSchema
from novel_rag.llm.usage import UsageRecord, UsageTracker

CACHE_FORMAT_VERSION = 1


def cache_key(
    identity: dict[str, Any],
    messages: Sequence[Message],
    schema: type[BaseModel] | None,
    prompt_version: str | None,
    tools: Sequence[ToolSchema] | None = None,
    sample: int = 0,
) -> str:
    payload: dict[str, Any] = {
        "v": CACHE_FORMAT_VERSION,
        "identity": identity,
        "messages": list(messages),
        "schema": schema.model_json_schema() if schema is not None else None,
        "prompt_version": prompt_version,
    }
    if tools:
        payload["tools"] = list(tools)
    if sample:
        payload["sample"] = int(sample)
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def open_disk_cache(directory: Path) -> MutableMapping[str, Any]:
    import diskcache

    directory.mkdir(parents=True, exist_ok=True)
    return diskcache.Cache(str(directory))


class CachedChatModel(ChatModel):
    def __init__(self, inner: ChatModel, store: MutableMapping[str, Any], tracker: UsageTracker):
        self.inner = inner
        self.store = store
        self.tracker = tracker
        self.task = inner.task
        self.model_name = inner.model_name

    def cache_identity(self) -> dict[str, Any]:
        return self.inner.cache_identity()

    def _lookup(
        self,
        messages: Sequence[Message],
        schema: type[BaseModel] | None,
        prompt_version: str | None,
        tools: Sequence[ToolSchema] | None = None,
        sample: int = 0,
    ) -> tuple[str, ChatResult | None]:
        key = cache_key(self.cache_identity(), messages, schema, prompt_version, tools, sample)
        t0 = time.perf_counter()
        hit = self.store.get(key)
        if hit is None:
            return key, None
        parsed = schema.model_validate(hit["parsed"]) if schema is not None and hit["parsed"] is not None else None
        calls = tuple(ToolCall.parse(c["id"], c["name"], c["arguments"]) for c in hit.get("tool_calls", []))
        result = ChatResult(
            text=hit["text"],
            model=hit["model"],
            usage=TokenUsage(**hit["usage"]),
            latency_s=time.perf_counter() - t0,
            parsed=parsed,
            cached=True,
            attempts=hit.get("attempts", 1),
            tool_calls=calls,
        )
        self.tracker.record(UsageRecord(self.task, result.model, 0, 0, result.latency_s, cached=True, attempts=1))
        return key, result

    def _save(self, key: str, result: ChatResult) -> None:
        self.store[key] = {
            "text": result.text,
            "model": result.model,
            "usage": asdict(result.usage),
            "parsed": result.parsed.model_dump(mode="json") if result.parsed is not None else None,
            "attempts": result.attempts,
            **({"tool_calls": [c.to_dict() for c in result.tool_calls]} if result.tool_calls else {}),
        }

    def chat(self, messages, *, schema=None, prompt_version=None, tools=None, sample=0) -> ChatResult:
        key, hit = self._lookup(messages, schema, prompt_version, tools, sample)
        if hit is not None:
            return hit
        result = self.inner.chat(messages, schema=schema, prompt_version=prompt_version, tools=tools, sample=sample)
        self._save(key, result)
        return result

    async def achat(self, messages, *, schema=None, prompt_version=None, tools=None, sample=0) -> ChatResult:
        key, hit = self._lookup(messages, schema, prompt_version, tools, sample)
        if hit is not None:
            return hit
        result = await self.inner.achat(
            messages, schema=schema, prompt_version=prompt_version, tools=tools, sample=sample
        )
        self._save(key, result)
        return result
