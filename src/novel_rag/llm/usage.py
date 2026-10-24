"""LLM 用量记录：每次调用记下 token 数和耗时，评测时按任务 / 模型汇总。"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class UsageRecord:
    task: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool
    attempts: int = 1


class UsageTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[UsageRecord] = []

    def record(self, rec: UsageRecord) -> None:
        with self._lock:
            self._records.append(rec)

    @property
    def records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    def summary(self) -> dict[str, Any]:
        """按 (task, model) 汇总。缓存命中计入调用次数，但不计 token（没有实际花费）。"""
        groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {
                "calls": 0,
                "cache_hits": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_s": 0.0,
                "retries": 0,
            }
        )
        for r in self.records:
            g = groups[(r.task, r.model)]
            g["calls"] += 1
            g["latency_s"] += r.latency_s
            if r.cached:
                g["cache_hits"] += 1
                continue
            g["prompt_tokens"] += r.prompt_tokens
            g["completion_tokens"] += r.completion_tokens
            g["retries"] += r.attempts - 1
        return {f"{task}/{model}": v for (task, model), v in groups.items()}

    def dump(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.records]


_tracker = UsageTracker()


def get_usage_tracker() -> UsageTracker:
    return _tracker
