"""批量 LLM 任务的并发执行：逐个提交、限制在途数量、单个任务失败不影响其他任务，并定期打印进度。

章节摘要、图谱抽取、消歧判定共用。LLM 调用本身的并发上限、重试和缓存在 llm/ 中；这里只控制同时在途的任务数，
避免一次创建数千个协程，同时把异常收集为结果，让流水线不被个别失败阻塞。
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class JobResult(Generic[T, R]):
    job: T
    value: R | None
    error: BaseException | None

    @property
    def ok(self) -> bool:
        return self.error is None


async def run_jobs(
    jobs: Sequence[T],
    fn: Callable[[T], Awaitable[R]],
    *,
    concurrency: int,
    label: str = "jobs",
    progress_every: float = 30.0,
) -> list[JobResult[T, R]]:
    """按输入顺序返回结果。异常被捕获为 JobResult.error（KeyboardInterrupt 等仍会向上抛）。"""
    sem = asyncio.Semaphore(max(1, concurrency))
    results: list[JobResult[T, R] | None] = [None] * len(jobs)
    done = failed = 0
    t0 = last = time.monotonic()

    async def one(i: int, job: T) -> None:
        nonlocal done, failed, last
        async with sem:
            try:
                results[i] = JobResult(job, await fn(job), None)
            except Exception as e:  # noqa: BLE001 —— 单个任务失败只记录
                results[i] = JobResult(job, None, e)
                failed += 1
        done += 1
        now = time.monotonic()
        if now - last >= progress_every or done == len(jobs):
            last = now
            msg = f"[{label}] {done}/{len(jobs)} 完成，失败 {failed}，用时 {now - t0:.0f}s"
            print(msg, file=sys.stderr, flush=True)

    await asyncio.gather(*(one(i, j) for i, j in enumerate(jobs)))
    return [r for r in results if r is not None]
