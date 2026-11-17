"""本地 Web 演示（P6）：FastAPI 提供问答接口，用 SSE 把每个环节的事件实时推给页面；页面是不需要构建的静态单页。

范围限定：只监听本机（scripts/serve_demo.py 绑定 127.0.0.1），单用户，同一时刻只处理一个问题，不做鉴权。
问答在线程里运行（模型调用和 GPU 检索都是同步的），事件经 asyncio 队列转成 SSE。

接口：
    GET /                静态页面
    GET /api/meta        模式、各卷与章节（阅读进度选择用）
    GET /api/ask?q=问题&mode=agent&volume=asoiaf-1&chapter=30
                         SSE：event: step（过程事件，见 agent/loop.py、qa/route.py）……
                              event: result（最终结果，见 QAService.ask）或 event: error
不返回整段原文：过程事件里只有编号、"卷名 · 章节标题"、工具结果概况和证据笔记的短引文。
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from novel_rag.api.service import MODE_NAMES, MODES, QAService

STATIC = Path(__file__).resolve().parent / "static"
_DONE = object()


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(get_service: Callable[[], QAService]) -> FastAPI:
    """get_service：返回问答服务（第一次调用时才组装真实环境；测试注入假服务）。"""
    app = FastAPI(title="novel_rag 演示", docs_url=None, redoc_url=None, openapi_url=None)
    busy = threading.Lock()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", media_type="text/html; charset=utf-8")

    @app.get("/api/meta")
    def meta() -> JSONResponse:
        svc = get_service()
        return JSONResponse({"modes": [{"id": m, "name": MODE_NAMES[m]} for m in MODES], "default_mode": "agent",
                             "volumes": svc.volumes()})  # fmt: skip

    @app.get("/api/ask")
    async def ask(
        q: str = Query(..., max_length=200),
        mode: str = Query("agent"),
        volume: str | None = Query(None),
        chapter: int | None = Query(None),
    ) -> StreamingResponse:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def put(kind: str, data: Any) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (kind, data))

        def work() -> None:
            try:
                svc = get_service()
                max_ro = svc.reading_order_of(volume, chapter) if volume else None
                put("step", {"type": "start", "mode": mode, "progress": svc.describe_progress(max_ro)})
                res = svc.ask(q, mode, max_ro, on_event=lambda e: put("step", e))
                put("result", res)
            except ValueError as e:
                put("error", {"message": str(e)})
            except Exception as e:  # noqa: BLE001 —— 错误交给页面显示，不让连接挂住
                put("error", {"message": f"{type(e).__name__}: {str(e)[:300]}"})
            finally:
                busy.release()
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        async def stream() -> AsyncIterator[str]:
            if not busy.acquire(blocking=False):
                yield sse("error", {"message": "正在回答上一个问题，请稍后再问（演示只支持单用户）"})
                return
            threading.Thread(target=work, daemon=True).start()
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                kind, data = item
                yield sse(kind, data)

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)

    return app
