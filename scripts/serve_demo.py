"""启动本地 Web 演示（P6）：只监听本机，单用户。原文受版权保护，演示只在本地运行，不要对外暴露端口。

用法：
    uv run python scripts/serve_demo.py              # http://127.0.0.1:17860
    uv run python scripts/serve_demo.py --port 17861
需要先启动容器（docker compose up -d）和 LLM 代理；GPU 模型在第一次提问时加载（约半分钟）。
端口选在 15000 以上：本机 Windows 的动态端口段（1024–15000）可能被 WinNAT 随机保留。
"""

from __future__ import annotations

import argparse
import sys
import threading


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=17860)
    ap.add_argument("--warm", action="store_true", help="启动时就组装环境、加载 GPU 模型")
    args = ap.parse_args()

    import uvicorn

    from novel_rag.api.app import create_app
    from novel_rag.api.service import QAService
    from novel_rag.config import get_settings

    lock = threading.Lock()
    holder: dict[str, QAService] = {}

    def get_service() -> QAService:
        with lock:
            if "svc" not in holder:
                holder["svc"] = QAService.build(get_settings())
            return holder["svc"]

    if args.warm:
        get_service().system("agent")
    print(f"演示地址：http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(create_app(get_service), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
