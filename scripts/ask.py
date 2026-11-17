"""命令行问答（P6 统一入口）：固定 RAG / Agent，可指定阅读进度，终端里逐步打印过程。

原文只在本地处理；终端里只显示证据笔记的短引文和"卷名 · 章节标题"形式的引用，不打印整段原文。

用法：
    uv run python scripts/ask.py "问题"                                   # 默认 Agent，不限阅读进度
    uv run python scripts/ask.py --mode rag "问题"                        # 固定 RAG
    uv run python scripts/ask.py --volume 1 --chapter 30 "问题"           # 读到第 1 卷第 30 章（防剧透）
    uv run python scripts/ask.py --volume asoiaf-2 "问题"                 # 读完 asoiaf-2 这一卷
    uv run python scripts/ask.py --list-chapters 1                        # 列出第 1 卷的章号与标题
    uv run python scripts/ask.py --json "问题"                            # 结果以 JSON 输出（不打印过程）
--volume 可以是卷 ID，也可以是从 1 开始的卷序号；--chapter 是章号（--list-chapters 可查）。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from novel_rag.api.service import MODE_NAMES, MODES


def _args(a: Any) -> str:
    if not isinstance(a, dict):
        return "（参数无效）"
    return ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in a.items())


def printer(e: dict[str, Any]) -> None:
    """把事件打印成一条过程时间线。"""
    t = e.get("type")
    if t == "plan":
        print(f"\n【规划】{e['plan']}", flush=True)
    elif t == "retrieve":
        print(f"\n【检索】{len(e['chunks'])} 段：" + "；".join(c["where"] for c in e["chunks"][:8]), flush=True)
    elif t == "route":
        verdict = f"升级到 Agent（原因：{e['reason']}）" if e["escalate"] else "证据充分，直接返回"
        print(f"\n【路由】{verdict}", flush=True)
        for p in e.get("parts") or []:
            print(f"    {'✓' if p['supported'] else '✗'} {p['part']}")
    elif t == "act":
        if e["calls"]:
            print(f"\n【第 {e['round']} 轮 行动】", flush=True)
            for c in e["calls"]:
                print(f"    → {c['tool']}({_args(c['args'])})")
        if e["finish"]:
            print(f"\n【第 {e['round']} 轮 行动】结束检索", flush=True)
    elif t == "tool":
        mark = "✓" if e["ok"] else "✗"
        print(f"    {mark} {e['tool']}：{e['overview'][:150]}（{e['latency_ms'] / 1000:.1f} s）", flush=True)
    elif t == "reflect":
        if e.get("error"):
            print(f"  【反思】跳过（{e['error']}）", flush=True)
            return
        print(f"  【反思】新增 {len(e['added'])} 条笔记" + ("，证据已充分" if e.get("sufficient") else ""), flush=True)
        for n in e["added"]:
            q = f"（“{n['quote']}”）" if n.get("quote") else ""
            print(f"    + {n['claim']}{q} —— {n['where']}")
        if e.get("status"):
            print(f"    现状：{e['status']}")
    elif t == "stop":
        print(f"\n【停止】触及上限：{e['termination']}", flush=True)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?")
    ap.add_argument("--mode", choices=MODES, default="agent", help="rag / agent，默认 agent")
    ap.add_argument("--volume", help="阅读进度：卷 ID 或从 1 开始的卷序号")
    ap.add_argument("--chapter", type=int, help="阅读进度：读到这一章（章号）；不给则读完 --volume 这一卷")
    ap.add_argument("--max-reading-order", type=int, help="直接给全局阅读顺序（与 --volume 二选一）")
    ap.add_argument("--list-chapters", metavar="VOLUME", help="列出某一卷的章号与标题")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    ap.add_argument("--sample", type=int, default=0, help="采样编号（参与 LLM 缓存键；换一个得到新的样本）")
    args = ap.parse_args()

    from novel_rag.api.service import QAService
    from novel_rag.config import get_settings

    svc = QAService.build(get_settings())
    if args.list_chapters:
        ro = svc.reading_order_of(args.list_chapters)
        vol = next(v for v in svc.volumes() if any(c["reading_order"] == ro for c in v["chapters"]))
        print(f"《{vol['title']}》（{vol['book_id']}）")
        for c in vol["chapters"]:
            print(f"  {c['idx']:>3}  {c['title']}")
        return 0
    if not args.question:
        ap.error("需要一个问题")
    if args.chapter is not None and not args.volume:
        ap.error("--chapter 需要和 --volume 一起用")
    max_ro = args.max_reading_order
    if args.volume:
        max_ro = svc.reading_order_of(args.volume, args.chapter)

    if not args.json:
        print(f"问题：{args.question}", flush=True)
        print(f"模式：{MODE_NAMES[args.mode]}；阅读进度：{svc.describe_progress(max_ro)}", flush=True)
    res = svc.ask(args.question, args.mode, max_ro, on_event=None if args.json else printer, sample=args.sample)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0
    print("\n" + "=" * 60)
    print(f"答案：{res['answer']}")
    if res["rationale"]:
        print(f"依据：{res['rationale']}")
    for c in res["citations"]:
        print(f"  引用 [{c['chunk_id']}] {c['where']}")
    cost, lat = res["cost"], res["latency_ms"]
    tokens = int(cost.get("prompt_tokens", 0) + cost.get("completion_tokens", 0))
    route = f"；路由：{'升级' if res['route']['escalate'] else '未升级'}" if res["route"] else ""
    print(f"\n耗时 {lat.get('total', 0) / 1000:.1f} s；LLM 调用 {cost.get('llm_calls', 0)} 次；工具调用 "
          f"{cost.get('tool_calls', 0)} 次；token {tokens}{route}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
