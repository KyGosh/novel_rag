"""问答评测（P4）：在评测集上运行一个系统（闭卷 / 固定 RAG / 固定 GraphRAG / Agent），判分并落盘到 eval/results/。

用法：
    uv run python scripts/run_qa.py --system closed   --name p4_closed_dev
    uv run python scripts/run_qa.py --system rag      --name p4_rag_dev
    uv run python scripts/run_qa.py --system graphrag --name p4_graphrag_dev
    uv run python scripts/run_qa.py --system agent    --name p4_agent_dev_s0 --sample 0
    uv run python scripts/run_qa.py --system agent    --name p4_agent_dev_s1 --sample 1   # 第二次独立运行
    uv run python scripts/run_qa.py --system agent    --name smoke --smoke                 # 冒烟子集（每类 6 题）
    uv run python scripts/run_qa.py --system agent    --name lat --latency-sample 30 --concurrency 1 --no-cache
    uv run python scripts/run_qa.py --system agent    --name no_graph --drop-tools find_entity,graph_query

固定只跑 dev（test 留到 P5）；每题按 max_reading_order 过滤（防剧透）；检索范围为评测集涉及的全部卷。
--no-cache 关闭 LLM 结果缓存（延迟测量用）；--sample 换一个采样编号得到独立的新样本。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from novel_rag.eval.dataset import QUESTION_TYPES, DatasetManifest, EvalItem, load_dataset


def smoke_subset(items: list[EvalItem], per_type: int = 6, seed: str = "p4-smoke") -> list[EvalItem]:
    """冒烟子集：按题型分层，每类取 hash(seed + id) 最小的 per_type 题（固定、与运行无关）。"""
    out = []
    for t in QUESTION_TYPES:
        cands = sorted(
            (i for i in items if i.type == t), key=lambda i: hashlib.sha256(f"{seed}:{i.id}".encode()).hexdigest()
        )
        out += cands[:per_type]
    return out


def latency_subset(items: list[EvalItem], n: int, seed: str = "p4-latency") -> list[EvalItem]:
    per = max(1, n // len(QUESTION_TYPES))
    return smoke_subset(items, per, seed)[:n]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", required=True, choices=["closed", "rag", "graphrag", "agent"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", default="asoiaf_v1")
    parser.add_argument("--split", choices=["dev"], default="dev", help="P4 只跑 dev")
    parser.add_argument("--sample", type=int, default=0, help="采样编号（参与 LLM 缓存键）")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="只跑冒烟子集（每类 6 题）")
    parser.add_argument("--latency-sample", type=int, help="分层抽 N 题（延迟测量用）")
    parser.add_argument("--ids", help="只跑这些题（逗号分隔）")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-cache", action="store_true", help="关闭 LLM 缓存（延迟测量）")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--drop-tools", help="Agent 去掉这些工具（逗号分隔，消融用）")
    parser.add_argument("--thinking", action="store_true", help="Agent 的规划与行动开启思考")
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--max-tool-calls", type=int)
    args = parser.parse_args()

    if args.no_cache:
        os.environ["LLM__CACHE_ENABLED"] = "false"
    from novel_rag.config import get_settings
    from novel_rag.eval.runner import dataset_fingerprint, git_info
    from novel_rag.llm import get_chat_model, get_token_counter
    from novel_rag.qa.env import load_env
    from novel_rag.qa.runner import results_dir_name, run_qa_eval, summarize, write_qa_run
    from novel_rag.qa.systems import ClosedBookSystem, FixedRAGSystem
    from novel_rag.tools import build_backends

    s = get_settings()
    agent_over = {k: v for k, v in {"max_rounds": args.max_rounds, "max_tool_calls": args.max_tool_calls,
                                    "thinking": True if args.thinking else None}.items() if v is not None}  # fmt: skip
    if agent_over:
        s = s.model_copy(update={"agent": s.agent.model_copy(update=agent_over)})
    ds_dir = s.paths.datasets_dir / args.dataset
    manifest = DatasetManifest.model_validate_json((ds_dir / "manifest.json").read_text(encoding="utf-8"))
    items = load_dataset(ds_dir, s.paths.books_dir, split=args.split)
    subset = None
    if args.smoke:
        items, subset = smoke_subset(items), "smoke"
    elif args.latency_sample:
        items, subset = latency_subset(items, args.latency_sample), f"latency{args.latency_sample}"
    if args.ids:
        want = {x.strip() for x in args.ids.split(",")}
        items, subset = [i for i in items if i.id in want], "ids"
    if args.limit:
        items = items[: args.limit]

    scope = tuple(manifest.books)
    env = load_env(s, scope)
    if env.text_sha != manifest.books:
        print("当前定稿文本与评测集标注时不一致", file=sys.stderr)
        return 1
    counter = get_token_counter()

    @lru_cache(maxsize=200_000)
    def n_tokens(text: str) -> int:
        return counter.count([text])[0]

    answer_chat = get_chat_model("answer")
    abstain = s.answer.abstain_text
    if args.system == "closed":
        system = ClosedBookSystem(answer_chat, abstain)
    elif args.system in ("rag", "graphrag"):
        routes = ("bm25", "vector", "graph") if args.system == "graphrag" else None
        b = build_backends(s, scope, env.chunks, env.titles, routes=routes, with_graph=False, with_summaries=False)
        system = FixedRAGSystem(
            args.system, b, answer_chat, n_tokens, scope=scope, top_k=s.answer.fixed_top_k,
            pool=s.eval.context_pool, budget=s.answer.context_token_budget, abstain=abstain,
        )  # fmt: skip
    else:
        from novel_rag.agent.system import AgentSystem
        from novel_rag.tools import Toolbox

        b = build_backends(s, scope, env.chunks, env.titles)
        tb = Toolbox(b)
        if args.drop_tools:
            tb = tb.without(*[x.strip() for x in args.drop_tools.split(",")])
        system = AgentSystem(tb, s, scope=scope, count_tokens=n_tokens)

    out_name = results_dir_name(args.name)
    out = s.paths.results_dir / out_name
    rows = run_qa_eval(
        items, system, settings=s, spans=env.spans, judge=None if args.no_judge else get_chat_model("answer_judge"),
        sample=args.sample, concurrency=args.concurrency, trace_dir=s.paths.traces_dir / out_name,
        count_tokens=n_tokens, chunk_text=lambda c: env.chunks.by_id(c)["text"],
    )  # fmt: skip
    metrics = summarize(rows, s)
    run = {
        "name": args.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "dataset": {"name": manifest.name, "split": args.split, "subset": subset, "n_items": len(items),
                    "fingerprint": dataset_fingerprint(ds_dir), "ids": [i.id for i in items] if subset else None},
        "system": system.describe(),
        "sample": args.sample,
        "concurrency": args.concurrency,
        "llm_cache": not args.no_cache,
        "judge": None if args.no_judge else {"task": "answer_judge", "model": s.llm.tasks["answer_judge"].model},
        "scope": list(scope),
        "chunk_version": env.chunk_version,
        "spoiler_filter": True,
        "trace_dir": str(Path("data") / "traces" / out_name),
        "config": s.snapshot(),
    }  # fmt: skip
    write_qa_run(out, run, metrics, rows)
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"结果目录：{out}")
    print(json.dumps({"errors": metrics["errors"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
