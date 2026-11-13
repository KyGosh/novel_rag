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

P5 消融开关（只对 Agent 生效）：
    --no-graph                去掉全部图谱：去掉 find_entity / graph_query，原文检索也不再支持按实体限定
    --drop-tools graph_query  只去掉图谱查询；--drop-tools search_summaries,read_summary 去掉摘要
    --graph-scope person      图谱工具只看人物子图（运行环境注入，模型不可见）
    --act-memory raw          行动环节另外看到历次工具返回的原文
    --max-rounds / --max-tool-calls / --max-tokens   预算档位；--thinking 开启思考

默认只跑 dev。test 只在配置冻结后跑一次：必须同时给 --split test 和 --confirm-test，每次 test 运行都会追加记入
eval/results/test_runs.jsonl。每题按 max_reading_order 过滤（防剧透）；检索范围为评测集涉及的全部卷。
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
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--confirm-test", action="store_true", help="确认要跑 test（配置冻结后只跑一次）")
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
    parser.add_argument("--max-tokens", type=int, help="Agent 每题 token 上限")
    parser.add_argument("--no-graph", action="store_true", help="Agent 去掉全部图谱（含按实体限定的检索）")
    parser.add_argument("--graph-scope", choices=["all", "person"], default="all", help="图谱工具的实体范围")
    parser.add_argument("--act-memory", choices=["notes", "raw"], default="notes", help="行动环节的记忆模式")
    args = parser.parse_args()

    if args.split == "test" and not args.confirm_test:
        print("test 只在配置冻结后跑一次；确认要跑请加 --confirm-test", file=sys.stderr)
        return 2
    if args.split == "test" and (args.smoke or args.latency_sample or args.ids or args.limit):
        print("test 不支持子集运行（避免反复查看 test 结果）", file=sys.stderr)
        return 2

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
                                    "max_tokens": args.max_tokens, "thinking": True if args.thinking else None,
                                    "act_memory": "raw" if args.act_memory == "raw" else None}.items()
                  if v is not None}  # fmt: skip
    if agent_over:
        s = s.model_copy(update={"agent": s.agent.model_copy(update=agent_over)})
    if args.graph_scope == "person":
        s = s.model_copy(update={"tools": s.tools.model_copy(update={"graph_scope_entity_types": ("Person",)})})
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
        if args.no_graph:
            tb = tb.without_graph()
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
        "variant": {"no_graph": args.no_graph, "drop_tools": args.drop_tools, "graph_scope": args.graph_scope,
                    "act_memory": args.act_memory, "thinking": args.thinking},
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
    if args.split == "test":  # test 运行台账：只追加，不含逐题结果
        ledger = {"dir": out_name, "created": run["created"], "system": args.system, "sample": args.sample,
                  "dataset": args.dataset, "commit": run["git"]["commit"], "dirty": run["git"]["dirty"],
                  "n_items": len(items), "acc": metrics["overall"].get("acc")}  # fmt: skip
        with (s.paths.results_dir / "test_runs.jsonl").open("a", encoding="utf-8", newline="") as f:
            f.write(json.dumps(ledger, ensure_ascii=False) + "\n")
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"结果目录：{out}")
    print(json.dumps({"errors": metrics["errors"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
