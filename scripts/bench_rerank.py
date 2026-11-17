"""rerank 延迟排查（P4 方案第一节）：同一时刻、同一进程里，对 v0 与 v1 的 dev 问题各取 N 题，
先检索出 40 个候选，再单独计时 rerank（预热后），比较两套问题的 P50 / P95，并记录 GPU 占用情况。

v1 基线的 rerank P50（1221 ms）约是 P2（630 ms）的两倍；检索范围和 reranker 配置都相同，所以要分清是
问题本身（v1 问题更长）还是当时 GPU 被别的程序占用。结果写入 eval/results/{时间}_rerank_bench/。

用法：
    uv run python scripts/bench_rerank.py --n 40 --label wallpaper_on
    uv run python scripts/bench_rerank.py --n 40 --gaps 0,0.5,2 --label gap   # P6：两次 rerank 之间的空闲时长

--gaps（P6 B1 排查）：流水线中 rerank 的 P50（0.76 s）高于连续计时的基准（0.57 s），P4 推测是两次 rerank 之间
等待 LLM 时 GPU 降频。每道题按随机顺序在每种空闲时长下各计时一次（先空闲 gap 秒再 rerank），比较各档的 P50。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from statistics import mean

import numpy as np

from novel_rag.config import get_settings
from novel_rag.eval.dataset import load_dataset
from novel_rag.eval.runner import git_info
from novel_rag.llm import get_reranker, get_token_counter
from novel_rag.qa.env import load_env
from novel_rag.retrieval.factory import build_retriever
from novel_rag.retrieval.filters import SearchFilter


def gpu_snapshot() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,clocks.sm", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def gap_bench(args: argparse.Namespace, s) -> int:  # noqa: ANN001
    import random

    gaps = [float(x) for x in args.gaps.split(",")]
    env = load_env(s)
    s_norerank = s.model_copy(update={"retrieval": s.retrieval.model_copy(update={"rerank": False})})
    retr = build_retriever(s_norerank, env.scope, "hybrid", chunk_index=env.chunks)
    reranker = get_reranker()
    items = load_dataset(s.paths.datasets_dir / "asoiaf_v1", s.paths.books_dir, split="dev")[: args.n]
    rng = random.Random(20261117)
    times: dict[float, list[float]] = {g: [] for g in gaps}
    gpu_before = gpu_snapshot()
    for k, it in enumerate(items):
        flt = SearchFilter(book_ids=env.scope, max_reading_order=it.max_reading_order)
        texts = [h.fields.get("text", "") for h in retr.search(it.question, s.retrieval.rerank_candidates, flt)]
        if k == 0:
            reranker.score(it.question, texts)  # 预热
        order = gaps[:]
        rng.shuffle(order)
        for g in order:
            time.sleep(g)
            t0 = time.perf_counter()
            reranker.score(it.question, texts)
            times[g].append((time.perf_counter() - t0) * 1000)
    res = {
        "label": args.label, "n": len(items), "gpu_before": gpu_before, "gpu_after": gpu_snapshot(),
        "results": {str(g): {"p50": round(float(np.percentile(v, 50)), 1), "p95": round(float(np.percentile(v, 95)), 1),
                             "mean": round(mean(v), 1)} for g, v in times.items()},
    }  # fmt: skip
    d = s.paths.results_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_rerank_gap_bench"
    d.mkdir(parents=True)
    run = {"git": git_info(), "reranker": s.reranker.model_dump(), "candidates": s.retrieval.rerank_candidates,
           "gaps_s": gaps}  # fmt: skip
    (d / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"结果目录：{d}")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--label", default="", help="运行条件说明（如 wallpaper_on / wallpaper_off）")
    ap.add_argument("--gaps", help="逗号分隔的空闲秒数（P6：验证空闲后 GPU 降频的假设）")
    args = ap.parse_args()
    s = get_settings()
    if args.gaps:
        return gap_bench(args, s)
    env = load_env(s)
    s_norerank = s.model_copy(update={"retrieval": s.retrieval.model_copy(update={"rerank": False})})
    retr = build_retriever(s_norerank, env.scope, "hybrid", chunk_index=env.chunks)
    reranker = get_reranker()
    counter = get_token_counter()
    out: dict[str, dict] = {}
    warm = True
    for ds in ("asoiaf_v0", "asoiaf_v1"):
        items = load_dataset(s.paths.datasets_dir / ds, s.paths.books_dir, split="dev")[: args.n]
        times, qlens, ctoks = [], [], []
        for it in items:
            flt = SearchFilter(book_ids=env.scope, max_reading_order=it.max_reading_order)
            hits = retr.search(it.question, s.retrieval.rerank_candidates, flt)
            texts = [h.fields.get("text", "") for h in hits]
            if warm:
                reranker.score(it.question, texts)
                warm = False
            t0 = time.perf_counter()
            reranker.score(it.question, texts)
            times.append((time.perf_counter() - t0) * 1000)
            qlens.append(len(it.question))
            ctoks.append(sum(counter.count([it.question + t for t in texts])))
        out[ds] = {
            "n": len(items),
            "rerank_ms_p50": round(float(np.percentile(times, 50)), 1),
            "rerank_ms_p95": round(float(np.percentile(times, 95)), 1),
            "question_chars_mean": round(mean(qlens), 1),
            "pair_tokens_mean": round(mean(ctoks), 1),
            "ms_per_1k_tokens": round(1000 * mean(times) / mean(ctoks), 2),
        }
        print(ds, out[ds], flush=True)
    res = {"label": args.label, "gpu": gpu_snapshot(), "results": out}
    d = s.paths.results_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_rerank_bench"
    d.mkdir(parents=True)
    run = {"git": git_info(), "reranker": s.reranker.model_dump(), "candidates": s.retrieval.rerank_candidates}
    (d / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))
    print(f"结果目录：{d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
