"""检索评测运行器与实验记录。

每次运行写到 eval/results/{时间}_{名称}/：
    run.json            git commit（及是否有未提交改动）、去掉密钥的配置快照、评测集与划分、检索器参数、运行参数
    metrics.json        按题型 / 来源 / 总体的汇总指标及 bootstrap 置信区间，以及分阶段的检索延迟
    per_question.jsonl  逐题：top-K 的 chunk_id 与分数（含每一路的排名）、每段证据的命中排名、各项指标（不含原文）
    summary.md          便于阅读的汇总表
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from novel_rag.config import PROJECT_ROOT, Settings
from novel_rag.eval.dataset import EvalItem
from novel_rag.eval.metrics import Span, aggregate, coverage, metric_keys, score_ranking
from novel_rag.eval.stats import bootstrap_ci
from novel_rag.retrieval.base import Hit, Retriever
from novel_rag.retrieval.expand import Context, pack_contexts
from novel_rag.retrieval.filters import SearchFilter

TYPE_ORDER = ("fact", "relation", "multihop", "plot")
TYPE_NAMES = {"fact": "事实", "relation": "人物关系", "multihop": "多跳", "plot": "跨章节情节"}


def git_info(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def _hit_record(h: Hit) -> dict[str, Any]:
    rec: dict[str, Any] = {"rank": h.rank, "chunk_id": h.chunk_id, "score": round(h.score, 5)}
    routes = h.fields.get("routes")
    if routes:
        rec["routes"] = {n: d["rank"] for n, d in routes.items()}
    return rec


def _percentiles(values: list[float]) -> dict[str, float]:
    valid = [x for x in values if x == x]
    if not valid:
        return {}
    return {"p50": round(median(valid), 1), "p95": round(float(np.percentile(valid, 95)), 1)}


def add_confidence_intervals(summary: dict[str, Any], rows: list[dict[str, Any]], settings: Settings) -> None:
    """给总体与各题型的每个指标加 bootstrap 置信区间：summary["ci"][分组][指标] = [lo, hi]。"""
    e = settings.eval
    keys = metric_keys(rows, e.ks)
    groups = {"overall": rows, **{f"type:{t}": [r for r in rows if r["type"] == t] for t in TYPE_ORDER}}
    ci: dict[str, dict[str, list[float]]] = {}
    for g, rs in groups.items():
        if not rs:
            continue
        ci[g] = {}
        for k in keys:
            if k not in rs[0]["metrics"]:
                continue
            lo, hi = bootstrap_ci(
                [r["metrics"][k] for r in rs], n_samples=e.bootstrap_samples, seed=e.bootstrap_seed, level=e.ci_level
            )
            ci[g][k] = [round(lo, 4), round(hi, 4)]
    summary["ci"] = ci
    summary["ci_level"] = e.ci_level


def run_retrieval_eval(
    items: list[EvalItem],
    retriever: Retriever,
    spans: dict[str, Span],
    settings: Settings,
    *,
    base_filter: SearchFilter | None = None,
    spoiler_filter: bool = False,
    generator: Callable[[EvalItem, list[Hit]], dict[str, Any]] | None = None,
    expander: Callable[[list[Hit], SearchFilter], list[Context]] | None = None,
    count_tokens: Callable[[Context], int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """expander + count_tokens 都给出时，额外计算固定 token 预算下的证据覆盖率 cov@B（B 取 eval.token_budgets）。"""
    ks = settings.eval.ks
    with_context = expander is not None and count_tokens is not None
    # 排名指标只看前 max(ks) 个；装箱可以用更深的结果池，否则大预算下不扩展的一方会被池子大小卡住
    top_k = max(max(ks), settings.eval.context_pool) if with_context else max(ks)
    base = base_filter or SearchFilter()
    rel = {"min_ratio": settings.eval.relevance_min_ratio, "min_chars": settings.eval.relevance_min_chars}
    rows: list[dict[str, Any]] = []
    latencies: list[dict[str, float]] = []
    for item in items:
        flt = base.with_max_reading_order(item.max_reading_order) if spoiler_filter else base
        hits = retriever.search(item.question, top_k, flt)
        latencies.append(dict(getattr(retriever, "last_latency_ms", {}) or {}))
        ranked = [spans[h.chunk_id] for h in hits[: max(ks)]]
        metrics = score_ranking(ranked, item.evidence, ks, **rel)
        row = {
            "id": item.id,
            "type": item.type,
            "source": item.source,
            "split": item.split,
            "n_evidence": len(item.evidence),
            "evidence_ranks": metrics.pop("evidence_ranks"),
            "metrics": metrics,
            "hits": [_hit_record(h) for h in hits],
        }
        if with_context:
            contexts = expander(hits, flt)
            row["contexts"] = []
            for budget in settings.eval.token_budgets:
                packed = pack_contexts(contexts, budget, count_tokens)
                cspans = [Span(c.book_id, c.char_start, c.char_end) for c in packed]
                metrics[f"cov@{budget}"] = coverage(cspans, item.evidence, **rel)
                row["contexts"].append(
                    {"budget": budget, "n": len(packed), "tokens": sum(count_tokens(c) for c in packed)}
                )
        if generator is not None:
            row["generation"] = generator(item, hits)
        rows.append(row)
    summary = aggregate(rows, ks)
    add_confidence_intervals(summary, rows, settings)
    summary["latency_ms"] = _percentiles([lat.get("total", float("nan")) for lat in latencies])
    stages = sorted({k for lat in latencies for k in lat if k != "total"})
    summary["latency_stages_ms"] = {k: _percentiles([lat.get(k, float("nan")) for lat in latencies]) for k in stages}
    return rows, summary


def dataset_fingerprint(dataset_dir: Path) -> str:
    h = hashlib.sha256()
    for name in ("manifest.json", "items.jsonl"):
        h.update((dataset_dir / name).read_bytes())
    return h.hexdigest()


def render_summary(run: dict[str, Any], metrics: dict[str, Any], ks: tuple[int, ...]) -> str:
    cols = [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks] + ["mrr"]
    cols += [k for k in metrics["overall"] if k.startswith("cov@")]
    dirty = "（有未提交改动）" if run["git"]["dirty"] else ""
    lines = [
        f"# {run['name']}",
        "",
        f"- 时间：{run['created']}；commit `{run['git']['commit'][:10]}`{dirty}",
        f"- 评测集：{run['dataset']['name']}，划分 {run['dataset']['split']}，{metrics['overall']['n']} 题",
        f"- 检索器：{json.dumps(run['retriever'], ensure_ascii=False)}",
        f"- 分块版本：{run['chunk_version']}；防剧透过滤：{'开' if run['spoiler_filter'] else '关'}",
        f"- 相关性判定：重叠 ≥ min(区间 × {run['relevance']['min_ratio']}, {run['relevance']['min_chars']} 字)",
        f"- 检索延迟（ms）：{metrics.get('latency_ms')}；分阶段：{json.dumps(metrics.get('latency_stages_ms', {}))}",
        f"- 范围：{run.get('scope')}；small-to-big 窗口：{run.get('expand_window')}",
        "",
        "| 分组 | 题数 | " + " | ".join(cols) + " |",
        "|---" * (len(cols) + 2) + "|",
    ]

    def row(label: str, m: dict[str, Any]) -> str:
        return f"| {label} | {m['n']} | " + " | ".join(f"{m.get(c, 0):.3f}" for c in cols) + " |"

    for t in TYPE_ORDER:
        if t in metrics["by_type"]:
            lines.append(row(TYPE_NAMES[t], metrics["by_type"][t]))
    for s, m in metrics["by_source"].items():
        lines.append(row(f"来源：{s}", m))
    lines.append(row("**总体**", metrics["overall"]))
    return "\n".join(lines) + "\n"


def write_run(
    results_dir: Path,
    name: str,
    run: dict[str, Any],
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    ks: tuple[int, ...],
) -> Path:
    out = results_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "per_question.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline=""
    )
    (out / "summary.md").write_text(render_summary(run, metrics, ks), encoding="utf-8")
    return out
