"""检索评测运行器与实验记录。

每次运行写到 eval/results/{时间}_{名称}/：
    run.json            git commit（及是否有未提交改动）、去掉密钥的配置快照、评测集与划分、检索器参数、运行参数
    metrics.json        按题型 / 来源 / 总体的汇总指标，以及检索延迟
    per_question.jsonl  逐题：top-K 的 chunk_id 与分数、每段证据的命中排名、各项指标（不含原文）
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
from novel_rag.eval.metrics import Span, aggregate, score_ranking
from novel_rag.retrieval.base import Hit, Retriever
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


def run_retrieval_eval(
    items: list[EvalItem],
    retriever: Retriever,
    spans: dict[str, Span],
    settings: Settings,
    *,
    base_filter: SearchFilter | None = None,
    spoiler_filter: bool = False,
    generator: Callable[[EvalItem, list[Hit]], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ks = settings.eval.ks
    top_k = max(ks)
    base = base_filter or SearchFilter()
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for item in items:
        flt = base.with_max_reading_order(item.max_reading_order) if spoiler_filter else base
        hits = retriever.search(item.question, top_k, flt)
        latencies.append(getattr(retriever, "last_latency_ms", {}).get("total", float("nan")))
        ranked = [spans[h.chunk_id] for h in hits]
        metrics = score_ranking(
            ranked,
            item.evidence,
            ks,
            min_ratio=settings.eval.relevance_min_ratio,
            min_chars=settings.eval.relevance_min_chars,
        )
        row = {
            "id": item.id,
            "type": item.type,
            "source": item.source,
            "split": item.split,
            "n_evidence": len(item.evidence),
            "evidence_ranks": metrics.pop("evidence_ranks"),
            "metrics": metrics,
            "hits": [{"rank": h.rank, "chunk_id": h.chunk_id, "score": round(h.score, 5)} for h in hits],
        }
        if generator is not None:
            row["generation"] = generator(item, hits)
        rows.append(row)
    summary = aggregate(rows, ks)
    valid = [x for x in latencies if x == x]
    summary["latency_ms"] = (
        {"p50": round(median(valid), 1), "p95": round(float(np.percentile(valid, 95)), 1)} if valid else {}
    )
    return rows, summary


def dataset_fingerprint(dataset_dir: Path) -> str:
    h = hashlib.sha256()
    for name in ("manifest.json", "items.jsonl"):
        h.update((dataset_dir / name).read_bytes())
    return h.hexdigest()


def render_summary(run: dict[str, Any], metrics: dict[str, Any], ks: tuple[int, ...]) -> str:
    cols = [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks] + ["mrr"]
    dirty = "（有未提交改动）" if run["git"]["dirty"] else ""
    lines = [
        f"# {run['name']}",
        "",
        f"- 时间：{run['created']}；commit `{run['git']['commit'][:10]}`{dirty}",
        f"- 评测集：{run['dataset']['name']}，划分 {run['dataset']['split']}，{metrics['overall']['n']} 题",
        f"- 检索器：{json.dumps(run['retriever'], ensure_ascii=False)}",
        f"- 分块版本：{run['chunk_version']}；防剧透过滤：{'开' if run['spoiler_filter'] else '关'}",
        f"- 相关性判定：重叠 ≥ min(区间 × {run['relevance']['min_ratio']}, {run['relevance']['min_chars']} 字)",
        f"- 检索延迟：{metrics.get('latency_ms')}",
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
