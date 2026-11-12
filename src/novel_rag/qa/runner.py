"""问答评测运行器（P4）：对评测集逐题运行一个系统、判分、计算证据指标，并落盘。

结果目录 eval/results/{时间}_{名称}/（不含原文）：
    run.json            commit、配置快照、评测集、系统参数、采样编号、并发数
    metrics.json        按题型 / 总体的答案准确率、证据指标、成本、终止原因分布，带 bootstrap 置信区间
    per_question.jsonl  逐题：简短答案、判分、引用、证据集合（chunk_id）、各项指标、成本、延迟、终止原因、trace 摘要
    summary.md          汇总表
含原文的完整 trace 放在 data/traces/{结果目录名}/{题目 id}.json（不入库）。
"""

from __future__ import annotations

import json
import time
import traceback
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from novel_rag.config import Settings
from novel_rag.eval.dataset import EvalItem
from novel_rag.eval.metrics import Span, coverage, is_relevant, score_ranking
from novel_rag.eval.runner import TYPE_NAMES, TYPE_ORDER
from novel_rag.eval.stats import bootstrap_ci
from novel_rag.llm.base import ChatModel
from novel_rag.qa.scoring import score_answer
from novel_rag.qa.systems import QASystem
from novel_rag.retrieval.expand import Context, pack_contexts
from novel_rag.tools.base import SpoilerLeak

ANSWER_KEYS = ("acc", "partial", "acc_lenient", "abstain")
EVIDENCE_KEYS = ("process_cov", "note_recall", "cite_recall", "cite_precision", "ctx_recall")
RETRIEVAL_KEYS = ("recall@5", "recall@20", "mrr", "cov@4000")
COST_KEYS = ("llm_calls", "tool_calls", "tokens")


def evidence_metrics(
    rec: dict[str, Any], item: EvalItem, spans: dict[str, Span], rel: dict[str, Any]
) -> dict[str, float | None]:
    """所有证据指标沿用 P1 的相关性判定（字符区间重叠），不依赖 chunk_id。"""

    def cov(ids: Sequence[str] | None) -> float | None:
        if ids is None:
            return None
        return coverage([spans[c] for c in ids if c in spans], item.evidence, **rel)

    sets = rec.get("evidence_sets", {})
    cited = rec["answer"].get("citations", [])
    out: dict[str, float | None] = {
        "process_cov": cov(sets.get("shown")),
        "note_recall": cov(sets.get("notes")),
        "cite_recall": cov(cited) if "shown" in sets else None,
        "ctx_recall": cov(sets.get("context")) if sets.get("context") is not None and "shown" in sets else None,
        "cite_precision": None,
    }
    if cited:
        ok = sum(1 for c in cited if c in spans and any(is_relevant(spans[c], ev, **rel) for ev in item.evidence))
        out["cite_precision"] = ok / len(cited)
    return out


def retrieval_metrics(
    hits: list[str], item: EvalItem, spans: dict[str, Span], rel: dict[str, Any], count: Callable[[str], int],
    chunk_text: Callable[[str], str], budget: int,
) -> dict[str, float]:  # fmt: skip
    """固定 RAG：R@5 / R@20 / MRR（前 20）与 cov@budget（结果池按排名贪心装箱，与 P2、P3 口径一致）。"""
    m = score_ranking([spans[c] for c in hits[:20]], item.evidence, (5, 20), **rel)
    ctxs = [
        Context(i + 1, spans[c].book_id, 0, 0, 0, spans[c].start, spans[c].end, chunk_text(c), [c], [c])
        for i, c in enumerate(hits)
    ]
    packed = pack_contexts(ctxs, budget, lambda c: count(c.text))
    return {
        "recall@5": m["recall@5"],
        "recall@20": m["recall@20"],
        "mrr": m["mrr"],
        f"cov@{budget}": coverage([Span(c.book_id, c.char_start, c.char_end) for c in packed], item.evidence, **rel),
    }


def _run_one(
    system: QASystem, item: EvalItem, sample: int, trace_dir: Path | None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    t0 = time.perf_counter()
    try:
        out = system.run(item, sample)
        rec, trace = out.record, out.trace
    except SpoilerLeak as e:  # 防剧透兜底断言：严重错误，该题失败
        rec = {"answer": {"answer": "", "abstained": True, "citations": []}, "termination": "spoiler_violation",
               "error": str(e), "cost": {}, "latency_ms": {}}  # fmt: skip
        trace = {"error": str(e)}
    except Exception as e:  # noqa: BLE001 —— 单题失败不能中断整个运行
        rec = {"answer": {"answer": "", "abstained": True, "citations": []}, "termination": "error",
               "error": f"{type(e).__name__}: {e}", "cost": {}, "latency_ms": {}}  # fmt: skip
        trace = {"error": traceback.format_exc()}
    rec.setdefault("latency_ms", {}).setdefault("wall", round((time.perf_counter() - t0) * 1000, 1))
    if trace_dir is not None and trace:
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f"{item.id}.json").write_text(
            json.dumps({"id": item.id, "question": item.question, **trace}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    return rec, trace


def run_qa_eval(
    items: list[EvalItem],
    system: QASystem,
    *,
    settings: Settings,
    spans: dict[str, Span],
    judge: ChatModel | None,
    sample: int = 0,
    concurrency: int = 4,
    trace_dir: Path | None = None,
    count_tokens: Callable[[str], int] | None = None,
    chunk_text: Callable[[str], str] | None = None,
    progress: Callable[[str], None] | None = print,
) -> list[dict[str, Any]]:
    rel = {"min_ratio": settings.eval.relevance_min_ratio, "min_chars": settings.eval.relevance_min_chars}
    budget = settings.answer.context_token_budget
    t0 = time.monotonic()
    done = 0

    def task(item: EvalItem) -> dict[str, Any]:
        nonlocal done
        rec, _ = _run_one(system, item, sample, trace_dir)
        ans = rec["answer"]
        score = score_answer(
            question=item.question, gold=item.answer, aliases=item.answer_aliases, answer=ans.get("answer", ""),
            abstained=bool(ans.get("abstained")), judge=judge, min_chars=settings.answer.rule_min_chars,
        )  # fmt: skip
        label = score["label"]
        metrics: dict[str, float | None] = {
            "acc": float(label == "correct"),
            "partial": float(label == "partial"),
            "acc_lenient": float(label in ("correct", "partial")),
            "abstain": float(label == "abstain"),
            **evidence_metrics(rec, item, spans, rel),
        }
        if rec.get("hits") and count_tokens is not None and chunk_text is not None:
            metrics.update(retrieval_metrics(rec["hits"], item, spans, rel, count_tokens, chunk_text, budget))
        cost = rec.get("cost", {})
        metrics["llm_calls"] = float(cost.get("llm_calls", 0))
        metrics["tool_calls"] = float(cost.get("tool_calls", 0))
        metrics["tokens"] = float(cost.get("prompt_tokens", 0) + cost.get("completion_tokens", 0))
        row = {
            "id": item.id, "type": item.type, "split": item.split, "max_reading_order": item.max_reading_order,
            "label": label, "scored_by": score["by"], "judge_reason": score.get("reason"),
            "answer": ans.get("answer", ""), "abstained": bool(ans.get("abstained")),
            "citations": ans.get("citations", []), "invalid_citations": ans.get("invalid_citations", []),
            "metrics": metrics, "cost": cost, "latency_ms": rec.get("latency_ms", {}),
            "termination": rec.get("termination"), "error": rec.get("error"),
            **{k: rec[k] for k in ("hits", "evidence_sets", "agent", "context_tokens") if k in rec},
        }  # fmt: skip
        done += 1
        if progress and (done % 10 == 0 or done == len(items)):
            progress(f"[{system.name}] {done}/{len(items)}，用时 {time.monotonic() - t0:.0f}s")
        return row

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        rows = list(pool.map(task, items))
    return rows


# ---------------------------------------------------------------- 汇总


def _metric_names(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r["metrics"] if k not in keys]
    return keys


def summarize(rows: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    e = settings.eval

    def group(rs: list[dict[str, Any]]) -> dict[str, Any]:
        if not rs:
            return {"n": 0}
        out: dict[str, Any] = {"n": len(rs)}
        ci: dict[str, list[float]] = {}
        for k in _metric_names(rs):
            vals = [r["metrics"][k] for r in rs if r["metrics"].get(k) is not None]
            if not vals:
                continue
            out[k] = round(mean(vals), 4)
            if len(vals) < len(rs):
                out[f"{k}__n"] = len(vals)
            lo, hi = bootstrap_ci(vals, n_samples=e.bootstrap_samples, seed=e.bootstrap_seed, level=e.ci_level)
            ci[k] = [round(lo, 4), round(hi, 4)]
        out["ci"] = ci
        out["termination"] = dict(Counter(r.get("termination") for r in rs))
        out["labels"] = dict(Counter(r["label"] for r in rs))
        out["scored_by"] = dict(Counter(r["scored_by"] for r in rs))
        out["invalid_citations"] = sum(len(r.get("invalid_citations", [])) for r in rs)
        return out

    by_type = {t: group([r for r in rows if r["type"] == t]) for t in TYPE_ORDER if any(r["type"] == t for r in rows)}
    lat = [r["latency_ms"].get("total") for r in rows if r["latency_ms"].get("total") is not None]
    return {
        "overall": group(rows),
        "by_type": by_type,
        "latency_ms": {"p50": round(median(lat), 1), "p95": round(float(np.percentile(lat, 95)), 1)} if lat else {},
        "errors": sum(1 for r in rows if r.get("error")),
        "ci_level": e.ci_level,
    }


def render_summary(run: dict[str, Any], metrics: dict[str, Any]) -> str:
    cols = [c for c in (*ANSWER_KEYS, *EVIDENCE_KEYS, *RETRIEVAL_KEYS, *COST_KEYS) if c in metrics["overall"]]
    dirty = "（有未提交改动）" if run["git"]["dirty"] else ""
    lines = [
        f"# {run['name']}",
        "",
        f"- 时间：{run['created']}；commit `{run['git']['commit'][:10]}`{dirty}",
        f"- 系统：{json.dumps(run['system'], ensure_ascii=False)}",
        f"- 评测集：{run['dataset']['name']}，划分 {run['dataset']['split']}，{metrics['overall']['n']} 题；"
        f"采样编号 {run['sample']}；并发 {run['concurrency']}",
        f"- 终止原因：{metrics['overall'].get('termination')}；判分方式：{metrics['overall'].get('scored_by')}；"
        f"无效引用 {metrics['overall'].get('invalid_citations')} 个；出错 {metrics['errors']} 题",
        f"- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{metrics.get('latency_ms')}",
        "",
        "| 分组 | 题数 | " + " | ".join(cols) + " |",
        "|---" * (len(cols) + 2) + "|",
    ]

    def row(label: str, m: dict[str, Any]) -> str:
        cells = []
        for c in cols:
            v = m.get(c)
            cells.append("—" if v is None else (f"{v:.1f}" if c in COST_KEYS else f"{v:.3f}"))
        return f"| {label} | {m['n']} | " + " | ".join(cells) + " |"

    for t in TYPE_ORDER:
        if t in metrics["by_type"]:
            lines.append(row(TYPE_NAMES[t], metrics["by_type"][t]))
    lines.append(row("**总体**", metrics["overall"]))
    ci = metrics["overall"]["ci"].get("acc")
    if ci:
        lines += ["", f"总体准确率 {metrics['overall']['acc']:.3f}，{int(metrics['ci_level'] * 100)}% CI {ci}"]
    return "\n".join(lines) + "\n"


def results_dir_name(name: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}"


def write_qa_run(out: Path, run: dict[str, Any], metrics: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    out.mkdir(parents=True, exist_ok=False)
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "per_question.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline=""
    )
    (out / "summary.md").write_text(render_summary(run, metrics), encoding="utf-8")
    return out
