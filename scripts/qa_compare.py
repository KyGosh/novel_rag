"""P4 第一轮对比表：从 eval/results/ 下的问答运行结果生成（不重新调用任何模型）。

用法：
    uv run python scripts/qa_compare.py --out p4_comparison \
        --run closed=eval/results/..._p4_closed_dev --run rag=eval/results/..._p4_rag_dev \
        --run graphrag=eval/results/..._p4_graphrag_dev \
        --run agent=eval/results/..._p4_agent_dev_s0 --run agent_s1=eval/results/..._p4_agent_dev_s1 \
        --latency rag=eval/results/..._p4_lat_rag --latency agent=eval/results/..._p4_lat_agent

输出 eval/results/{out}.md 与 {out}.json：按题型的准确率（带置信区间）、与参照系统的配对差值、闭卷可答 / 不可答
子集、证据指标、成本、终止原因、Agent 两次运行的差异、失败自动分类、工具使用统计、延迟拆解。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from novel_rag.config import get_settings
from novel_rag.eval.dataset import load_dataset
from novel_rag.eval.runner import TYPE_NAMES, TYPE_ORDER
from novel_rag.eval.stats import bootstrap_ci, paired_diff
from novel_rag.qa.analysis import STAGE_NAMES, failure_table, tool_usage
from novel_rag.qa.env import load_env

GROUPS = (*TYPE_ORDER, "overall")
GNAME = {**TYPE_NAMES, "overall": "总体"}
EVID = ("ctx_recall", "process_cov", "note_recall", "cite_recall", "cite_precision")
RETR = ("recall@5", "recall@20", "mrr", "cov@4000")


def load_rows(d: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (d / "per_question.jsonl").read_text(encoding="utf-8").splitlines() if x]


def sel(rows: list[dict[str, Any]], g: str) -> list[dict[str, Any]]:
    return rows if g == "overall" else [r for r in rows if r["type"] == g]


def avg(rows: list[dict[str, Any]], k: str) -> float | None:
    v = [r["metrics"][k] for r in rows if r["metrics"].get(k) is not None]
    return mean(v) if v else None


def fmt(x: float | None, d: int = 3) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="名称=结果目录")
    ap.add_argument("--latency", action="append", default=[], help="名称=延迟运行的结果目录")
    ap.add_argument("--reference", default="rag")
    ap.add_argument("--closed", default="closed")
    ap.add_argument("--agents", default="agent,agent_s1", help="Agent 的两次运行（逗号分隔）")
    ap.add_argument("--dataset", default="asoiaf_v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s = get_settings()
    e = s.eval
    runs = {k: Path(v) for k, v in (x.split("=", 1) for x in args.run)}
    rows = {k: {r["id"]: r for r in load_rows(d)} for k, d in runs.items()}
    meta = {k: json.loads((d / "run.json").read_text(encoding="utf-8")) for k, d in runs.items()}
    ids = sorted(set.intersection(*(set(r) for r in rows.values())))
    for k, r in rows.items():
        if len(r) != len(ids):
            print(f"警告：{k} 有 {len(r)} 题，只比较共同的 {len(ids)} 题", file=sys.stderr)
    R = {k: [rows[k][i] for i in ids] for k in rows}
    ci = {"n_samples": e.bootstrap_samples, "seed": e.bootstrap_seed, "level": e.ci_level}
    names = list(runs)
    out: dict[str, Any] = {
        "runs": {k: str(v) for k, v in runs.items()},
        "n": len(ids),
        "commits": {k: m["git"]["commit"][:10] for k, m in meta.items()},
    }
    md = [f"# P4 第一轮对比（{args.dataset} dev，{len(ids)} 题）", ""]
    md.append("运行：" + "；".join(f"{k} = `{runs[k].name}`（commit `{out['commits'][k]}`）" for k in names))
    md.append("")

    # ---- 1. 准确率
    md += [
        "## 1. 答案准确率（correct 比例，95% bootstrap CI）",
        "",
        "| 系统 | " + " | ".join(f"{GNAME[g]} (n={len(sel(R[names[0]], g))})" for g in GROUPS) + " |",
        "|---" * (len(GROUPS) + 1) + "|",
    ]
    acc: dict[str, dict[str, Any]] = {}
    for k in names:
        acc[k] = {}
        cells = []
        for g in GROUPS:
            v = [r["metrics"]["acc"] for r in sel(R[k], g)]
            lo, hi = bootstrap_ci(v, **ci)
            acc[k][g] = {
                "acc": mean(v),
                "ci": [lo, hi],
                "partial": avg(sel(R[k], g), "partial"),
                "abstain": avg(sel(R[k], g), "abstain"),
            }
            cells.append(f"{mean(v):.3f} [{lo:.2f}, {hi:.2f}]")
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    md += [
        "",
        "部分对 / 弃答率（总体）："
        + "；".join(f"{k} {fmt(acc[k]['overall']['partial'])} / {fmt(acc[k]['overall']['abstain'])}" for k in names),
        "",
    ]
    out["accuracy"] = acc

    # ---- 2. 配对差值
    ref = args.reference
    if ref in R:
        md += [
            f"## 2. 与 {ref} 的配对差值（准确率，差值 [95% CI]，p = 重采样中差值 ≤ 0 的比例）",
            "",
            "| 系统 | " + " | ".join(GNAME[g] for g in GROUPS) + " |",
            "|---" * (len(GROUPS) + 1) + "|",
        ]
        out["paired_vs_ref"] = {}
        for k in names:
            if k == ref:
                continue
            cells, d = [], {}
            for g in GROUPS:
                a = [r["metrics"]["acc"] for r in sel(R[ref], g)]
                b = [r["metrics"]["acc"] for r in sel(R[k], g)]
                pdf = paired_diff(a, b, **ci)
                d[g] = pdf
                cells.append(f"{pdf['diff']:+.3f} [{pdf['lo']:+.2f}, {pdf['hi']:+.2f}] p={pdf['p_le0']:.2f}")
            out["paired_vs_ref"][k] = d
            md.append(f"| {k} | " + " | ".join(cells) + " |")
        md.append("")

    # ---- 3. 闭卷子集
    if args.closed in R:
        cc = {r["id"] for r in R[args.closed] if r["label"] == "correct"}
        md += [
            f"## 3. 闭卷可答子集（闭卷答对 {len(cc)} 题 / 闭卷没答对 {len(ids) - len(cc)} 题）",
            "",
            "| 系统 | 闭卷答对的题 | 闭卷没答对的题 | 其中多跳+跨章节（闭卷没答对） |",
            "|---|---|---|---|",
        ]
        out["closed_subsets"] = {}
        for k in names:
            a = [r["metrics"]["acc"] for r in R[k] if r["id"] in cc]
            b = [r["metrics"]["acc"] for r in R[k] if r["id"] not in cc]
            h = [r["metrics"]["acc"] for r in R[k] if r["id"] not in cc and r["type"] in ("multihop", "plot")]
            out["closed_subsets"][k] = {
                "closed_correct": mean(a) if a else None,
                "closed_wrong": mean(b) if b else None,
                "closed_wrong_hard": mean(h) if h else None,
                "n": [len(a), len(b), len(h)],
            }
            md.append(
                f"| {k} | {fmt(mean(a) if a else None)} | {fmt(mean(b) if b else None)} | "
                f"{fmt(mean(h) if h else None)} (n={len(h)}) |"
            )
        md.append("")

    # ---- 4. 证据指标
    md += [
        "## 4. 证据指标（总体 / 多跳 / 跨章节）",
        "",
        "同预算召回 ctx_recall = 作答环节实际看到的上下文（4000 token）覆盖的金标比例；固定 RAG 另列 R@5 / R@20 / "
        "cov@4000（结果池 40 个按排名装箱，与 P3 基线口径一致）。",
        "",
        "| 系统 | 分组 | " + " | ".join((*EVID, *RETR)) + " |",
        "|---" * (len(EVID) + len(RETR) + 2) + "|",
    ]
    out["evidence"] = {}
    for k in names:
        out["evidence"][k] = {}
        for g in ("overall", "multihop", "plot"):
            vals = {m: avg(sel(R[k], g), m) for m in (*EVID, *RETR)}
            out["evidence"][k][g] = vals
            if all(v is None for v in vals.values()):
                continue
            md.append(f"| {k} | {GNAME[g]} | " + " | ".join(fmt(vals[m]) for m in (*EVID, *RETR)) + " |")
    md.append("")

    # ---- 5. 成本与终止原因
    md += [
        "## 5. 成本与终止原因（每题平均）",
        "",
        "| 系统 | LLM 调用 | 工具调用 | token | 终止原因 |",
        "|---|---|---|---|---|",
    ]
    out["cost"] = {}
    for k in names:
        c = {m: avg(R[k], m) for m in ("llm_calls", "tool_calls", "tokens")}
        term = dict(Counter(r.get("termination") for r in R[k]))
        out["cost"][k] = {**c, "termination": term}
        md.append(f"| {k} | {fmt(c['llm_calls'], 2)} | {fmt(c['tool_calls'], 2)} | {fmt(c['tokens'], 0)} | {term} |")
    md.append("")

    # ---- 6. Agent 两次运行的差异
    agents = [a for a in args.agents.split(",") if a in R]
    if len(agents) == 2:
        a0, a1 = agents
        same = mean(float(x["label"] == y["label"]) for x, y in zip(R[a0], R[a1], strict=True))
        md += [
            f"## 6. Agent 两次独立运行（{a0} / {a1}，采样编号不同）",
            "",
            f"逐题判分一致率 {same:.3f}。",
            "",
            "| 分组 | " + f"{a0} | {a1} | 差值 {a1}−{a0} [95% CI] |",
            "|---|---|---|---|",
        ]
        out["agent_runs"] = {"label_agreement": same, "by_group": {}}
        for g in GROUPS:
            x = [r["metrics"]["acc"] for r in sel(R[a0], g)]
            y = [r["metrics"]["acc"] for r in sel(R[a1], g)]
            pdf = paired_diff(x, y, **ci)
            out["agent_runs"]["by_group"][g] = {"a0": mean(x), "a1": mean(y), **pdf}
            diff = f"{pdf['diff']:+.3f} [{pdf['lo']:+.2f}, {pdf['hi']:+.2f}]"
            md.append(f"| {GNAME[g]} | {mean(x):.3f} | {mean(y):.3f} | {diff} |")
        md.append("")

    # ---- 7. 失败分类与工具使用
    if agents:
        env = load_env(s, tuple(meta[agents[0]]["scope"]))
        items = {i.id: i for i in load_dataset(s.paths.datasets_dir / args.dataset, s.paths.books_dir, split="dev")}
        rel = {"min_ratio": e.relevance_min_ratio, "min_chars": e.relevance_min_chars}
        ft = failure_table(R[agents[0]], items, env.spans, rel)
        out["failures"] = {k: v for k, v in ft.items() if k != "per_question"}
        keys = [k for k in STAGE_NAMES if any(k in v for v in ft["by_type"].values())]
        md += [
            f"## 7. Agent 失败自动分类（{agents[0]}；没答对的题按环节归一类，附加标记可多选）",
            "",
            "| 题型 | 没答对 | " + " | ".join(STAGE_NAMES[k] for k in keys) + " |",
            "|---" * (len(keys) + 2) + "|",
        ]
        for t in (*TYPE_ORDER, "overall"):
            c = ft["overall"] if t == "overall" else ft["by_type"].get(t, {})
            md.append(f"| {GNAME[t]} | {c.get('_failed', 0)} | " + " | ".join(str(c.get(k, 0)) for k in keys) + " |")
        tu = tool_usage(R[agents[0]], items, env.spans, rel)
        out["tool_usage"] = tu
        md += [
            "",
            "## 8. 工具使用统计",
            "",
            "| 工具 | 每题调用 | 使用该工具的题目比例 | 出错次数 | 正确答案所引金标证据的来源 |",
            "|---|---|---|---|---|",
        ]
        for tname in sorted(tu["calls"], key=lambda x: -tu["calls"][x]):
            md.append(
                f"| {tname} | {tu['calls_per_question'][tname]:.2f} | {tu['questions_using'][tname]:.2f} | "
                f"{tu['errors'].get(tname, 0)} | {tu['correct_cited_gold_source'].get(tname, 0)} |"
            )
        md.append("")

    # ---- 9. 延迟
    if args.latency:
        md += [
            "## 9. 延迟（单独运行：逐题顺序执行、独占 GPU、关闭 LLM 缓存；ms）",
            "",
            "| 系统 | 题数 | 总 P50 | 总 P95 | LLM P50 | 检索 P50 | rerank P50 | 图谱 P50 | 读取 P50 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        out["latency"] = {}
        for x in args.latency:
            k, d = x.split("=", 1)
            lr = load_rows(Path(d))

            def pct(key: str, q: float, lr=lr) -> float | None:
                v = [r["latency_ms"].get(key, 0.0) for r in lr if r["latency_ms"].get("total") is not None]
                return float(np.percentile(v, q)) if v else None

            lat = {
                key: {"p50": pct(key, 50), "p95": pct(key, 95)}
                for key in ("total", "llm", "retrieval", "rerank", "graph", "read")
            }
            out["latency"][k] = {"n": len(lr), **lat, "dir": d}
            md.append(
                f"| {k} | {len(lr)} | {fmt(lat['total']['p50'], 0)} | {fmt(lat['total']['p95'], 0)} | "
                f"{fmt(lat['llm']['p50'], 0)} | {fmt(lat['retrieval']['p50'], 0)} | {fmt(lat['rerank']['p50'], 0)} | "
                f"{fmt(lat['graph']['p50'], 0)} | {fmt(lat['read']['p50'], 0)} |"
            )
        md.append("")

    path = s.paths.results_dir / f"{args.out}.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    (s.paths.results_dir / f"{args.out}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print("\n".join(md))
    print(f"\n已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
