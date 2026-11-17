# ruff: noqa: E501 —— 含较长的中文表头与说明文字
"""延迟拆解报告（P6 B1）：从延迟测量运行（逐题顺序执行、关闭 LLM 缓存）汇总各环节耗时，不手抄数字。

用法：
    uv run python scripts/latency_report.py \
        --run "Agent（逐个执行工具）=p6_lat_agent_seq" --run "Agent（并行执行工具）=p6_lat_agent_par" \
        --run "固定 RAG=p6_lat_rag" --run "按需升级=p6_lat_cascade" \
        --replay seq=p6_replaytime_seq_r1,p6_replaytime_seq_r2 --replay par=p6_replaytime_par_r1,p6_replaytime_par_r2 \
        --rerank-gap p6 --out p6_latency
- 每个系统：端到端延迟（平均 / P50 / P95）、每题 LLM 调用、LLM 耗时、工具墙钟时间、rerank；Agent 再按环节拆
  （规划 / 行动 / 反思 / 作答，每次调用的平均耗时与每题次数；各工具的次数与单次耗时）。
- --replay：缓存回放（LLM 命中缓存、几乎不耗时）下同一批题逐个执行与并行执行工具的配对比较——轨迹完全相同，
  差别只来自工具执行方式，比关闭缓存的端到端测量干净得多。
- --rerank-gap：rerank 空闲时长实验（scripts/bench_rerank.py --gaps）的结果目录名后缀（取最新一次）。
输出 eval/results/{out}.{md,json}。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from statistics import mean, median
from typing import Any

import numpy as np

from novel_rag.config import get_settings
from novel_rag.eval.ablation import find_run, load_run


def pct(v: list[float], q: float) -> float:
    return float(np.percentile(v, q)) if v else float("nan")


def system_stats(run: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in run["rows"] if not r.get("error")]
    lat = [r["latency_ms"] for r in rows]
    tot = [x["total"] for x in lat]
    out: dict[str, Any] = {
        "dir": run["dir"], "n": len(rows), "errors": len(run["rows"]) - len(rows),
        "llm_cache": run["run"].get("llm_cache"), "concurrency": run["run"].get("concurrency"),
        "created": run["run"]["created"], "acc": mean(r["label"] == "correct" for r in rows) if rows[0].get("label") else None,
        "total": {"mean": mean(tot), "p50": median(tot), "p95": pct(tot, 95)},
        "llm_calls": mean(r["metrics"]["llm_calls"] for r in rows),
        "tokens": mean(r["metrics"]["tokens"] for r in rows),
    }  # fmt: skip
    for key in (
        "llm",
        "tools_wall",
        "retrieval",
        "rerank",
        "graph",
        "read",
        "agent.llm",
        "agent.tools_wall",
        "stage1",
        "route",
    ):
        vals = [x.get(key, 0.0) for x in lat]
        if any(vals):
            out[key] = mean(vals)
    esc = [bool((r.get("route") or {}).get("escalate")) for r in rows]
    if any(r.get("route") for r in rows):
        out["escalation"] = mean(esc)
    # Agent 按环节拆
    steps: dict[str, list[float]] = defaultdict(list)
    tools: dict[str, list[float]] = defaultdict(list)
    per_q: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        tr = (r.get("agent") or {}).get("trace", [])
        if not tr:
            continue
        cnt: dict[str, int] = defaultdict(int)
        for st in tr:
            if st["step"] in ("plan", "act", "reflect") and "latency_ms" in st:
                steps[st["step"]].append(st["latency_ms"])
                cnt[st["step"]] += 1
            elif st["step"] == "tool":
                tools[st["tool"]].append(st.get("latency_ms", 0.0))
                cnt["tool"] += 1
        for k in ("plan", "act", "reflect", "tool"):
            per_q[k].append(cnt[k])
    if steps:
        out["stages"] = {k: {"per_q": mean(per_q[k]), "ms": mean(v)} for k, v in steps.items()}
        out["stages"]["tool"] = {"per_q": mean(per_q["tool"]), "ms": mean(x for v in tools.values() for x in v)}
        out["tools"] = {
            k: {"per_q": len(v) / len(rows), "ms": mean(v), "p50": median(v)}
            for k, v in sorted(tools.items(), key=lambda kv: -len(kv[1]))
        }
        ans = [r["latency_ms"]["llm"] - sum(st.get("latency_ms", 0.0) for st in r["agent"]["trace"] if st["step"] in ("plan", "act", "reflect"))
               for r in rows if r.get("agent")]  # fmt: skip
        out["stages"]["answer"] = {"per_q": 1.0, "ms": mean(ans)}
    return out


def replay_pair(seq: list[dict[str, Any]], par: list[dict[str, Any]]) -> dict[str, Any]:
    def wall(r: dict[str, Any]) -> float:
        return sum(st.get("wall_ms", 0.0) for st in r["agent"]["trace"] if st["step"] == "execute")

    def tsum(r: dict[str, Any]) -> float:
        return sum(st.get("latency_ms", 0.0) for st in r["agent"]["trace"] if st["step"] == "tool")

    def per_q(runs: list[dict[str, Any]], f) -> dict[str, float]:  # noqa: ANN001
        ids = set.intersection(*({r["id"] for r in run["rows"] if not r.get("error")} for run in runs))
        return {i: mean(f(next(r for r in run["rows"] if r["id"] == i)) for run in runs) for i in ids}

    a, b = per_q(seq, wall), per_q(par, wall)
    ids = sorted(set(a) & set(b))
    multi = per_q(par, lambda r: sum(1 for st in r["agent"]["trace"] if st["step"] == "execute" and st["n"] > 1))
    return {
        "n": len(ids), "seq_dirs": [r["dir"] for r in seq], "par_dirs": [r["dir"] for r in par],
        "seq_tools_wall_ms": mean(a[i] for i in ids), "par_tools_wall_ms": mean(b[i] for i in ids),
        "saved_ms": mean(a[i] - b[i] for i in ids), "saved_ratio": 1 - sum(b[i] for i in ids) / sum(a[i] for i in ids),
        "multi_call_rounds_per_q": mean(multi[i] for i in ids),
        "seq_tool_sum_ms": mean(per_q(seq, tsum)[i] for i in ids),
    }  # fmt: skip


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], help="标签=实验名")
    ap.add_argument("--replay", action="append", default=[], help="seq=名1,名2 / par=名1,名2")
    ap.add_argument("--rerank-gap", help="rerank 空闲时长实验：结果目录名后缀（rerank_gap_bench）")
    ap.add_argument("--out", default="p6_latency")
    args = ap.parse_args()
    rd = get_settings().paths.results_dir
    res: dict[str, Any] = {"systems": {}}
    for spec in args.run:
        label, name = spec.split("=", 1)
        res["systems"][label] = system_stats(load_run(find_run(rd, name)))
    rp = {k: [load_run(find_run(rd, n)) for n in v.split(",")] for k, v in (x.split("=", 1) for x in args.replay)}
    if "seq" in rp and "par" in rp:
        res["replay"] = replay_pair(rp["seq"], rp["par"])
    if args.rerank_gap:
        d = sorted(p for p in rd.glob("*_rerank_gap_bench") if p.is_dir())[-1]
        res["rerank_gap"] = {"dir": d.name, **json.loads((d / "metrics.json").read_text(encoding="utf-8"))}

    s1 = lambda ms: f"{ms / 1000:.2f} s"  # noqa: E731
    md = ["# 延迟拆解（P6 B1）", "",
          "口径与 P4 相同：v1 dev 分层抽 28 题、逐题顺序执行（并发 1）、关闭 LLM 缓存，同一时段内依次测完。", "",
          "## 端到端", "", "| 系统 | 运行 | 题数 | 平均 | P50 | P95 | 每题 LLM 调用 | LLM 耗时 | 工具墙钟 | rerank | 每题 token | 准确率 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for label, v in res["systems"].items():
        acc = "—" if v["acc"] is None else f"{v['acc']:.3f}"
        md.append(
            f"| {label} | `{v['dir']}` | {v['n']} | {s1(v['total']['mean'])} | {s1(v['total']['p50'])} | {s1(v['total']['p95'])} | "
            f"{v['llm_calls']:.2f} | {s1(v.get('llm', v.get('agent.llm', 0)))} | {s1(v.get('tools_wall', v.get('retrieval', 0)))} | "
            f"{s1(v.get('rerank', 0))} | {v['tokens']:.0f} | {acc} |"
        )
    for label, v in res["systems"].items():
        if "stages" not in v:
            continue
        md += ["", f"## {label}：按环节拆", "", "| 环节 | 每题次数 | 单次平均 | 每题合计 |", "|---|---|---|---|"]
        names = {
            "plan": "规划（LLM）",
            "act": "行动（LLM）",
            "reflect": "反思（LLM）",
            "answer": "作答（LLM）",
            "tool": "工具执行（单次耗时之和）",
        }
        for k in ("plan", "act", "reflect", "answer", "tool"):
            st = v["stages"].get(k)
            if st:
                md.append(f"| {names[k]} | {st['per_q']:.2f} | {s1(st['ms'])} | {s1(st['per_q'] * st['ms'])} |")
        if "tools_wall" in v:
            md.append(f"| 工具执行（墙钟，同一轮并行时小于单次之和） | — | — | {s1(v['tools_wall'])} |")
        md += ["", "| 工具 | 每题次数 | 单次平均 | 单次 P50 |", "|---|---|---|---|"]
        for t, x in v["tools"].items():
            md.append(f"| {t} | {x['per_q']:.2f} | {s1(x['ms'])} | {s1(x['p50'])} |")
    if "replay" in res:
        r = res["replay"]
        md += ["", "## 工具并行执行的收益（缓存回放，配对比较）", "",
               f"- 同一批 {r['n']} 题、并发 1，LLM 全部命中缓存，逐个执行（{', '.join(r['seq_dirs'])}）与并行执行（{', '.join(r['par_dirs'])}）的工具调用轨迹完全相同，各跑两次取每题平均。",
               f"- 每题工具墙钟时间：逐个 {s1(r['seq_tools_wall_ms'])} → 并行 {s1(r['par_tools_wall_ms'])}，每题省 {s1(r['saved_ms'])}（{r['saved_ratio']:.0%}）；每题平均有 {r['multi_call_rounds_per_q']:.2f} 轮一次调用了多个工具。",
               "- GPU 上的检索（编码、rerank）仍由一把锁串行，所以并行主要让图谱查询、读原文、ES 查询与 GPU 检索重叠。"]  # fmt: skip
    if "rerank_gap" in res:
        g = res["rerank_gap"]
        md += ["", "## rerank 偏慢的排查（两次 rerank 之间的空闲时长）", "",
               f"- `{g['dir']}`：{g['n']} 道 dev 题，每题检索出 40 个候选后，按随机顺序在每种空闲时长下各计时一次 rerank（先空闲再计时）。GPU：开始 {g['gpu_before']}，结束 {g['gpu_after']}。", "",
               "| 空闲时长 | P50 | P95 | 平均 |", "|---|---|---|---|"]  # fmt: skip
        for k, x in g["results"].items():
            md.append(f"| {k} s | {x['p50']:.0f} ms | {x['p95']:.0f} ms | {x['mean']:.0f} ms |")
    (rd / f"{args.out}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (rd / f"{args.out}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
