# ruff: noqa: E501 —— 含较长的中文表头与说明文字
"""模型漂移检查（P5 方案第七节）：比较若干次问答运行在同一批题上的判分与工具调用模式。

代理背后的模型可能在服务端更新，跨天的运行就不可比了。做法：阶段开始和结束时各用固定采样编号、关闭缓存跑一次
冒烟子集，再和已有的参照运行放在一起两两比较。同一模型的两次独立采样本来就有差异，所以要有参照对（例如 P4 的两次
dev 运行在同一批题上的差异）来判断"差异是否超出正常的运行间噪声"。

用法：
    uv run python scripts/drift_check.py --name p5_start \
        --run p4_s0=eval/results/..._p4_agent_dev_s0 --run p4_s1=eval/results/..._p4_agent_dev_s1 \
        --run start=eval/results/..._p5_drift_start
只比较所有运行共有的题（冒烟子集 24 题）。输出 eval/results/{时间}_drift_{name}/。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from novel_rag.config import get_settings

TOOLS = ("search_text", "read_text", "search_summaries", "read_summary", "find_entity", "graph_query")


def load(d: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = [json.loads(x) for x in (d / "per_question.jsonl").read_text(encoding="utf-8").splitlines() if x]
    return {r["id"]: r for r in rows}, json.loads((d / "run.json").read_text(encoding="utf-8"))


def profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tc: Counter = Counter()
    for r in rows:
        tc.update((r.get("agent") or {}).get("tool_counts", {}))
    n = len(rows)
    return {
        "acc": mean(r["metrics"]["acc"] for r in rows),
        "acc_lenient": mean(r["metrics"]["acc_lenient"] for r in rows),
        "tokens": mean(r["metrics"]["tokens"] for r in rows),
        "llm_calls": mean(r["metrics"]["llm_calls"] for r in rows),
        "tool_calls": mean(r["metrics"]["tool_calls"] for r in rows),
        "rounds": mean((r.get("agent") or {}).get("rounds", 0) for r in rows),
        "tools_per_question": {t: round(tc.get(t, 0) / n, 2) for t in TOOLS},
        "termination": dict(Counter(r.get("termination") for r in rows)),
    }


def first_tools(r: dict[str, Any]) -> tuple[str, ...]:
    """第一轮调用的工具名（排序后）：规划与第一步行动是否走同一条路。"""
    return tuple(sorted(s["tool"] for s in (r.get("agent") or {}).get("trace", [])
                        if s.get("step") == "tool" and s.get("round") == 1))  # fmt: skip


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="名称=结果目录（至少两个）")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    s = get_settings()
    runs = {k: load(Path(v)) for k, v in (x.split("=", 1) for x in args.run)}
    ids = sorted(set.intersection(*(set(r[0]) for r in runs.values())))
    R = {k: [rows[i] for i in ids] for k, (rows, _) in runs.items()}
    prof = {k: profile(v) for k, v in R.items()}
    pairs = {}
    for a, b in combinations(R, 2):
        pairs[f"{a} vs {b}"] = {
            "label_agreement": mean(float(x["label"] == y["label"]) for x, y in zip(R[a], R[b], strict=True)),
            "correct_agreement": mean(
                float((x["label"] == "correct") == (y["label"] == "correct")) for x, y in zip(R[a], R[b], strict=True)
            ),  # fmt: skip
            "first_round_same_tools": mean(
                float(first_tools(x) == first_tools(y)) for x, y in zip(R[a], R[b], strict=True)
            ),
            "acc_diff": prof[b]["acc"] - prof[a]["acc"],
            "tokens_ratio": prof[b]["tokens"] / prof[a]["tokens"],
            "tool_calls_ratio": prof[b]["tool_calls"] / prof[a]["tool_calls"],
        }
    out = {
        "name": args.name, "created": datetime.now().isoformat(timespec="seconds"), "n": len(ids), "ids": ids,
        "runs": {k: {"dir": Path(v).name, "created": runs[k][1]["created"], "sample": runs[k][1].get("sample"),
                     "llm_cache": runs[k][1].get("llm_cache"), "model": runs[k][1]["system"].get("models")}
                 for k, v in (x.split("=", 1) for x in args.run)},
        "profiles": prof, "pairs": pairs,
    }  # fmt: skip
    md = [f"# 模型漂移检查：{args.name}（{len(ids)} 题）", ""]
    md += [f"- {k} = `{v['dir']}`（{v['created']}，采样编号 {v['sample']}，LLM 缓存 {'开' if v['llm_cache'] else '关'}）"
           for k, v in out["runs"].items()]  # fmt: skip
    md += ["", "## 各次运行的概况", "", "| 运行 | 准确率 | 含部分对 | LLM 调用 | 工具调用 | 轮数 | token | "
           + " | ".join(TOOLS) + " | 终止原因 |", "|---" * (8 + len(TOOLS)) + "|"]  # fmt: skip
    for k, p in prof.items():
        md.append(f"| {k} | {p['acc']:.3f} | {p['acc_lenient']:.3f} | {p['llm_calls']:.2f} | {p['tool_calls']:.2f} | "
                  f"{p['rounds']:.2f} | {p['tokens']:.0f} | "
                  + " | ".join(f"{p['tools_per_question'][t]:.2f}" for t in TOOLS) + f" | {p['termination']} |")  # fmt: skip
    md += ["", "## 两两比较", "", "| 对比 | 逐题判分一致率 | 对 / 不对一致率 | 第一轮工具组合相同的题 | 准确率差 | token 比 | 工具调用比 |",
           "|---|---|---|---|---|---|---|"]  # fmt: skip
    for k, p in pairs.items():
        md.append(f"| {k} | {p['label_agreement']:.3f} | {p['correct_agreement']:.3f} | {p['first_round_same_tools']:.3f} | "
                  f"{p['acc_diff']:+.3f} | {p['tokens_ratio']:.2f} | {p['tool_calls_ratio']:.2f} |")  # fmt: skip
    d = s.paths.results_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_drift_{args.name}"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (d / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n结果目录：{d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
