"""缓存回放一致性检查（P6 B1）：行为不变的工程改动之后，从 LLM 缓存回放一次已有的运行，逐题比较。

比较项（逐题）：简短答案、判分、引用、终止原因、token 用量，以及工具调用轨迹（每一轮调用了哪些工具、参数是什么、
是否成功、展示了哪些 chunk）。全部相同才算"行为不变"。延迟不比较。

用法：
    uv run python scripts/replay_check.py --name b1_dev \
        --base eval/results/20261112-005605_p4_agent_dev_s0 --new eval/results/..._p6_replay_s0
只比较两次运行共有的题。输出 eval/results/{时间}_replay_{name}/（summary.md、metrics.json）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from novel_rag.config import get_settings


def load_rows(d: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(x) for x in (d / "per_question.jsonl").read_text(encoding="utf-8").splitlines() if x]
    return {r["id"]: r for r in rows}


def trajectory(row: dict[str, Any]) -> list[tuple[Any, ...]]:
    """工具调用轨迹：按 trace 顺序的 (轮次, 工具, 参数, 是否成功, 展示的 chunk)。"""
    out = []
    for st in (row.get("agent") or {}).get("trace", []):
        if st.get("step") == "tool":
            args = json.dumps(st.get("args"), ensure_ascii=False, sort_keys=True)
            out.append((st.get("round"), st.get("tool"), args, st.get("ok"), tuple(st.get("shown") or [])))
    return out


FIELDS = {
    "answer": lambda r: r.get("answer"),
    "label": lambda r: r.get("label"),
    "citations": lambda r: r.get("citations"),
    "termination": lambda r: r.get("termination"),
    "tokens": lambda r: (r.get("metrics") or {}).get("tokens"),
    "llm_calls": lambda r: (r.get("metrics") or {}).get("llm_calls"),
    "trajectory": trajectory,
    "error": lambda r: r.get("error"),
}


def compare(base: dict[str, dict[str, Any]], new: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids = sorted(set(base) & set(new))
    diffs: dict[str, list[str]] = {k: [] for k in FIELDS}
    for i in ids:
        for k, f in FIELDS.items():
            if f(base[i]) != f(new[i]):
                diffs[k].append(i)
    return {
        "n": len(ids),
        "identical": all(not v for v in diffs.values()),
        "mismatch_counts": {k: len(v) for k, v in diffs.items()},
        "mismatch_ids": {k: v for k, v in diffs.items() if v},
        "new_errors": sum(1 for i in ids if new[i].get("error")),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="参照运行的结果目录")
    ap.add_argument("--new", required=True, help="回放运行的结果目录")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    base, new = Path(args.base), Path(args.new)
    res = {"name": args.name, "created": datetime.now().isoformat(timespec="seconds"), "base": base.name,
           "new": new.name, **compare(load_rows(base), load_rows(new))}  # fmt: skip
    md = [
        f"# 缓存回放一致性：{args.name}",
        "",
        f"- 参照 `{base.name}`，回放 `{new.name}`，共有 {res['n']} 题",
        f"- 结论：**{'逐题完全一致' if res['identical'] else '存在差异'}**；回放出错 {res['new_errors']} 题",
        "",
        "| 比较项 | 不一致的题数 |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in res["mismatch_counts"].items()],
    ]
    d = get_settings().paths.results_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_replay_{args.name}"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (d / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n结果目录：{d}")
    return 0 if res["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
