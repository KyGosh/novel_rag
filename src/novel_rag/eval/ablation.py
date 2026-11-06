"""消融表：从 eval/results/ 中的实验记录汇总生成，不手抄数字。

规格文件（如 eval/ablations/v1.json）列出每一行：
    {"group": "融合", "label": "+RRF（BM25-IK + 向量）", "run": "p2a_hybrid", "ref": "p2a_vector"}
run / ref 是实验名称（结果目录名去掉时间前缀），取同名中最新的一次。ref 为该行的对照：
在同一批题上做配对 bootstrap，报告差值及其置信区间。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from novel_rag.eval.runner import TYPE_NAMES, TYPE_ORDER
from novel_rag.eval.stats import paired_diff

HEADLINE = ("recall@5", "recall@20", "mrr", "cov@4000")
DIFF_METRICS = ("recall@5", "recall@20", "mrr", "cov@4000")


def find_run(results_dir: Path, name: str) -> Path:
    cands = sorted(p for p in results_dir.glob(f"*_{name}") if p.is_dir() and p.name.split("_", 1)[1] == name)
    if not cands:
        raise FileNotFoundError(f"{results_dir} 下没有名为 {name} 的实验")
    return cands[-1]


def load_run(path: Path) -> dict[str, Any]:
    return {
        "dir": path.name,
        "run": json.loads((path / "run.json").read_text(encoding="utf-8")),
        "metrics": json.loads((path / "metrics.json").read_text(encoding="utf-8")),
        "rows": [json.loads(x) for x in (path / "per_question.jsonl").read_text(encoding="utf-8").splitlines() if x],
    }


def _paired(a: dict[str, Any], b: dict[str, Any], metric: str, cfg: dict[str, Any]) -> dict[str, float] | None:
    ra = {r["id"]: r["metrics"] for r in a["rows"]}
    rb = {r["id"]: r["metrics"] for r in b["rows"]}
    if set(ra) != set(rb):
        raise ValueError(f"{a['dir']} 与 {b['dir']} 的题目集合不同，不能做配对比较")
    ids = sorted(ra)
    if not all(metric in ra[i] and metric in rb[i] for i in ids):
        return None
    return paired_diff(
        [ra[i][metric] for i in ids],
        [rb[i][metric] for i in ids],
        n_samples=cfg["bootstrap_samples"],
        seed=cfg["bootstrap_seed"],
        level=cfg["ci_level"],
    )


def build_table(spec: dict[str, Any], results_dir: Path, eval_cfg: dict[str, Any]) -> dict[str, Any]:
    cache: dict[str, dict[str, Any]] = {}

    def get(name: str) -> dict[str, Any]:
        if name not in cache:
            cache[name] = load_run(find_run(results_dir, name))
        return cache[name]

    rows = []
    for row in spec["rows"]:
        cur = get(row["run"])
        m = cur["metrics"]
        out = {
            **row,
            "dir": cur["dir"],
            "commit": cur["run"]["git"]["commit"][:10],
            "dirty": cur["run"]["git"]["dirty"],
            "split": cur["run"]["dataset"]["split"],
            "n": m["overall"]["n"],
            "overall": {k: m["overall"].get(k) for k in HEADLINE},
            "ci": {k: m.get("ci", {}).get("overall", {}).get(k) for k in HEADLINE},
            "by_type": {t: {k: m["by_type"].get(t, {}).get(k) for k in HEADLINE} for t in TYPE_ORDER},
            "latency_ms": m.get("latency_ms", {}),
            "rerank_ms": m.get("latency_stages_ms", {}).get("rerank", {}),
            "diff": {},
        }
        if row.get("ref"):
            ref = get(row["ref"])
            out["ref_dir"] = ref["dir"]
            for k in DIFF_METRICS:
                d = _paired(ref, cur, k, eval_cfg)
                if d is not None:
                    out["diff"][k] = d
        rows.append(out)
    return {"title": spec.get("title", ""), "notes": spec.get("notes", []), "rows": rows, "eval": eval_cfg}


def _f(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


def _ci(v: float | None, ci: list[float] | None) -> str:
    if v is None:
        return "–"
    return f"{v:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{v:.3f}"


def _diff(d: dict[str, float] | None) -> str:
    if not d:
        return "–"
    mark = "↑" if d["lo"] > 0 else ("↓" if d["hi"] < 0 else "")
    return f"{d['diff']:+.3f} [{d['lo']:+.2f}, {d['hi']:+.2f}]{mark}"


def render_markdown(table: dict[str, Any]) -> str:
    level = int(round(table["eval"]["ci_level"] * 100))
    lines = [f"# {table['title']}", ""]
    lines += [f"- {n}" for n in table["notes"]]
    lines += [
        f"- 方括号为 {level}% bootstrap 置信区间（按题重采样 {table['eval']['bootstrap_samples']} 次）；"
        "差值列是相对于对照行的配对差值，↑/↓ 表示区间不含 0。",
        "",
        "## 总体",
        "",
        "| 组 | 配置 | 题数 | R@5 | R@20 | MRR | cov@4000 | 延迟 P50/P95 ms | rerank P50 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table["rows"]:
        o, ci, lat = r["overall"], r["ci"], r["latency_ms"]
        lines.append(
            f"| {r['group']} | {r['label']} | {r['n']} | "
            + " | ".join(_ci(o[k], ci[k]) for k in HEADLINE)
            + f" | {lat.get('p50', '–')}/{lat.get('p95', '–')} | {r['rerank_ms'].get('p50', '–')} |"
        )
    lines += [
        "",
        "## 相对对照行的配对差值",
        "",
        "| 配置 | 对照 | ΔR@5 | ΔR@20 | ΔMRR | Δcov@4000 |",
        "|---|---|---|---|---|---|",
    ]
    labels = {r["run"]: r["label"] for r in table["rows"]}
    for r in table["rows"]:
        if r.get("ref"):
            lines.append(
                f"| {r['label']} | {labels.get(r['ref'], r['ref'])} | "
                + " | ".join(_diff(r["diff"].get(k)) for k in DIFF_METRICS)
                + " |"
            )
    for metric, title in (("recall@20", "R@20"), ("cov@4000", "cov@4000"), ("recall@5", "R@5")):
        lines += ["", f"## 按题型：{title}", "", "| 配置 | " + " | ".join(TYPE_NAMES[t] for t in TYPE_ORDER) + " |"]
        lines.append("|---" * (len(TYPE_ORDER) + 1) + "|")
        for r in table["rows"]:
            lines.append(f"| {r['label']} | " + " | ".join(_f(r["by_type"][t][metric]) for t in TYPE_ORDER) + " |")
    lines += ["", "## 实验记录", "", "| 配置 | 结果目录 | commit | 划分 |", "|---|---|---|---|"]
    for r in table["rows"]:
        dirty = "（dirty）" if r["dirty"] else ""
        lines.append(f"| {r['label']} | `{r['dir']}` | `{r['commit']}`{dirty} | {r['split']} |")
    return "\n".join(lines) + "\n"
