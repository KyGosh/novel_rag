# ruff: noqa: E501 —— 含较长的中文表头与说明文字
"""Agent 消融表（P5）：从 eval/results/ 的问答运行结果汇总，不手抄数字。

规格文件（eval/ablations/qa_v1.json）：
    {"title": ..., "notes": [...],
     "default": {"label": "默认配置", "runs": ["p4_agent_dev_s0", "p4_agent_dev_s1"]},
     "rows": [{"group": "图谱", "label": "去掉全部图谱", "runs": ["p5_abl_nograph_s0", "p5_abl_nograph_s1"]}, ...]}
runs 是实验名称（结果目录名去掉时间前缀，取同名中最新的一次）；同一变体的多次采样先按题取平均，再与默认配置
（同样按题取平均）在同一批题上做配对 bootstrap。默认配置各次采样之间的差值作为噪声参照。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from novel_rag.eval.ablation import find_run, load_run
from novel_rag.eval.runner import TYPE_NAMES, TYPE_ORDER
from novel_rag.eval.stats import bootstrap_ci, paired_diff

GROUPS = (*TYPE_ORDER, "overall")
GNAME = {**TYPE_NAMES, "overall": "总体"}
ACC = ("acc", "acc_lenient")
EVID = ("ctx_recall", "process_cov", "note_recall", "cite_recall")
COST = ("llm_calls", "tool_calls", "tokens")


def per_question(runs: list[dict[str, Any]], metric: str) -> dict[str, float]:
    """同一变体多次采样：每题取各次采样的平均（某次没有该指标则跳过；都没有的题不出现）。"""
    out: dict[str, float] = {}
    ids = set.intersection(*(set(r["id"] for r in run["rows"]) for run in runs))
    for i in ids:
        vals = [r["metrics"].get(metric) for run in runs for r in run["rows"] if r["id"] == i]
        vals = [v for v in vals if v is not None]
        if vals:
            out[i] = mean(vals)
    return out


def _types(runs: list[dict[str, Any]]) -> dict[str, str]:
    return {r["id"]: r["type"] for r in runs[0]["rows"]}


def _ids(types: dict[str, str], group: str) -> list[str]:
    return sorted(i for i, t in types.items() if group == "overall" or t == group)


def summarize_variant(runs: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    types = _types(runs)
    out: dict[str, Any] = {"n_samples": len(runs), "dirs": [r["dir"] for r in runs], "groups": {}}
    out["commits"] = [r["run"]["git"]["commit"][:10] + ("（dirty）" if r["run"]["git"]["dirty"] else "") for r in runs]
    out["dates"] = [r["run"]["created"][:16] for r in runs]
    for g in GROUPS:
        ids = _ids(types, g)
        rec: dict[str, Any] = {"n": len(ids)}
        for m in (*ACC, *EVID, *COST):
            pq = per_question(runs, m)
            vals = [pq[i] for i in ids if i in pq]
            if not vals:
                continue
            rec[m] = mean(vals)
            if m in ACC:
                lo, hi = bootstrap_ci(vals, n_samples=cfg["bootstrap_samples"], seed=cfg["bootstrap_seed"],
                                      level=cfg["ci_level"])  # fmt: skip
                rec[f"{m}_ci"] = [lo, hi]
        out["groups"][g] = rec
    term: Counter = Counter()
    for run in runs:
        term.update(r.get("termination") for r in run["rows"])
    out["termination"] = {k: round(v / len(runs), 1) for k, v in term.items()}
    out["sample_acc"] = [mean(r["metrics"]["acc"] for r in run["rows"]) for run in runs]
    return out


def paired(
    ref: list[dict[str, Any]], cur: list[dict[str, Any]], metric: str, group: str, cfg: dict[str, Any]
) -> dict[str, float] | None:
    """cur 相对 ref 的配对差值（两边都先按题对采样取平均）。"""
    a, b = per_question(ref, metric), per_question(cur, metric)
    ids = [i for i in _ids(_types(ref), group) if i in a and i in b]
    if not ids:
        return None
    return paired_diff([a[i] for i in ids], [b[i] for i in ids], n_samples=cfg["bootstrap_samples"],
                       seed=cfg["bootstrap_seed"], level=cfg["ci_level"])  # fmt: skip


def build_table(spec: dict[str, Any], results_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    def get(names: list[str]) -> list[dict[str, Any]]:
        return [load_run(find_run(results_dir, n)) for n in names]

    default = get(spec["default"]["runs"])
    ids0 = {r["id"] for r in default[0]["rows"]}
    table: dict[str, Any] = {"title": spec.get("title", ""), "notes": spec.get("notes", []), "eval": cfg}
    table["default"] = {"label": spec["default"]["label"], **summarize_variant(default, cfg)}
    # 噪声参照：默认配置的各次采样两两之间的差值
    noise = []
    for x in range(len(default)):
        for y in range(x + 1, len(default)):
            d = {g: paired([default[x]], [default[y]], "acc", g, cfg) for g in GROUPS}
            agree = mean(float(a["label"] == b["label"]) for a, b in zip(
                sorted(default[x]["rows"], key=lambda r: r["id"]), sorted(default[y]["rows"], key=lambda r: r["id"]),
                strict=True))  # fmt: skip
            noise.append({"pair": [default[x]["dir"], default[y]["dir"]], "diff": d, "label_agreement": agree})
    table["noise"] = noise
    rows = []
    for row in spec["rows"]:
        cur = get(row["runs"])
        if {r["id"] for r in cur[0]["rows"]} != ids0:
            raise ValueError(f"{row['label']} 与默认配置的题目集合不同，不能做配对比较")
        rec = {**row, **summarize_variant(cur, cfg), "diff": {}, "diff_lenient": {}, "diff_evid": {}}
        for g in GROUPS:
            rec["diff"][g] = paired(default, cur, "acc", g, cfg)
            rec["diff_lenient"][g] = paired(default, cur, "acc_lenient", g, cfg)
        for m in EVID:
            rec["diff_evid"][m] = paired(default, cur, m, "overall", cfg)
        base = table["default"]["groups"]["overall"]
        rec["cost_ratio"] = {m: rec["groups"]["overall"][m] / base[m] for m in COST if base.get(m)}
        rows.append(rec)
    table["rows"] = rows
    return table


def _d(d: dict[str, float] | None) -> str:
    if not d:
        return "–"
    mark = "↑" if d["lo"] > 0 else ("↓" if d["hi"] < 0 else "")
    return f"{d['diff']:+.3f} [{d['lo']:+.2f}, {d['hi']:+.2f}]{mark}"


def _ci(rec: dict[str, Any], m: str = "acc") -> str:
    if m not in rec:
        return "–"
    ci = rec.get(f"{m}_ci")
    return f"{rec[m]:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{rec[m]:.3f}"


def render_markdown(table: dict[str, Any]) -> str:
    level = int(round(table["eval"]["ci_level"] * 100))
    all_rows = [table["default"], *table["rows"]]
    L = [f"# {table['title']}", ""]
    L += [f"- {n}" for n in table["notes"]]
    L += [
        f"- 方括号为 {level}% bootstrap 置信区间（按题重采样 {table['eval']['bootstrap_samples']} 次）。同一变体有多次采样时，"
        "每题先对各次采样取平均（一题的准确率可以是 0.5），再对题目求均值。",
        "- 差值 = 变体 − 默认配置（默认配置取两次运行的按题平均），同一批题配对；↑/↓ 表示区间不含 0。",
        "",
        "## 1. 答案准确率（correct 比例）",
        "",
        "| 组 | 变体 | 采样次数 | "
        + " | ".join(f"{GNAME[g]} (n={table['default']['groups'][g]['n']})" for g in GROUPS)
        + " |",
        "|---" * (len(GROUPS) + 3) + "|",
    ]
    for r in all_rows:
        L.append(f"| {r.get('group', '—')} | {r['label']} | {r['n_samples']} | "
                 + " | ".join(_ci(r["groups"][g]) for g in GROUPS) + " |")  # fmt: skip
    L += ["", "## 2. 相对默认配置的配对差值（准确率）", "",
          "| 组 | 变体 | " + " | ".join(GNAME[g] for g in GROUPS) + " |", "|---" * (len(GROUPS) + 2) + "|"]  # fmt: skip
    for n in table["noise"]:
        L.append(
            "| 噪声参照 | 默认配置：第二次运行 − 第一次运行 | " + " | ".join(_d(n["diff"][g]) for g in GROUPS) + " |"
        )
    for r in table["rows"]:
        L.append(f"| {r['group']} | {r['label']} | " + " | ".join(_d(r["diff"][g]) for g in GROUPS) + " |")
    if table["noise"]:
        L += ["", "噪声参照是同一配置两次独立运行之间的差值（逐题判分一致率 "
              + "、".join(f"{n['label_agreement']:.3f}" for n in table["noise"])
              + "）。只跑一次的变体，其差值里含有同样量级的运行间噪声。"]  # fmt: skip
    L += ["", "## 3. 含“部分对”的准确率（correct + partial）及其差值", "",
          "| 变体 | 总体 | 差值（总体） | 多跳 | 差值（多跳） | 跨章节情节 | 差值（跨章节） |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    d0 = table["default"]["groups"]
    L.append(f"| {table['default']['label']} | {d0['overall']['acc_lenient']:.3f} | – | "
             f"{d0['multihop']['acc_lenient']:.3f} | – | {d0['plot']['acc_lenient']:.3f} | – |")  # fmt: skip
    for r in table["rows"]:
        g = r["groups"]
        L.append(f"| {r['label']} | {g['overall']['acc_lenient']:.3f} | {_d(r['diff_lenient']['overall'])} | "
                 f"{g['multihop']['acc_lenient']:.3f} | {_d(r['diff_lenient']['multihop'])} | "
                 f"{g['plot']['acc_lenient']:.3f} | {_d(r['diff_lenient']['plot'])} |")  # fmt: skip
    L += ["", "## 4. 证据指标（总体；差值相对默认配置）", "",
          "ctx_recall = 作答上下文（4000 token）覆盖的金标比例；process_cov = 过程中看到过原文的金标比例；"
          "note_recall = 进了笔记的金标比例；cite_recall = 被答案引用的金标比例。", "",
          "| 变体 | " + " | ".join(EVID) + " | Δctx_recall | Δprocess_cov | Δnote_recall |", "|---" * (len(EVID) + 4) + "|"]  # fmt: skip
    for r in all_rows:
        o = r["groups"]["overall"]
        dv = r.get("diff_evid", {})
        L.append(f"| {r['label']} | " + " | ".join(f"{o[m]:.3f}" if m in o else "–" for m in EVID)
                 + f" | {_d(dv.get('ctx_recall'))} | {_d(dv.get('process_cov'))} | {_d(dv.get('note_recall'))} |")  # fmt: skip
    L += ["", "## 5. 成本与终止原因（每题平均；倍数相对默认配置）", "",
          "| 变体 | LLM 调用 | 工具调用 | token | token 倍数 | 终止原因（每次运行平均题数） |", "|---|---|---|---|---|---|"]  # fmt: skip
    for r in all_rows:
        o = r["groups"]["overall"]
        ratio = r.get("cost_ratio", {}).get("tokens")
        term = "、".join(f"{k} {v:g}" for k, v in sorted(r["termination"].items(), key=lambda x: -x[1]))
        L.append(f"| {r['label']} | {o['llm_calls']:.2f} | {o['tool_calls']:.2f} | {o['tokens']:.0f} | "
                 f"{f'{ratio:.2f}×' if ratio else '–'} | {term} |")  # fmt: skip
    L += [
        "",
        "## 6. 实验记录",
        "",
        "| 变体 | 结果目录 | commit | 时间 | 各次采样的总体准确率 |",
        "|---|---|---|---|---|",
    ]
    for r in all_rows:
        L.append(f"| {r['label']} | " + "<br>".join(f"`{d}`" for d in r["dirs"]) + " | " + "<br>".join(r["commits"])
                 + " | " + "<br>".join(r["dates"]) + " | " + " / ".join(f"{a:.3f}" for a in r["sample_acc"]) + " |")  # fmt: skip
    return "\n".join(L) + "\n"
