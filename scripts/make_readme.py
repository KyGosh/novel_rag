# ruff: noqa: E501 —— 含较长的中文表格与说明文字
"""README 的表格、数字和图（P6 A 线）：全部从 eval/results/ 的结果文件生成，不手抄。

README.md 里形如
    <!-- BEGIN GENERATED: 名称 -->
    ……
    <!-- END GENERATED: 名称 -->
的区块由本脚本整体替换；区块外的文字手写，不含具体数字。图写到 assets/（matplotlib 是开发依赖）。

用法：
    uv run python scripts/make_readme.py            # 重写 README 中的生成区块和 assets/*.png
    uv run python scripts/make_readme.py --check    # 只检查 README 是否与结果文件一致（不一致退出码 1）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from novel_rag.config import PROJECT_ROOT

RD = PROJECT_ROOT / "eval" / "results"
ASSETS = PROJECT_ROOT / "assets"
README = PROJECT_ROOT / "README.md"
TYPES = ("fact", "relation", "multihop", "plot")
TNAME = {"fact": "事实", "relation": "人物关系", "multihop": "多跳", "plot": "跨章节情节", "overall": "总体"}
SYS = {"closed": "闭卷", "rag": "固定 RAG", "graphrag": "固定 GraphRAG", "agent": "Agent"}
# 参考调色板（dataviz skill，浅色）：分类色按固定顺序
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def J(name: str) -> dict[str, Any]:
    return json.loads((RD / name).read_text(encoding="utf-8"))


def f3(x: float) -> str:
    return f"{x:.3f}"


def d2(p: dict[str, float]) -> str:
    return f"{p['diff']:+.3f} [{p['lo']:+.2f}, {p['hi']:+.2f}]"


# ---------------------------------------------------------------- 数据


def test_results() -> dict[str, Any]:
    d = J("p5_final_comparison.json")
    acc = {s: {g: d["accuracy"][s][g]["acc"] for g in (*TYPES, "overall")} for s in ("closed", "rag", "graphrag")}
    acc["agent"] = {g: mean(d["accuracy"][k][g]["acc"] for k in ("agent", "agent_s1")) for g in (*TYPES, "overall")}
    samples = [d["accuracy"][k]["overall"]["acc"] for k in ("agent", "agent_s1")]
    cost = {s: d["cost"][s] for s in ("closed", "rag", "graphrag")}
    cost["agent"] = {
        k: mean(d["cost"][a][k] for a in ("agent", "agent_s1")) for k in ("llm_calls", "tool_calls", "tokens")
    }
    n = {g: None for g in (*TYPES, "overall")}
    head = (RD / "p5_final_comparison.md").read_text(encoding="utf-8")
    for g, label in (*((t, TNAME[t]) for t in TYPES), ("overall", "总体")):
        m = re.search(rf"{label} \(n=(\d+)\)", head)
        n[g] = int(m.group(1)) if m else None
    return {"raw": d, "acc": acc, "samples": samples, "cost": cost, "n": n}


def latency() -> dict[str, Any] | None:
    p = RD / "p6_latency.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------------------------------------------------------------- 区块


def block_main(t: dict[str, Any], lat: dict[str, Any] | None) -> str:
    d = t["raw"]
    lines = ["| 系统 | " + " | ".join(f"{TNAME[g]}（{t['n'][g]}）" for g in (*TYPES, "overall")) + " | 每题 LLM 调用 | 每题 token |",
             "|---" * 8 + "|"]  # fmt: skip
    for s in ("closed", "rag", "graphrag", "agent"):
        a, c = t["acc"][s], t["cost"][s]
        name = SYS[s] + ("（两次采样平均）" if s == "agent" else "")
        cells = [f"**{f3(a[g])}**" if s == "agent" else f3(a[g]) for g in (*TYPES, "overall")]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {c['llm_calls']:.1f} | {c['tokens']:,.0f} |")
    lines += ["", f"Agent 两次采样的总体准确率：{' / '.join(f3(x) for x in t['samples'])}（逐题判分一致率 {d['agent_runs']['label_agreement']:.3f}）。"
              "准确率 = judge 判为完全正确的比例（部分正确不计）；95% 置信区间、含部分正确的口径见 "
              "[`eval/results/p5_final_comparison.md`](eval/results/p5_final_comparison.md)。"]  # fmt: skip
    lines += ["", "**Agent 相对固定 RAG 的配对差值**（差值 [95% bootstrap CI]）：", "",
              "| 采样 | " + " | ".join(TNAME[g] for g in (*TYPES, "overall")) + " |", "|---" * 6 + "|"]  # fmt: skip
    for k, label in (("agent", "第一次"), ("agent_s1", "第二次")):
        lines.append(f"| {label} | " + " | ".join(d2(d["paired_vs_ref"][k][g]) for g in (*TYPES, "overall")) + " |")
    return "\n".join(lines)


def block_latency(lat: dict[str, Any] | None) -> str:
    if lat is None:
        d = J("p4_comparison.json")["latency"]
        lines = ["| 系统 | P50 | P95 |", "|---|---|---|"]
        for s in ("closed", "rag", "agent"):
            lines.append(f"| {SYS[s]} | {d[s]['total']['p50'] / 1000:.1f} s | {d[s]['total']['p95'] / 1000:.1f} s |")
        return "\n".join(lines) + "\n\n（P4 测量，`eval/results/p4_comparison.md`）"
    lines = ["| 系统 | 平均 | P50 | P95 | 每题 LLM 调用 | 其中 LLM 耗时 |", "|---|---|---|---|---|---|"]
    for label, v in lat["systems"].items():
        lines.append(f"| {label} | {v['total']['mean'] / 1000:.1f} s | {v['total']['p50'] / 1000:.1f} s | {v['total']['p95'] / 1000:.1f} s | "
                     f"{v['llm_calls']:.1f} | {v.get('llm', 0) / 1000:.1f} s |")  # fmt: skip
    agent = next((v for k, v in lat["systems"].items() if "stages" in v), None)
    if agent:
        st = agent["stages"]
        lines += [
            "",
            "Agent 每题的时间花在哪里：",
            "",
            "| 环节 | 每题次数 | 单次平均 | 每题合计 |",
            "|---|---|---|---|",
        ]
        names = {
            "plan": "规划（LLM）",
            "act": "行动（LLM）",
            "reflect": "反思（LLM）",
            "answer": "作答（LLM）",
            "tool": "工具执行",
        }
        for k in ("plan", "act", "reflect", "answer", "tool"):
            if k in st:
                lines.append(
                    f"| {names[k]} | {st[k]['per_q']:.2f} | {st[k]['ms'] / 1000:.2f} s | {st[k]['per_q'] * st[k]['ms'] / 1000:.1f} s |"
                )
    if "replay" in lat:
        r = lat["replay"]
        lines += ["", f"同一轮内的工具调用改为并行执行：缓存回放下（轨迹完全相同）每题工具墙钟时间 {r['seq_tools_wall_ms'] / 1000:.2f} s → "
                  f"{r['par_tools_wall_ms'] / 1000:.2f} s，没有可测的收益——检索时间主要是 GPU 上的编码和 rerank，单卡上只能串行。"]  # fmt: skip
    return "\n".join(lines) + "\n\n（测量口径与明细：[`eval/results/p6_latency.md`](eval/results/p6_latency.md)）"


def block_findings(t: dict[str, Any]) -> str:
    d = t["raw"]
    abl = J("ablation_qa_v1.json")
    rows = {r["label"]: r for r in abl["rows"]}
    v0 = J("p5_v0_comparison.json")
    cs = d["closed_subsets"]
    ret = {r["run"]: r for r in J("ablation_v1.json")["rows"]}
    fe = J("p6_route_feasibility.json")

    def ab(label_part: str) -> dict[str, Any]:
        return next(r for k, r in rows.items() if label_part in k)

    nog, nosum, low = ab("去掉全部图谱"), ab("去掉摘要"), ab("预算低档")
    raw = ab("原始历史")
    sel, ver = fe["signals"]["selfcheck"]["by_group"]["overall"], fe["signals"]["verify"]["by_group"]["overall"]
    lines = [
        f"1. **收益来自多轮检索本身，集中在多跳和跨章节题。** test 上 Agent 相对固定 RAG：多跳 {d2(d['paired_vs_ref']['agent']['multihop'])} / "
        f"{d2(d['paired_vs_ref']['agent_s1']['multihop'])}，跨章节 {d2(d['paired_vs_ref']['agent']['plot'])} / {d2(d['paired_vs_ref']['agent_s1']['plot'])}；"
        f"事实题和人物关系题持平（固定 RAG 已接近满分）。",
        f"2. **提升不是来自模型自己读过这套书。** 闭卷基线 test 准确率 {f3(t['acc']['closed']['overall'])}；在闭卷没答对的多跳 + 跨章节题（{cs['agent']['n'][2]} 题）上，"
        f"固定 RAG {f3(cs['rag']['closed_wrong_hard'])}，Agent {f3(cs['agent']['closed_wrong_hard'])} / {f3(cs['agent_s1']['closed_wrong_hard'])}。",
        f"3. **图谱和摘要工具没有带来可测的准确率提升**（负面结果）。dev 消融：去掉全部图谱 {d2(nog['diff']['overall'])}（token {nog['cost_ratio']['tokens']:.2f}×），"
        f"去掉摘要 {d2(nosum['diff']['overall'])}（token {nosum['cost_ratio']['tokens']:.2f}×）；固定 GraphRAG 相对固定 RAG 在 test 上 {d2(d['paired_vs_ref']['graphrag']['overall'])}。"
        "图谱工具按用户决定保留在默认配置里，但准确率的提升不能归功于它。",
        f"4. **预算有下限，记忆方式省 token。** 预算减半 {d2(low['diff']['overall'])}（显著变差）；行动环节改看原始工具输出（不用证据笔记）准确率 {d2(raw['diff']['overall'])}，"
        f"token {raw['cost_ratio']['tokens']:.2f}×——只看证据笔记更省。",
        f"5. **检索链路**（P2，v0 dev）：RRF 融合相对纯向量 cov@4000 {d2(ret['p2a_rrf_ik']['diff']['cov@4000'])}，再加 rerank R@5 {d2(ret['p2a_rr40']['diff']['recall@5'])}；"
        f"small-to-big（±1 邻块）在固定 4000 token 预算下 cov@4000 {d2(ret['p2a_rr40_s2b1']['diff']['cov@4000'])}，所以固定流程不扩展、按需读原文交给 Agent 的工具。",
        f"6. **按需升级没有做成**（负面结果）。想法是先走固定 RAG、证据不足才启动 Agent。事先定的门槛是路由对“固定 RAG 没答对的题”的召回率 ≥ 0.75：独立核对 {f3(ver['recall'])}、"
        f"作答自评 {f3(sel['recall'])}，都不达标，按规则放弃，没有跑 dev 三次采样和 test（[`p6_route_feasibility.md`](eval/results/p6_route_feasibility.md)）。",
        f"7. **v0 佐证**（120 题、人工确认过的评测集）：闭卷 {f3(v0['accuracy']['closed']['overall']['acc'])}、固定 RAG {f3(v0['accuracy']['rag']['overall']['acc'])}、"
        f"Agent {f3(v0['accuracy']['agent']['overall']['acc'])}，方向与 v1 一致。",
    ]
    return "\n".join(lines)


def block_ablation() -> str:
    a = J("ablation_qa_v1.json")
    dft = a["default"]
    lines = [f"默认配置三次 dev 运行：{' / '.join(f3(x) for x in dft['sample_acc'])}（平均 {f3(dft['groups']['overall']['acc'])}，每题 {dft['groups']['overall']['tokens']:,.0f} token）。", "",
             "| 变体 | 采样次数 | 准确率 | 相对默认配置 [95% CI] | 多跳 / 跨章节相对默认 | token |", "|---|---|---|---|---|---|"]  # fmt: skip
    for r in a["rows"]:
        hard = r["diff"].get("hard") or {}
        hd = f"{hard['diff']:+.3f}" if hard else "—"
        if not hard:
            ps = [r["diff"].get(t) for t in ("multihop", "plot")]
            hd = " / ".join(f"{p['diff']:+.3f}" for p in ps if p) if all(ps) else "—"
        lines.append(
            f"| {r['label']} | {r['n_samples']} | {f3(r['groups']['overall']['acc'])} | {d2(r['diff']['overall'])} | {hd} | {r['cost_ratio']['tokens']:.2f}× |"
        )
    return "\n".join(lines) + "\n\n（[`eval/results/ablation_qa_v1.md`](eval/results/ablation_qa_v1.md)）"


def block_failures() -> str:
    md = (RD / "p5_failure_analysis.md").read_text(encoding="utf-8")
    m = re.search(r"\| 主要原因 \| 题数 \| 说明 \|\n(?:\|.*\|\n)+", md)
    if not m:
        raise RuntimeError("p5_failure_analysis.md 里找不到失败原因表")
    table = m.group(0).strip().splitlines()
    out = [re.sub(r"\| ([^|]*) \|$", "|", x) if i else "| 主要原因 | 题数 |" for i, x in enumerate(table)]
    out[1] = "|---|---|"
    return ("Agent 在 dev 上至少一次没答对的 28 题（Claude 归因，未经人工复核）：\n\n" + "\n".join(out)
            + "\n\n（[`eval/results/p5_failure_analysis.md`](eval/results/p5_failure_analysis.md)）")  # fmt: skip


def block_eval_meta(t: dict[str, Any]) -> str:
    ja = json.loads((RD / "20261112-175041_judge_agreement" / "metrics.json").read_text(encoding="utf-8"))
    ledger = [json.loads(x) for x in (RD / "test_runs.jsonl").read_text(encoding="utf-8").splitlines() if x]
    commits = sorted({x["commit"][:7] for x in ledger})
    return "\n".join([
        f"- judge 与人工的一致率：dev 上分层抽 {ja['overall']['n']} 条，一致率 {ja['overall']['agreement']:.2f}，Cohen's κ {ja['overall']['kappa']:.2f}；"
        f"跨章节题的“对 / 部分对”边界最不稳。test 上没有做人工抽查。",
        f"- test 台账（[`eval/results/test_runs.jsonl`](eval/results/test_runs.jsonl)）：共 {len(ledger)} 次 test 运行，全部来自配置冻结后的同一次确认（commit {', '.join(commits)}）。",
        f"- Agent 两次运行之间的噪声：test 上总体差 {d2(t['raw']['agent_runs']['by_group']['overall'])}，逐题判分一致率 {t['raw']['agent_runs']['label_agreement']:.3f}。",
    ])  # fmt: skip


# ---------------------------------------------------------------- 图


def figures(t: dict[str, Any]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})  # fmt: skip
    ASSETS.mkdir(exist_ok=True)
    out = []
    # 1. test 各系统按题型的准确率（分组柱状图）
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    groups = (*TYPES, "overall")
    systems = ("closed", "rag", "graphrag", "agent")
    w = 0.19
    for k, s in enumerate(systems):
        xs = [i + (k - 1.5) * (w + 0.01) for i in range(len(groups))]
        ax.bar(xs, [t["acc"][s][g] for g in groups], width=w, color=SERIES[k], label=SYS[s] + ("（两次平均）" if s == "agent" else ""),
               edgecolor=SURFACE, linewidth=1)  # fmt: skip
    for i, g in enumerate(groups):  # 只标 Agent 与固定 RAG 的数值
        for k, s in ((1, "rag"), (3, "agent")):
            x = i + (k - 1.5) * (w + 0.01)
            ax.text(
                x, t["acc"][s][g] + 0.015, f"{t['acc'][s][g]:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2
            )
    ax.set_xticks(range(len(groups)), [f"{TNAME[g]}\n(n={t['n'][g]})" for g in groups], color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("准确率（完全正确）", color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=9, labelcolor=INK)
    ax.set_title("v1 test（141 题）各系统按题型的准确率", color=INK, fontsize=11, pad=26, loc="left")
    fig.tight_layout()
    p = ASSETS / "accuracy_by_type.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    out.append(p)

    # 2. 成本—准确率（dev）：x = 每题 token，y = dev 准确率（闭卷只在表格里，放进来会把纵轴压扁）
    p4 = J("p4_comparison.json")
    abl = J("ablation_qa_v1.json")
    fe = J("p6_route_feasibility.json")
    pts = [  # (标签, token, 准确率, 系列)
        ("固定 RAG", p4["cost"]["rag"]["tokens"], p4["accuracy"]["rag"]["overall"]["acc"], 0),
        ("固定 GraphRAG", p4["cost"]["graphrag"]["tokens"], p4["accuracy"]["graphrag"]["overall"]["acc"], 0),
        ("Agent 默认", abl["default"]["groups"]["overall"]["tokens"], abl["default"]["groups"]["overall"]["acc"], 1),
    ]
    short = {
        "去掉全部图谱": "去掉图谱",
        "去掉摘要": "去掉摘要",
        "预算低档": "预算低档",
        "预算高档": "预算高档",
        "开启思考": "开启思考",
        "原始历史": "看原始历史",
    }
    for r in abl["rows"]:
        lab = next((v for k, v in short.items() if k in r["label"]), None)
        if lab:
            pts.append((lab, r["groups"]["overall"]["tokens"], r["groups"]["overall"]["acc"], 1))
    for sig, lab in (("verify", "按需升级·核对（估算）"), ("selfcheck", "按需升级·自评（估算）")):
        s = fe["signals"][sig]
        pts.append((lab, mean(s["est_tokens"]), mean(s["est_acc"]), 2))
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    names = ("固定流程", "Agent（默认与消融变体）", "按需升级（离线估算，未采纳）")
    for k in range(3):
        sel = [x for x in pts if x[3] == k]
        ax.scatter([x[1] for x in sel], [x[2] for x in sel], s=46, color=SERIES[k] if k < 2 else SURFACE,
                   edgecolors=SERIES[k], linewidths=2 if k == 2 else 1.2, label=names[k], zorder=3)  # fmt: skip
    offs = {"Agent 默认": (-7, -13, "right"), "去掉图谱": (-7, 3, "right"), "去掉摘要": (0, 8, "center"),
            "看原始历史": (7, -11, "left"), "预算高档": (4, -15, "left"), "开启思考": (0, 8, "center"),
            "预算低档": (7, -4, "left"), "按需升级·核对（估算）": (0, -16, "center"), "按需升级·自评（估算）": (0, 8, "center"),
            "固定 RAG": (7, 2, "left"), "固定 GraphRAG": (7, -10, "left")}  # fmt: skip
    for lab, x, y, _k in pts:
        dx, dy, ha = offs.get(lab, (7, -4, "left"))
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK2, ha=ha)
    ax.set_xlim(0, 28000)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v / 1000:.0f}k"))
    ax.set_xlabel("每题 token", color=INK2)
    ax.set_ylabel("v1 dev 准确率", color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    ax.set_ylim(0.64, 0.92)
    ax.set_title("成本与准确率（v1 dev 141 题；Agent 默认为三次采样平均）", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    p = ASSETS / "cost_accuracy.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    out.append(p)
    return out


# ---------------------------------------------------------------- 主程序


def render_blocks() -> dict[str, str]:
    t = test_results()
    lat = latency()
    return {
        "main_results": block_main(t, lat),
        "latency": block_latency(lat),
        "findings": block_findings(t),
        "ablation": block_ablation(),
        "failures": block_failures(),
        "eval_meta": block_eval_meta(t),
    }


def apply(readme: str, blocks: dict[str, str]) -> str:
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- BEGIN GENERATED: {name} -->\n).*?(<!-- END GENERATED: {name} -->)", re.S)
        if not pat.search(readme):
            raise RuntimeError(f"README 里没有生成区块 {name}")
        readme = pat.sub(lambda m, b=body: m.group(1) + b.strip() + "\n" + m.group(2), readme)
    return readme


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    cur = README.read_text(encoding="utf-8")
    new = apply(cur, render_blocks())
    if args.check:
        print("README 与结果文件一致" if new == cur else "README 与结果文件不一致，运行 scripts/make_readme.py")
        return 0 if new == cur else 1
    README.write_text(new, encoding="utf-8", newline="\n")
    for p in figures(test_results()):
        print(f"图：{p.relative_to(PROJECT_ROOT)}")
    print("README 生成区块已更新")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
