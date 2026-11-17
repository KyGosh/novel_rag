# ruff: noqa: E501 —— 含较长的中文表头与说明文字
"""按需升级（P6 B2）的可行性验证与 dev 评估报告：从 eval/results/ 的运行结果汇总，不手抄数字。

可行性（P6 方案第三节第 3 小节第 1 步）：
    uv run python scripts/route_report.py feasibility --verify p6_route_verify_s0 --selfcheck p6_route_selfcheck_s0 \
        --rag-ref p4_rag_dev --agents p4_agent_dev_s0,p4_agent_dev_s1,p5_default_dev_s2
  - 两次 dry 运行（只做第一阶段和路由、判分第一阶段的答案）。"没答对" = 判分不是 correct（部分对也算没答对）。
  - 每个信号（rules = 只看两条规则，即只看弃答的下限；verify；selfcheck）：对没答对的题的召回率、升级比例、
    答对的题被升级的比例（误报率），按题型。
  - 门槛（事先定好）：召回率 ≥ 0.75 且升级比例 ≤ 50%。都达标选 selfcheck（少一次调用），verify 的召回率高出
    0.05 以上则选 verify；都不达标则放弃按需升级。
  - 附：与 Agent 的几次 dev 运行离线组合的准确率估算（升级的题取 Agent 的判分），只作参考。
  - verify 的第一阶段与固定 RAG 完全相同：与 --rag-ref 逐题比较答案和判分（缓存回放应完全一致）。
  输出 eval/results/p6_route_feasibility.{md,json}。

dev 评估（第 3–5 步）：
    uv run python scripts/route_report.py evaluate --signal verify \
        --cascade p6_cascade_verify_s0,p6_cascade_verify_s1,p6_cascade_verify_s2 \
        --dry p6_route_verify_s0,p6_route_verify_s1,p6_route_verify_s2 \
        --agents p4_agent_dev_s0,p4_agent_dev_s1,p5_default_dev_s2 \
        --latency cascade=p6_lat_cascade,agent=p6_lat_agent_par
  - 三次采样：按需升级第 k 次与 Agent 第 k 次运行配对（Agent 部分从缓存回放）；固定 RAG 的三次采样 = dry 运行。
  - 报告：路由召回率 / 误报率 / 升级比例（每次采样），按题型准确率，相对 Agent、相对固定 RAG 的配对差值，
    每题 LLM 调用数与 token，延迟（单独的延迟测量），采纳规则逐条判定。
  - 一致性检查：升级的题，Agent 部分的工具轨迹和答案与对应的 Agent 运行相同，且没有新的 LLM 调用；
    没升级的题，答案与同一采样的 dry 运行相同。
  输出 eval/results/p6_cascade_dev.{md,json}。
"""

from __future__ import annotations

import argparse
import json
import sys
from statistics import mean, median
from typing import Any

from novel_rag.config import get_settings
from novel_rag.eval.ablation import find_run, load_run
from novel_rag.eval.runner import TYPE_NAMES, TYPE_ORDER
from novel_rag.eval.stats import paired_diff

GNAME = {**TYPE_NAMES, "overall": "总体", "hard": "多跳 + 跨章节"}
GROUPS = (*TYPE_ORDER, "hard", "overall")
RECALL_MIN, ESC_MAX, PREFER_MARGIN = 0.75, 0.50, 0.05


def in_group(t: str, g: str) -> bool:
    return g == "overall" or t == g or (g == "hard" and t in ("multihop", "plot"))


def rows_of(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in run["rows"]}


def correct(r: dict[str, Any]) -> bool:
    return r["label"] == "correct"


def signal_stats(rows: dict[str, dict[str, Any]], escalate: dict[str, bool]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for g in GROUPS:
        ids = [i for i, r in rows.items() if in_group(r["type"], g)]
        wrong = [i for i in ids if not correct(rows[i])]
        right = [i for i in ids if correct(rows[i])]
        out[g] = {
            "n": len(ids), "n_wrong": len(wrong),
            "recall": mean(escalate[i] for i in wrong) if wrong else None,
            "escalation": mean(escalate[i] for i in ids) if ids else None,
            "false_alarm": mean(escalate[i] for i in right) if right else None,
            "stage1_acc": mean(correct(rows[i]) for i in ids) if ids else None,
        }  # fmt: skip
    return out


def combined_acc(stage1: dict[str, dict[str, Any]], escalate: dict[str, bool], agent: dict[str, dict[str, Any]],
                 g: str = "overall") -> float:  # fmt: skip
    ids = [i for i, r in stage1.items() if in_group(r["type"], g)]
    return mean(correct(agent[i]) if escalate[i] else correct(stage1[i]) for i in ids)


def f3(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def feasibility(args: argparse.Namespace, rd, cfg) -> dict[str, Any]:  # noqa: ANN001
    ver = load_run(find_run(rd, args.verify))
    sc = load_run(find_run(rd, args.selfcheck))
    rv, rs = rows_of(ver), rows_of(sc)
    agents = [rows_of(load_run(find_run(rd, n))) for n in args.agents.split(",")] if args.agents else []
    signals = {
        "rules": (rv, {i: r["route"]["rule"] is not None for i, r in rv.items()}),
        "verify": (rv, {i: bool(r["route"]["escalate"]) for i, r in rv.items()}),
        "selfcheck": (rs, {i: bool(r["route"]["escalate"]) for i, r in rs.items()}),
    }
    res: dict[str, Any] = {"runs": {"verify": ver["dir"], "selfcheck": sc["dir"]}, "signals": {}}
    for name, (rows, esc) in signals.items():
        st = signal_stats(rows, esc)
        o = st["overall"]
        passed = o["recall"] is not None and o["recall"] >= RECALL_MIN and o["escalation"] <= ESC_MAX
        reasons: dict[str, int] = {}
        for r in rows.values():
            k = r["route"]["reason"] or "none"
            reasons[k] = reasons.get(k, 0) + 1
        est = [combined_acc(rows, esc, a) for a in agents]
        est_hard = [combined_acc(rows, esc, a, "hard") for a in agents]
        res["signals"][name] = {"by_group": st, "passed": passed, "reasons": reasons, "est_acc": est,
                                "est_acc_hard": est_hard}  # fmt: skip
    if agents:
        res["agent_acc"] = [mean(correct(r) for r in a.values()) for a in agents]
        res["agent_acc_hard"] = [mean(correct(r) for r in a.values() if in_group(r["type"], "hard")) for a in agents]
    # 决策（事先定好的规则）
    a, b = res["signals"]["selfcheck"], res["signals"]["verify"]
    if a["passed"] and b["passed"]:
        choice = (
            "verify"
            if b["by_group"]["overall"]["recall"] >= a["by_group"]["overall"]["recall"] + PREFER_MARGIN
            else "selfcheck"
        )
    elif a["passed"] or b["passed"]:
        choice = "selfcheck" if a["passed"] else "verify"
    else:
        choice = None
    res["choice"] = choice
    if args.rag_ref:
        ref = rows_of(load_run(find_run(rd, args.rag_ref)))
        ids = sorted(set(ref) & set(rv))
        res["stage1_vs_rag_ref"] = {
            "ref": args.rag_ref, "n": len(ids),
            "answer_same": sum(ref[i]["answer"] == rv[i]["answer"] for i in ids),
            "label_same": sum(ref[i]["label"] == rv[i]["label"] for i in ids),
        }  # fmt: skip
    md = ["# 按需升级：路由信号可行性（v1 dev）", "",
          f"- 运行：verify `{ver['dir']}`；selfcheck `{sc['dir']}`（dry：只做第一阶段和路由，判分第一阶段答案）",
          "- 没答对 = 判分不是 correct（部分对也算）。召回率 = 没答对的题中被升级的比例；误报率 = 答对的题中被升级的比例。",
          f"- 门槛（事先定好）：召回率 ≥ {RECALL_MIN}，升级比例 ≤ {ESC_MAX:.0%}。都达标选 selfcheck，verify 召回率高出 {PREFER_MARGIN} 以上则选 verify。",
          "- rules = 只看两条不花调用的规则（弃答、引用为空或无效），是下限；verify 和 selfcheck 都先过这两条规则。"]  # fmt: skip
    if "stage1_vs_rag_ref" in res:
        x = res["stage1_vs_rag_ref"]
        md.append(
            f"- verify 的第一阶段与 `{x['ref']}` 逐题比较：答案相同 {x['answer_same']}/{x['n']}，判分相同 {x['label_same']}/{x['n']}"
        )
    md += [
        "",
        "| 信号 | 第一阶段准确率 | 没答对 | 召回率 | 升级比例 | 误报率 | 达标 | 升级原因 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in res["signals"].items():
        o = s["by_group"]["overall"]
        md.append(
            f"| {name} | {f3(o['stage1_acc'])} | {o['n_wrong']}/{o['n']} | {f3(o['recall'])} | {f3(o['escalation'])} | "
            f"{f3(o['false_alarm'])} | {'是' if s['passed'] else '否'} | {s['reasons']} |"
        )
    md += [
        "",
        "按题型（召回率 / 升级比例）：",
        "",
        "| 信号 | " + " | ".join(GNAME[g] for g in GROUPS) + " |",
        "|---" * (len(GROUPS) + 1) + "|",
    ]
    for name, s in res["signals"].items():
        md.append(
            f"| {name} | "
            + " | ".join(f"{f3(s['by_group'][g]['recall'])} / {f3(s['by_group'][g]['escalation'])}" for g in GROUPS)
            + " |"
        )
    if agents:
        md += ["", "离线估算（升级的题取 Agent 各次 dev 运行的判分，只作参考）：", "",
               "| 路由 | 总体准确率（各次） | 平均 | 多跳 + 跨章节（平均） |", "|---|---|---|---|",
               f"| 全部走 Agent | {' / '.join(f3(x) for x in res['agent_acc'])} | {f3(mean(res['agent_acc']))} | {f3(mean(res['agent_acc_hard']))} |"]  # fmt: skip
        for name, s in res["signals"].items():
            md.append(
                f"| {name} | {' / '.join(f3(x) for x in s['est_acc'])} | {f3(mean(s['est_acc']))} | {f3(mean(s['est_acc_hard']))} |"
            )
    md += ["", f"**结论（按事先定好的规则）**：{'选用 ' + choice if choice else '两个信号都不达标，放弃按需升级'}"]
    res["markdown"] = "\n".join(md) + "\n"
    return res


def evaluate(args: argparse.Namespace, rd, cfg) -> dict[str, Any]:  # noqa: ANN001
    cas = [load_run(find_run(rd, n)) for n in args.cascade.split(",")]
    dry = [load_run(find_run(rd, n)) for n in args.dry.split(",")]
    agt = [load_run(find_run(rd, n)) for n in args.agents.split(",")]
    C, D, A = [rows_of(r) for r in cas], [rows_of(r) for r in dry], [rows_of(r) for r in agt]
    ids = sorted(set.intersection(*(set(x) for x in (*C, *D, *A))))
    types = {i: C[0][i]["type"] for i in ids}
    bs = {"n_samples": cfg.bootstrap_samples, "seed": cfg.bootstrap_seed, "level": cfg.ci_level}

    def pq(runs: list[dict[str, dict[str, Any]]], f) -> dict[str, float]:  # noqa: ANN001
        return {i: mean(f(r[i]) for r in runs) for i in ids}

    acc = {"cascade": pq(C, correct), "agent": pq(A, correct), "rag": pq(D, correct)}
    res: dict[str, Any] = {"signal": args.signal, "n": len(ids), "runs": {"cascade": [r["dir"] for r in cas],
                           "dry": [r["dir"] for r in dry], "agents": [r["dir"] for r in agt]}}  # fmt: skip
    res["acc"] = {s: {g: mean(v[i] for i in ids if in_group(types[i], g)) for g in GROUPS} for s, v in acc.items()}
    res["sample_acc"] = {
        s: [mean(correct(r[i]) for i in ids) for r in R] for s, R in (("cascade", C), ("agent", A), ("rag", D))
    }
    res["paired"] = {}
    for ref in ("agent", "rag"):
        res["paired"][ref] = {}
        for g in GROUPS:
            gi = [i for i in ids if in_group(types[i], g)]
            res["paired"][ref][g] = paired_diff([acc[ref][i] for i in gi], [acc["cascade"][i] for i in gi], **bs)
    # 路由（每次采样：dry 运行有第一阶段的判分）
    res["route"] = []
    for k, (c, d) in enumerate(zip(C, D, strict=True)):
        esc = {i: bool(c[i]["route"]["escalate"]) for i in ids}
        st = signal_stats({i: d[i] for i in ids}, esc)
        res["route"].append({g: st[g] for g in ("overall", "hard")} | {"sample": k})

    # 成本
    def cost(runs, key):  # noqa: ANN001, ANN202
        return mean(mean(r[i]["metrics"][key] for i in ids) for r in runs)

    res["cost"] = {
        s: {k: cost(R, k) for k in ("llm_calls", "tool_calls", "tokens")}
        for s, R in (("cascade", C), ("agent", A), ("rag", D))
    }
    # 延迟（单独的延迟测量：逐题顺序执行、关闭缓存）
    res["latency"] = {}
    for spec in (args.latency or "").split(","):
        if "=" in spec:
            k, n = spec.split("=", 1)
            run = load_run(find_run(rd, n))
            lat = [r["latency_ms"]["total"] for r in run["rows"] if r["latency_ms"].get("total")]
            tok = [r["metrics"]["tokens"] for r in run["rows"]]
            esc = [bool((r.get("route") or {}).get("escalate")) for r in run["rows"]] if k == "cascade" else None
            res["latency"][k] = {"dir": run["dir"], "n": len(lat), "mean_ms": mean(lat), "p50_ms": median(lat),
                                 "tokens_mean": mean(tok), "acc": mean(correct(r) for r in run["rows"]),
                                 "escalation": mean(esc) if esc else None}  # fmt: skip
    # 一致性检查
    chk = {"escalated": 0, "agent_answer_diff": 0, "agent_traj_diff": 0, "agent_uncached_steps": 0,
           "kept": 0, "kept_answer_diff": 0}  # fmt: skip
    for c, d, a in zip(C, D, A, strict=True):
        for i in ids:
            if c[i]["route"]["escalate"]:
                chk["escalated"] += 1
                chk["agent_answer_diff"] += c[i]["answer"] != a[i]["answer"]
                tr = lambda r: [(s.get("round"), s.get("tool"), json.dumps(s.get("args"), sort_keys=True, ensure_ascii=False))  # noqa: E731
                                for s in (r.get("agent") or {}).get("trace", []) if s.get("step") == "tool"]  # fmt: skip
                chk["agent_traj_diff"] += tr(c[i]) != tr(a[i])
                chk["agent_uncached_steps"] += sum(
                    1
                    for s in c[i]["agent"]["trace"]
                    if s.get("step") in ("plan", "act", "reflect") and s.get("cached") is False
                )
            else:
                chk["kept"] += 1
                chk["kept_answer_diff"] += c[i]["answer"] != d[i]["answer"]
    res["consistency"] = chk
    # 采纳规则（事先定好，四条同时满足）
    ac, aa = res["acc"]["cascade"], res["acc"]["agent"]
    rule = {
        "acc_overall": ac["overall"] >= aa["overall"] - 0.03,
        "acc_hard": ac["hard"] >= aa["hard"] - 0.05,
        "vs_rag_ci_excl_0": res["paired"]["rag"]["overall"]["lo"] > 0,
    }
    L = res["latency"]
    if "cascade" in L and "agent" in L:
        rule["latency_30"] = L["cascade"]["mean_ms"] <= 0.7 * L["agent"]["mean_ms"]
    rule["tokens_30"] = res["cost"]["cascade"]["tokens"] <= 0.7 * res["cost"]["agent"]["tokens"]
    res["adoption"] = {"rules": rule, "adopt": all(rule.values()) and "latency_30" in rule}

    md = [f"# 按需升级（{args.signal}）：v1 dev 评估", "",
          f"- {len(ids)} 题，三次采样：按需升级第 k 次与 Agent 第 k 次 dev 运行配对（Agent 部分从缓存回放）；固定 RAG 的三次采样 = 第一阶段的 dry 运行。",
          f"- 按需升级 {', '.join('`' + r['dir'] + '`' for r in cas)}",
          f"- Agent {', '.join('`' + r['dir'] + '`' for r in agt)}",
          f"- 固定 RAG（dry）{', '.join('`' + r['dir'] + '`' for r in dry)}", "",
          "## 准确率（三次采样按题平均）", "",
          "| 系统 | " + " | ".join(GNAME[g] for g in GROUPS) + " | 各次总体 |", "|---" * (len(GROUPS) + 2) + "|"]  # fmt: skip
    for s, label in (("rag", "固定 RAG"), ("cascade", "按需升级"), ("agent", "Agent")):
        md.append(
            f"| {label} | "
            + " | ".join(f3(res["acc"][s][g]) for g in GROUPS)
            + " | "
            + " / ".join(f3(x) for x in res["sample_acc"][s])
            + " |"
        )
    md += [
        "",
        "## 配对差值（按需升级 − 参照，95% CI）",
        "",
        "| 参照 | " + " | ".join(GNAME[g] for g in GROUPS) + " |",
        "|---" * (len(GROUPS) + 1) + "|",
    ]
    for ref, label in (("agent", "Agent"), ("rag", "固定 RAG")):
        md.append(
            f"| {label} | "
            + " | ".join(
                f"{p['diff']:+.3f} [{p['lo']:+.2f}, {p['hi']:+.2f}]" for p in (res["paired"][ref][g] for g in GROUPS)
            )
            + " |"
        )
    md += [
        "",
        "## 路由（每次采样；召回率 / 误报率相对第一阶段自己的判分）",
        "",
        "| 采样 | 升级比例 | 召回率 | 误报率 | 多跳 + 跨章节升级比例 | 多跳 + 跨章节召回率 |",
        "|---|---|---|---|---|---|",
    ]
    for r in res["route"]:
        o, h = r["overall"], r["hard"]
        md.append(
            f"| {r['sample']} | {f3(o['escalation'])} | {f3(o['recall'])} | {f3(o['false_alarm'])} | {f3(h['escalation'])} | {f3(h['recall'])} |"
        )
    md += ["", "## 成本与延迟", "", "| 系统 | 每题 LLM 调用 | 每题工具调用 | 每题 token |", "|---|---|---|---|"]
    for s, label in (("rag", "固定 RAG"), ("cascade", "按需升级"), ("agent", "Agent")):
        c = res["cost"][s]
        md.append(f"| {label} | {c['llm_calls']:.2f} | {c['tool_calls']:.2f} | {c['tokens']:.0f} |")
    if L:
        md += [
            "",
            "延迟（单独测量：同一时段、逐题顺序执行、关闭 LLM 缓存）：",
            "",
            "| 系统 | 运行 | 题数 | 平均 | P50 | 每题 token | 准确率 | 升级比例 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for k, v in L.items():
            md.append(
                f"| {k} | `{v['dir']}` | {v['n']} | {v['mean_ms'] / 1000:.1f} s | {v['p50_ms'] / 1000:.1f} s | {v['tokens_mean']:.0f} | {f3(v['acc'])} | {f3(v['escalation'])} |"
            )
    md += ["", "## 一致性检查", "", f"- 升级的题（三次采样合计 {chk['escalated']}）：答案与对应 Agent 运行不同 {chk['agent_answer_diff']}，工具轨迹不同 {chk['agent_traj_diff']}，未命中缓存的 Agent 环节 {chk['agent_uncached_steps']}",
           f"- 没升级的题（合计 {chk['kept']}）：答案与同一采样的 dry 运行不同 {chk['kept_answer_diff']}", "",
           "## 采纳规则（事先定好，四条同时满足才列为推荐的省成本模式）", ""]  # fmt: skip
    names = {"acc_overall": "总体准确率不低于 Agent 0.03 以上", "acc_hard": "多跳 + 跨章节不低于 Agent 0.05 以上",
             "vs_rag_ci_excl_0": "相对固定 RAG 的配对差值区间不含 0", "latency_30": "平均延迟比 Agent 低 30% 以上",
             "tokens_30": "平均 token 比 Agent 低 30% 以上"}  # fmt: skip
    for k, v in rule.items():
        md.append(f"- {names[k]}：{'满足' if v else '不满足'}")
    md += ["", f"**结论**：{'满足采纳规则' if res['adoption']['adopt'] else '不满足采纳规则'}"]
    res["markdown"] = "\n".join(md) + "\n"
    return res


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("feasibility")
    f.add_argument("--verify", required=True)
    f.add_argument("--selfcheck", required=True)
    f.add_argument("--rag-ref")
    f.add_argument("--agents")
    f.add_argument("--out", default="p6_route_feasibility")
    e = sub.add_parser("evaluate")
    e.add_argument("--signal", required=True)
    e.add_argument("--cascade", required=True)
    e.add_argument("--dry", required=True)
    e.add_argument("--agents", required=True)
    e.add_argument("--latency")
    e.add_argument("--out", default="p6_cascade_dev")
    args = ap.parse_args()
    s = get_settings()
    rd = s.paths.results_dir
    res = feasibility(args, rd, s.eval) if args.cmd == "feasibility" else evaluate(args, rd, s.eval)
    md = res.pop("markdown")
    (rd / f"{args.out}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (rd / f"{args.out}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"写入 {rd / (args.out + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
