# ruff: noqa: E501 —— 报告渲染脚本，含较长的中文表头与说明文字
"""P5 失败模式与收益来源分析（方案第三节）：只读 P4 已有的运行结果与 trace，不调用任何模型。

用法：
    uv run python scripts/p5_analysis.py digest        # 逐题阅读材料 → data/reports/p5/（含原文，不入库）
    uv run python scripts/p5_analysis.py cite-sample   # 抽查"不在金标里的引用" → data/reports/p5/cite_review.md
    uv run python scripts/p5_analysis.py report        # 报告 → eval/results/p5_failure_analysis.md / .json

report 把自动统计（题组、环节分类、金标证据轨迹、收益机制）与人工归因（eval/analysis/p5_failure_labels.json，
由 Claude 逐题阅读 trace 后填写：原因类别 + 简短转述，不含原文）合并。报告只写题号、类别和转述。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from novel_rag.config import get_settings
from novel_rag.corpus import BookText
from novel_rag.eval.dataset import load_dataset
from novel_rag.eval.runner import TYPE_NAMES, TYPE_ORDER
from novel_rag.qa.analysis import STAGE_NAMES, classify
from novel_rag.qa.env import load_env
from novel_rag.qa.trail import MODE_NAMES, evidence_state, evidence_trail, gain_source, question_groups, rag_evidence

DEFAULT_RUNS = {
    "rag": "20261112-004212_p4_rag_dev",
    "s0": "20261112-005605_p4_agent_dev_s0",
    "s1": "20261112-010511_p4_agent_dev_s1",
}
GROUP_NAMES = {
    "stable_fail": "稳定失败（两次都没答对）",
    "unstable": "不稳定（两次结论不同）",
    "regress": "回退（固定 RAG 对、Agent 第一次运行错）",
    "regress_s1": "回退（固定 RAG 对、Agent 第二次运行错）",
    "gain": "收益（固定 RAG 错、Agent 第一次运行对）",
    "gain_s1": "收益（固定 RAG 错、Agent 第二次运行对）",
}


def load_rows(d: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(x) for x in (d / "per_question.jsonl").read_text(encoding="utf-8").splitlines() if x]
    return {r["id"]: r for r in rows}


class Ctx:
    def __init__(self, args: argparse.Namespace):
        self.s = get_settings()
        e = self.s.eval
        self.rel = {"min_ratio": e.relevance_min_ratio, "min_chars": e.relevance_min_chars}
        res = self.s.paths.results_dir
        self.run_dirs = {k: res / getattr(args, k) for k in DEFAULT_RUNS}
        self.rows = {k: load_rows(d) for k, d in self.run_dirs.items()}
        meta = json.loads((self.run_dirs["s0"] / "run.json").read_text(encoding="utf-8"))
        self.env = load_env(self.s, tuple(meta["scope"]))
        self.spans = self.env.spans
        ds = self.s.paths.datasets_dir / args.dataset
        self.items = {i.id: i for i in load_dataset(ds, self.s.paths.books_dir, split="dev")}
        self.groups = question_groups(self.rows["rag"], self.rows["s0"], self.rows["s1"])
        self.out_dir = self.s.paths.data_dir / "reports" / "p5"
        self._books: dict[str, BookText] = {}

    def book(self, book_id: str) -> BookText:
        if book_id not in self._books:
            self._books[book_id] = BookText.load(self.s.paths.books_dir / book_id)
        return self._books[book_id]

    def where(self, chunk_id: str) -> str:
        c = self.env.chunks.by_id(chunk_id)
        return f"{self.env.titles[c['book_id']]}·{c.get('chapter_title') or c['chapter_idx']}"

    def trace(self, run: str, qid: str) -> dict[str, Any]:
        p = self.s.paths.traces_dir / self.run_dirs[run].name / f"{qid}.json"
        return json.loads(p.read_text(encoding="utf-8"))

    def stage(self, run: str, qid: str) -> dict[str, Any]:
        return classify(self.rows[run][qid], self.items[qid], self.spans, self.rel)


# ---------------------------------------------------------------- digest（含原文，只写 data/）


def _args(a: dict[str, Any] | None) -> str:
    return ", ".join(f"{k}={v}" for k, v in (a or {}).items())


def digest_question(cx: Ctx, qid: str, *, full: bool) -> list[str]:
    it = cx.items[qid]
    out = [f"## {qid}（{TYPE_NAMES[it.type]}，max_ro={it.max_reading_order}）", f"问：{it.question}",
           f"金标答案：{it.answer}；其他写法：{it.answer_aliases}"]  # fmt: skip
    rag = cx.rows["rag"][qid]
    re_ = rag_evidence(rag, it, cx.spans, cx.rel)
    for k, ev in enumerate(it.evidence):
        ch = cx.book(ev.book_id).chapter(ev.chapter_idx)
        text = cx.book(ev.book_id).text[ev.char_start : ev.char_end]
        out.append(f"- 证据{k}（{ev.book_id} 第{ev.chapter_idx}章 ro={ch.reading_order}，RAG 排名 {re_[k]['rank']}，"
                   f"进上下文 {re_[k]['in_context']}）：{text}")  # fmt: skip
    out.append(f"固定 RAG：{rag['label']}｜答：{rag['answer']}｜{rag.get('judge_reason') or ''}")
    for run in ("s0", "s1"):
        r = cx.rows[run][qid]
        st = cx.stage(run, qid)
        trail = evidence_trail(r, it, cx.spans, cx.rel)
        a = r.get("agent", {})
        out.append(f"### {run}：{r['label']}（{r['scored_by']}）｜终止 {r['termination']}｜{a.get('rounds')} 轮 "
                   f"{r['cost'].get('tool_calls')} 次工具 {int(r['metrics']['tokens'])} token｜"
                   f"环节 {st['stage']} {st['flags']}")  # fmt: skip
        out.append(f"答：{r['answer']}｜judge：{r.get('judge_reason') or ''}｜引用 {r['citations']}")
        for k, t in enumerate(trail):
            sh = t["shown"]
            sh_s = f"R{sh['round']} {sh['mode']} {sh['chunk_id']}" if sh else "未展示"
            rf = t["ref"]
            rf_s = f"；编号先见于 R{rf['round']} {rf['tool']}" if rf else ""
            out.append(f"- 证据{k}：{sh_s}{rf_s}；进笔记 {t['noted']}；被引用 {t['cited']}")
        if not full:
            continue
        tr = cx.trace(run, qid)
        out.append(f"计划：{(tr.get('plan') or '')[:300]}")
        for stp in tr.get("trace", []):
            if stp["step"] == "tool":
                res = f"展示 {stp.get('shown')}" if stp.get("shown") else ""
                if stp.get("refs"):
                    res += f" 编号 {len(stp['refs'])} 个"
                meta = stp.get("meta") or {}
                extra = {k: meta[k] for k in ("names", "chapters", "op", "rows", "entity_chunks", "entity_filter")
                         if k in meta}  # fmt: skip
                err = "" if stp.get("ok") else f" 错误({stp.get('error_kind')}) {stp.get('observation', '')[:80]}"
                out.append(f"  R{stp['round']} {stp['tool']}({_args(stp.get('args'))}) → {res} {extra or ''}{err}")
            elif stp["step"] == "reflect":
                rej = [f"{x.get('chunk_id')}:{x.get('why')}" for x in stp.get("rejected", [])]
                out.append(f"  R{stp['round']} 反思：新增 {len(stp.get('added', []))} 条，作废 {rej}；"
                           f"sufficient={stp.get('sufficient')}；现状：{stp.get('status', stp.get('error', ''))}")  # fmt: skip
        for n in tr.get("notes", []):
            gold = any(t["noted"] and t["shown"] and t["shown"]["chunk_id"] == n["chunk_id"] for t in trail)
            out.append(f"  笔记[{n['chunk_id']} R{n['round']} 原文={n['seen_text']} 引文核对={n['quote_ok']}"
                       f"{' 金标' if gold else ''}] {n['claim']}")  # fmt: skip
        out.append(f"  结束理由：{(tr.get('finish_reason') or '')[:200]}")
        out.append(f"  作答依据：{(tr.get('answer') or {}).get('rationale', '')}")
    out.append("")
    return out


def compact_question(cx: Ctx, qid: str, runs: tuple[str, ...] = ("s0", "s1")) -> list[str]:
    """精简版阅读材料：每次运行只留判分、证据轨迹、工具序列（一行）、笔记要点和作答依据。"""
    it = cx.items[qid]
    rag = cx.rows["rag"][qid]
    re_ = rag_evidence(rag, it, cx.spans, cx.rel)
    out = [f"## {qid} 问：{it.question}", f"金标：{it.answer} {it.answer_aliases or ''}"]
    for k, ev in enumerate(it.evidence):
        text = cx.book(ev.book_id).text[ev.char_start : ev.char_end]
        out.append(f"- 证据{k} {ev.book_id}:{ev.chapter_idx} RAG排名{re_[k]['rank']}：{text[:70]}")
    out.append(f"RAG {rag['label']}：{rag['answer'][:40]}")
    for run in runs:
        r = cx.rows[run][qid]
        st = cx.stage(run, qid)
        trail = evidence_trail(r, it, cx.spans, cx.rel)
        tr = cx.trace(run, qid)
        ev_s = " ".join(
            f"E{k}:{('R' + str(t['shown']['round']) + t['shown']['mode'][:8]) if t['shown'] else '未见'}"
            f"{'/记' if t['noted'] else ''}{'/引' if t['cited'] else ''}" for k, t in enumerate(trail)
        )  # fmt: skip
        out.append(f"### {run} {r['label']}｜{r['termination']}｜{st['stage']}{st['flags']}｜{ev_s}")
        out.append(f"答：{r['answer'][:60]}｜judge：{(r.get('judge_reason') or '')[:70]}")
        calls: dict[int, list[str]] = {}
        for stp in tr.get("trace", []):
            if stp["step"] != "tool":
                continue
            a = stp.get("args") or {}
            key = str(a.get("query") or a.get("name") or a.get("chunk_id") or a.get("chapter") or a.get("op") or "")
            flt = "".join(f"[{k[:3]}]" for k in ("entity_id", "volume", "chapter_from") if a.get(k))
            calls.setdefault(stp["round"], []).append(
                f"{stp['tool'].replace('search_', 's_').replace('read_', 'r_')}({key[:22]}){flt}"
                f"{'' if stp.get('ok') else '✗'}"
            )
        out.append("  " + " ‖ ".join(f"R{k} " + "; ".join(v) for k, v in calls.items()))
        gold_ids = {t["shown"]["chunk_id"] for t in trail if t["shown"]}
        seen = set()
        for n in tr.get("notes", []):
            if n["chunk_id"] in seen:
                continue
            seen.add(n["chunk_id"])
            mark = ("金" if n["chunk_id"] in gold_ids else "") + ("" if n["seen_text"] else "摘")
            out.append(f"  ·{n['chunk_id'].split('-')[-1]}{mark} {n['claim'][:48]}")
        out.append(f"  依据：{(tr.get('answer') or {}).get('rationale', '')[:110]}")
    out.append("")
    return out


def cmd_digest(cx: Ctx) -> None:
    cx.out_dir.mkdir(parents=True, exist_ok=True)
    g = cx.groups
    compact = {
        "compact_stable_fail": g["stable_fail"],
        "compact_unstable": g["unstable"],
    }
    for name, ids in compact.items():
        lines: list[str] = []
        for qid in ids:
            lines += compact_question(cx, qid)
        (cx.out_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    plot_lines = []
    for qid, it in cx.items.items():
        if it.type == "plot":
            labs = "/".join(cx.rows[r][qid]["label"][:3] for r in ("rag", "s0", "s1"))
            plot_lines.append(f"{qid}｜{len(it.evidence)}段｜{labs}｜{it.question}｜{it.answer}")
    (cx.out_dir / "plot_list.md").write_text("\n".join(plot_lines), encoding="utf-8")
    files = {
        "digest_stable_fail": (g["stable_fail"], True),
        "digest_unstable": (g["unstable"], True),
        "digest_regress": (sorted(set(g["regress"]) | set(g["regress_s1"])), True),
        "digest_gain": (sorted(set(g["gain"]) | set(g["gain_s1"])), False),
        "digest_plot_all": ([i for i, it in cx.items.items() if it.type == "plot"], False),
    }
    for name, (ids, full) in files.items():
        lines = [f"# {name}（{len(ids)} 题；含原文，仅本地）", ""]
        for qid in ids:
            lines += digest_question(cx, qid, full=full)
        (cx.out_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"{name}: {len(ids)} 题 → {cx.out_dir / (name + '.md')}")


# ---------------------------------------------------------------- 引用抽查


def non_gold_citations(cx: Ctx, run: str) -> list[dict[str, Any]]:
    """答对的题里，与金标不重叠的引用。"""
    from novel_rag.eval.metrics import is_relevant

    out = []
    for qid, r in cx.rows[run].items():
        if r["label"] != "correct":
            continue
        it = cx.items[qid]
        for c in r.get("citations", []):
            if c in cx.spans and not any(is_relevant(cx.spans[c], ev, **cx.rel) for ev in it.evidence):
                out.append({"key": f"{run}|{qid}|{c}", "run": run, "id": qid, "chunk_id": c})
    return out


def cite_sample(cx: Ctx, n: int, seed: str) -> list[dict[str, Any]]:
    pool = non_gold_citations(cx, "s0") + non_gold_citations(cx, "rag")
    pool.sort(key=lambda x: hashlib.sha256(f"{seed}:{x['key']}".encode()).hexdigest())
    half = n // 2
    picked = [x for x in pool if x["run"] == "s0"][:half] + [x for x in pool if x["run"] == "rag"][: n - half]
    return picked


def cmd_cite_sample(cx: Ctx, n: int, seed: str) -> None:
    cx.out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# 引用抽查（{n} 条不在金标里的引用，种子 {seed}；含原文，仅本地）", ""]
    for x in cite_sample(cx, n, seed):
        it, r = cx.items[x["id"]], cx.rows[x["run"]][x["id"]]
        lines += [f"## {x['key']}", f"问：{it.question}", f"金标答案：{it.answer}", f"系统答案：{r['answer']}",
                  f"引用段落（{cx.where(x['chunk_id'])}）：{cx.env.chunks.by_id(x['chunk_id'])['text']}", ""]  # fmt: skip
    (cx.out_dir / "cite_review.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {cx.out_dir / 'cite_review.md'}")


# ---------------------------------------------------------------- 报告（不含原文）


def _pct(a: int, b: int) -> str:
    return f"{a}/{b}（{a / b:.0%}）" if b else "0/0"


def cmd_report(cx: Ctx, labels_path: Path, out_name: str, n_cite: int, seed: str) -> None:
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    causes: dict[str, Any] = labels.get("questions", {})
    cause_names: dict[str, str] = labels.get("cause_names", {})
    g, rows, items = cx.groups, cx.rows, cx.items
    out: dict[str, Any] = {"runs": {k: d.name for k, d in cx.run_dirs.items()}, "groups": g}
    md = [
        "# P5 失败模式与收益来源分析（asoiaf_v1 dev，141 题）",
        "",
        "- 数据来源：P4 的固定 RAG 运行与 Agent 的两次独立运行，"
        + "；".join(f"{k} = `{d.name}`" for k, d in cx.run_dirs.items())
        + "。没有新增模型调用。",  # fmt: skip
        "- 自动部分（题组、环节、金标证据轨迹、收益机制）由 `scripts/p5_analysis.py report` 从 `per_question.jsonl` "
        "计算；原因归类由 Claude 逐题阅读 `data/traces/` 中的完整 trace 后填写（`eval/analysis/p5_failure_labels.json`）。",
        "- **未经用户审阅**：原因归类是模型判定，可能带有判定者自身的偏差；v1 评测集本身未经人工复核。",
        "- 本报告只写题号、类别和简短转述，不含原文。",
        "",
        "## 1. 题组",
        "",
        "| 题组 | 题数 | " + " | ".join(TYPE_NAMES[t] for t in TYPE_ORDER) + " |",
        "|---" * (len(TYPE_ORDER) + 2) + "|",
    ]
    for k, name in GROUP_NAMES.items():
        c = Counter(items[i].type for i in g[k])
        md.append(f"| {name} | {len(g[k])} | " + " | ".join(str(c.get(t, 0)) for t in TYPE_ORDER) + " |")
    md.append("")

    # ---- 2. 自动环节分类（两次运行）
    md += ["## 2. 没答对的题按环节自动分类（两次运行）", "",
           "| 运行 | 没答对 | " + " | ".join(STAGE_NAMES[k] for k in STAGE_NAMES if k != "correct_bad_citation") + " |",
           "|---" * (len(STAGE_NAMES) + 1) + "|"]  # fmt: skip
    out["stages"] = {}
    for run in ("s0", "s1"):
        c: Counter = Counter()
        failed = [i for i in rows[run] if rows[run][i]["label"] != "correct"]
        for i in failed:
            st = cx.stage(run, i)
            c.update([st["stage"], *st["flags"]])
        out["stages"][run] = {"failed": len(failed), **c}
        md.append(f"| {run} | {len(failed)} | " + " | ".join(str(c.get(k, 0)) for k in STAGE_NAMES
                                                           if k != "correct_bad_citation") + " |")  # fmt: skip
    md.append("")

    # ---- 3–5. 失败题组：逐题
    def question_table(ids: list[str], title: str, key: str) -> None:
        md.extend(
            [
                f"## {title}",
                "",
                "| 题号 | 题型 | s0 判分 / 环节 | s1 判分 / 环节 | 原因类别 | 简述 |",
                "|---|---|---|---|---|---|",
            ]
        )
        cc: Counter = Counter()
        for i in ids:
            cells = []
            for run in ("s0", "s1"):
                r, st = rows[run][i], cx.stage(run, i)
                stage = STAGE_NAMES.get(st["stage"], "—") if st["stage"] else "—"
                cells.append(f"{r['label']} / {stage}")
            lab = causes.get(i, {})
            cs = lab.get("causes", [])
            cc.update(cs)
            md.append(f"| {i} | {TYPE_NAMES[items[i].type]} | {cells[0]} | {cells[1]} | "
                      f"{'、'.join(cause_names.get(c, c) for c in cs) or '（待归类）'} | {lab.get('note', '')} |")  # fmt: skip
        out[key] = {"ids": ids, "causes": dict(cc)}
        md.append("")
        if cc:
            md.append(
                "原因分布（一题可有多个原因）："
                + "；".join(f"{cause_names.get(k, k)} {v}" for k, v in cc.most_common())
            )
            md.append("")

    question_table(g["stable_fail"], "3. 稳定失败（两次都没答对，10 题）", "stable_fail")
    question_table(g["unstable"], "4. 不稳定（两次结论不同）", "unstable")
    question_table(
        sorted(set(g["regress"]) | set(g["regress_s1"])), "5. 回退（固定 RAG 对、Agent 某一次错）", "regress"
    )

    # ---- 6. 收益来源
    md += ["## 6. 收益来源（固定 RAG 没答对、Agent 答对）", ""]
    out["gain"] = {}
    for run, key in (("s0", "gain"), ("s1", "gain_s1")):
        gs = {i: gain_source(rows["rag"][i], rows[run][i], items[i], cx.spans, cx.rel) for i in g[key]}
        kinds = Counter(x["kind"] for x in gs.values())
        mech: Counter = Counter()
        for x in gs.values():
            mech.update(x["mechanisms"])
        retr = [x for x in gs.values() if x["kind"] == "retrieval"]
        first_r = Counter("第 1 轮" if x["last_found_round"] == 1 else "第 2 轮及以后" for x in retr)
        rag_abst = sum(1 for i in gs if rows["rag"][i]["label"] == "abstain")
        out["gain"][run] = {
            "n": len(gs), "kinds": dict(kinds), "mechanisms": dict(mech), "found_round": dict(first_r),
            "rag_abstained": rag_abst,
            "used_find_entity": sum(x["used_find_entity"] for x in gs.values()),
            "used_graph_query": sum(x["used_graph_query"] for x in gs.values()),
            "used_summaries": sum(x["used_summaries"] for x in gs.values()),
            "per_question": {i: {k: v for k, v in x.items() if k not in ("trail", "rag")} for i, x in gs.items()},
        }  # fmt: skip
        n = len(gs)
        md += [
            f"### {run}（{n} 题："
            + "、".join(f"{TYPE_NAMES[t]} {c}" for t, c in Counter(items[i].type for i in gs).items())
            + "）",  # fmt: skip
            "",
            f"- 固定 RAG 在这些题上：弃答 {rag_abst}，答错或部分对 {n - rag_abst}。",
            f"- **检索收益** {_pct(kinds.get('retrieval', 0), n)}：Agent 以原文形式找到了固定 RAG 上下文里没有的金标证据。",
            f"- **证据整理收益** {_pct(kinds.get('rag_had_evidence', 0), n)}：固定 RAG 的上下文里金标证据已经齐全却没答对，"
            "Agent 经笔记整理后答对。",
            f"- 其他 {_pct(kinds.get('other_evidence', 0), n)}：Agent 没有多找到金标证据也答对（靠金标以外的段落）。",
            "- 检索收益题中，补上的金标证据第一次出现的方式（一题可有多种）："
            + "；".join(f"{MODE_NAMES[k]} {v}" for k, v in mech.most_common())
            + "。",  # fmt: skip
            "- 检索收益题中，最后一段补上的证据出现在："
            + "；".join(f"{k} {v} 题" for k, v in sorted(first_r.items()))
            + "。",
            f"- 这些题里用过实体查找的 {out['gain'][run]['used_find_entity']} 题、图谱查询 "
            f"{out['gain'][run]['used_graph_query']} 题、摘要工具 {out['gain'][run]['used_summaries']} 题。",
            "",
        ]
    md += ["逐题（第一次运行）：", "", "| 题号 | 题型 | 收益类型 | RAG 缺的证据 / 总数 | 补上的方式 | 补齐轮次 | 实体查找 | 图谱查询 | 摘要 |",
           "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    kind_names = {"retrieval": "检索", "rag_had_evidence": "证据整理", "other_evidence": "其他"}
    for i, x in out["gain"]["s0"]["per_question"].items():
        yn = lambda b: "是" if b else ""  # noqa: E731
        md.append(f"| {i} | {TYPE_NAMES[items[i].type]} | {kind_names[x['kind']]} | {x['rag_missed']}/{x['n_evidence']} | "
                  f"{'、'.join(k for k in x['mechanisms'])} | {x['last_found_round'] or ''} | {yn(x['used_find_entity'])} | "
                  f"{yn(x['used_graph_query'])} | {yn(x['used_summaries'])} |")  # fmt: skip
    md.append("")
    if labels.get("gain_notes"):
        md += [labels["gain_notes"], ""]

    # ---- 7. 跨章节题专项
    plot = [i for i in items if items[i].type == "plot"]
    md += ["## 7. 跨章节题专项（34 题）", "",
           "证据到位情况：all_noted = 金标证据全部进了笔记；all_shown = 全部看到过原文但没记全；partial = 只到位一部分；"
           "none = 一段都没看到。", "",
           "| 运行 | 证据到位情况 | 题数 | correct | partial | wrong | abstain |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    out["plot"] = {}
    for run in ("s0", "s1"):
        tab: dict[str, Counter] = {}
        for i in plot:
            stt = evidence_state(evidence_trail(rows[run][i], items[i], cx.spans, cx.rel))
            tab.setdefault(stt, Counter())[rows[run][i]["label"]] += 1
        out["plot"][run] = {k: dict(v) for k, v in tab.items()}
        for stt in ("all_noted", "all_shown", "partial", "none"):
            c = tab.get(stt, Counter())
            md.append(f"| {run} | {stt} | {sum(c.values())} | {c.get('correct', 0)} | {c.get('partial', 0)} | "
                      f"{c.get('wrong', 0)} | {c.get('abstain', 0)} |")  # fmt: skip
    md.append("")
    tags = labels.get("plot_tags", {})
    if tags:
        agg = [i for i in plot if tags.get(i, {}).get("aggregate")]
        md += ["汇总类题目（答案需要列出两项及以上内容，由 Claude 按金标答案标注）与其余题目对比：", "",
               "| 子集 | 题数 | s0 correct | s0 partial | s1 correct | s1 partial | 固定 RAG correct | 固定 RAG partial |",
               "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        out["plot"]["aggregate"] = {}
        for name, ids in (("汇总类", agg), ("非汇总类", [i for i in plot if i not in agg])):
            cells = []
            for run in ("s0", "s1", "rag"):
                cells += [f"{mean(rows[run][i]['label'] == 'correct' for i in ids):.3f}",
                          f"{mean(rows[run][i]['label'] == 'partial' for i in ids):.3f}"]  # fmt: skip
            out["plot"]["aggregate"][name] = {"n": len(ids), "ids": ids, "cells": cells}
            md.append(f"| {name} | {len(ids)} | " + " | ".join(cells) + " |")
        md.append("")
    if labels.get("plot_notes"):
        md += [labels["plot_notes"], ""]

    # ---- 8. 引用抽查
    cites = labels.get("citations", {})
    sample = cite_sample(cx, n_cite, seed)
    md += ["## 8. 引用准确率是下界：抽查不在金标里的引用", ""]
    out["citations"] = {}
    for run in ("s0", "rag"):
        ng = non_gold_citations(cx, run)
        total = sum(len(r.get("citations", [])) for r in rows[run].values() if r["label"] == "correct")
        picked = [x for x in sample if x["run"] == run]
        v = Counter(cites.get(x["key"], {}).get("verdict", "unlabeled") for x in picked)
        out["citations"][run] = {
            "correct_citations": total,
            "non_gold": len(ng),
            "sampled": len(picked),
            "verdicts": dict(v),
        }
        if total and picked and "unlabeled" not in v:
            sup, par = v.get("supports", 0) / len(picked), v.get("partial", 0) / len(picked)
            gold_share = 1 - len(ng) / total
            est, est_l = gold_share + (1 - gold_share) * sup, gold_share + (1 - gold_share) * (sup + par)
            out["citations"][run].update({"est_precision": est, "est_precision_lenient": est_l})
            md.append(f"- {run}：答对的题共 {total} 条引用，其中 {len(ng)} 条与金标不重叠；抽 {len(picked)} 条："
                      f"支持答案 {v.get('supports', 0)}、部分支持 {v.get('partial', 0)}、不支持 {v.get('no', 0)}。"
                      f"按此折算，答对题的引用准确率约 {est:.2f}（含部分支持 {est_l:.2f}），金标口径为 {gold_share:.2f}。")  # fmt: skip
        else:
            md.append(
                f"- {run}：答对的题共 {total} 条引用，其中 {len(ng)} 条与金标不重叠；抽查 {len(picked)} 条（待标注）。"
            )
    md += ["", "判定方式：Claude 对照问题、金标答案和被引段落原文判断该段是否支持答案。**模型判定，未经人工核对。**"
           f"抽样按 sha256(种子 `{seed}` + 条目键) 排序取前若干条，可复现。", ""]  # fmt: skip
    if labels.get("citation_notes"):
        md += [labels["citation_notes"], ""]
    if labels.get("conclusions"):
        md += ["## 9. 结论与对定向改进的判断", "", labels["conclusions"], ""]

    res = cx.s.paths.results_dir
    (res / f"{out_name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (res / f"{out_name}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print("\n".join(md))
    print(f"\n已写入 {res / (out_name + '.md')}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["digest", "cite-sample", "report"])
    for k, v in DEFAULT_RUNS.items():
        ap.add_argument(f"--{k}", default=v, help=f"{k} 运行的结果目录名")
    ap.add_argument("--dataset", default="asoiaf_v1")
    ap.add_argument("--labels", type=Path, default=Path("eval/analysis/p5_failure_labels.json"))
    ap.add_argument("--out", default="p5_failure_analysis")
    ap.add_argument("--n-cite", type=int, default=30)
    ap.add_argument("--seed", default="p5-cite")
    args = ap.parse_args()
    cx = Ctx(args)
    if args.cmd == "digest":
        cmd_digest(cx)
    elif args.cmd == "cite-sample":
        cmd_cite_sample(cx, args.n_cite, args.seed)
    else:
        cmd_report(cx, args.labels, args.out, args.n_cite, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
