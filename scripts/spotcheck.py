"""定稿前的人工抽查清单：从最近一次导入质检报告中摘出指定卷的要点，生成一份便于逐项确认的 Markdown。

用法：
    uv run python scripts/spotcheck.py --books asoiaf-2,asoiaf-3,asoiaf-4,asoiaf-5
输出到 data/reports/spotcheck_{时间}.md（含原文片段，只在本地，不入库）。确认无误后再用
`scripts/ingest.py ... --finalize 2,3,4,5` 定稿。

清单内容（每卷）：
1. 概况：字数、章数、hash、注释数、未写入 text.txt 的内容。
2. 章节边界：每章标题、字数、开头与结尾片段；字数异常的章节标记 ⚠。
3. 注释剥离：每条注释的内容及其在正文中的位置（前后各 20 字，※ 为原标记处）。
4. 规范化：各规则命中次数、按分册的判定、每条规则的样例；残留检查与罕见字符。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import median

from novel_rag.config import get_settings

CTX = 20


def _latest_report(reports_dir: Path) -> Path:
    files = sorted(reports_dir.glob("ingest_*.json"))
    if not files:
        raise SystemExit(f"{reports_dir} 下没有导入质检报告，先运行 scripts/ingest.py")
    return files[-1]


def _one_line(s: str) -> str:
    return s.replace("\n", "⏎").replace("|", "｜")


def render_volume(v: dict, book_dir: Path, sample_per_rule: int) -> list[str]:
    text = (book_dir / "text.txt").read_text(encoding="utf-8")
    notes = [json.loads(x) for x in (book_dir / "notes.jsonl").read_text(encoding="utf-8").splitlines() if x]
    lines = [f"## {v['book_id']} {v['title']}", ""]
    t = v["text"]
    lines += [
        f"- [ ] 概况：{t['chars']} 字，{v['counts']['chapters']} 章（{v['counts']}），reading_order {v['reading_order']}",  # noqa: E501
        f"- sha256 `{t['sha256']}`；定稿状态 {t['lock']}",
        "- 未写入 text.txt 的内容："
        + (
            "；".join(f"[{o['section_type']}] {o['title'] or '（无标题）'} {o['chars']} 字" for o in v["others"])
            or "无"
        ),
        "",
        "### 1. 章节边界（每章开头 / 结尾；⚠ 为字数偏离中位数过多，或结尾不像句末）",
        "",
        "| 章 | 标题 | 字数 | 开头 | 结尾 |",
        "| --- | --- | --- | --- | --- |",
    ]
    chars = {c["chapter_idx"]: c["chars"] for c in v["chapters"]}
    med = median(chars.values())
    for sc in v["spot_check"]:
        n = chars.get(sc["chapter_idx"], 0)
        tail = sc["tail"].rstrip("⏎")
        odd = n < med / 4 or n > med * 3 or not tail.endswith(("。", "”", "！", "？", "…", "」", "』", "）"))
        flag = "⚠ " if odd else ""
        lines.append(
            f"| {flag}{sc['chapter_idx']} | {sc['title']} | {n} | {_one_line(sc['head'][:24])} | {_one_line(tail[-24:])} |"  # noqa: E501
        )
    lines += ["", f"### 2. 注释剥离（{len(notes)} 条，※ 为原标记位置）", ""]
    for n in notes:
        o = n["offset"]
        ctx = _one_line(text[max(0, o - CTX) : o]) + "※" + _one_line(text[o : o + CTX])
        lines.append(f"- [ ] 第 {n['chapter_idx']} 章 {n['label']}「{_one_line(n['text'])}」 ← …{ctx}…")
    if not notes:
        lines.append("- 无")
    norm = v["normalization"]
    lines += ["", "### 3. 规范化", "", "命中次数：" + "，".join(f"{k} {c}" for k, c in sorted(norm["hits"].items()))]
    lines += [f"- {d}" for d in norm.get("decisions", [])]
    for rule, samples in sorted(norm.get("samples", {}).items()):
        lines.append(f"- [ ] **{rule}**（{norm['hits'].get(rule, '?')} 处），样例：")
        for s in samples[:sample_per_rule]:
            lines.append(
                f"  - {_one_line(s.get('before', ''))}【{_one_line(s.get('match', ''))}】{_one_line(s.get('after', ''))}"  # noqa: E501
            )
    lines += ["", "### 4. 残留与罕见字符", ""]
    for k, r in v["residuals"].items():
        if r["count"]:
            lines.append(f"- [ ] {k}：{r['count']}；例：" + " / ".join(_one_line(x) for x in r["samples"][:5]))
    lines.append(
        "- 罕见字符："
        + "；".join(
            f"{c['char']!r}({c['codepoint']})×{c['count']} {_one_line(c['sample'])}" for c in v["rare_chars"][:10]
        )
    )
    return [*lines, ""]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--books", required=True, help="逗号分隔的 book_id")
    parser.add_argument("--report", type=Path, help="质检报告 json，默认取最近一份")
    parser.add_argument("--samples", type=int, default=6, help="每条规范化规则展示几个样例")
    args = parser.parse_args()

    s = get_settings()
    report_path = args.report or _latest_report(s.paths.reports_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    wanted = [b.strip() for b in args.books.split(",") if b.strip()]
    vols = {v["book_id"]: v for v in report["volumes"]}
    missing = [b for b in wanted if b not in vols]
    if missing:
        print(f"报告 {report_path.name} 中没有这些卷：{missing}", file=sys.stderr)
        return 1
    out = [
        "# 定稿前人工抽查清单",
        "",
        f"- 来源报告：`{report_path.name}`（状态 {report['status']}）",
        "- 逐项打勾；发现问题记下卷、章与片段，修规则后重新导入再生成本清单。",
        "",
    ]
    for b in wanted:
        out += render_volume(vols[b], s.paths.books_dir / b, args.samples)
    dest = s.paths.reports_dir / f"spotcheck_{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    dest.write_text("\n".join(out), encoding="utf-8")
    print(f"已写入：{dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
