"""导入质检报告：结构对齐、规范化命中、残留检查、注释配对、文本统计与定稿状态，外加人工抽查清单。

报告写在 data/reports/ 下（不入库），可以包含原文样例。有 error 时流水线不写任何产物。
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any

from novel_rag.ingestion.assemble import VolumeOutput
from novel_rag.ingestion.ir import Document
from novel_rag.ingestion.normalize import NormalizeStats

_RESIDUALS = {
    "note_marker": re.compile(r"\[\d{1,3}\]|［\d{1,3}］|[①-⑳]"),
    "ascii_qmark": re.compile(r"\?"),
    "html_entity": re.compile(r"&[a-zA-Z]{2,8};|&#x?[0-9a-fA-F]{1,6};"),
    "control_char": re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),
    "private_use": re.compile(r"[-]"),
    "replacement_char": re.compile(r"[￼�]"),
    "halfwidth_punct_in_cjk": re.compile(r"(?<=[一-鿿])[,;:!](?=[一-鿿])"),
}
_COMMON = re.compile(
    r"[一-鿿\n，。！？；：、“”‘’（）《》〈〉【】—…·～「」『』﹏ A-Za-z0-9]"
)  # 罕见字符排行时排除的常见字符


def _context(text: str, start: int, end: int, width: int = 12) -> str:
    return f"{text[max(0, start - width) : start]}【{text[start:end]}】{text[end : end + width]}".replace("\n", "⏎")


def _residuals(text: str, n: int) -> dict[str, Any]:
    out = {}
    for name, pattern in _RESIDUALS.items():
        ms = list(pattern.finditer(text))
        out[name] = {"count": len(ms), "samples": [_context(text, m.start(), m.end()) for m in ms[:n]]}
    return out


def _rare_chars(text: str, top: int = 30) -> list[dict[str, Any]]:
    counter = Counter(ch for ch in text if not _COMMON.match(ch))
    rows = []
    for ch, cnt in counter.most_common(top):
        i = text.index(ch)
        rows.append({"char": ch, "codepoint": f"U+{ord(ch):04X}", "count": cnt, "sample": _context(text, i, i + 1)})
    return rows


def _invariants(out: VolumeOutput) -> list[str]:
    """偏移不变量：章节区间切片的首行就是章节标题；区间单调；注释与插图锚点落在所属章节内。"""
    errors = []
    text = out.text
    prev_end = -1
    for c in out.chapters:
        if text[c["char_start"] : c["title_end"]] != c["title"]:
            errors.append(f"第 {c['chapter_idx']} 章的标题区间与标题不一致")
        if c["char_start"] <= prev_end or c["char_end"] < c["title_end"]:
            errors.append(f"第 {c['chapter_idx']} 章的区间不单调")
        prev_end = c["char_end"]
    spans = {c["chapter_idx"]: (c["char_start"], c["char_end"] + 1) for c in out.chapters}
    for row in out.notes + out.image_anchors:
        s, e = spans[row["chapter_idx"]]
        if not s <= row["offset"] <= e:
            errors.append(f"锚点 {row.get('note_id') or row.get('src')} 不在第 {row['chapter_idx']} 章区间内")
    return errors


def build_report(
    doc: Document,
    series_id: str,
    part_stats: list[NormalizeStats],
    outputs: list[VolumeOutput],
    previous_sha: dict[str, str | None],
    lock_states: dict[str, str],
    *,
    book_config_source: str | None,
    sample_size: int,
    force: bool,
) -> dict[str, Any]:
    errors = [{"code": i.code, "message": i.message, "locator": i.locator} for i in doc.issues if i.level == "error"]
    warnings = [
        {"code": i.code, "message": i.message, "locator": i.locator} for i in doc.issues if i.level == "warning"
    ]

    volumes = []
    for out in outputs:
        v = out.volume
        stats = NormalizeStats(
            sample_size=sample_size, sample_sizes=dict(part_stats[0].sample_sizes) if part_stats else {}
        )
        for i in v.part_indices:
            stats.merge(part_stats[i])
        for msg in _invariants(out):
            errors.append({"code": "offset_invariant", "message": f"{v.book_id}：{msg}", "locator": ""})
        lock = lock_states.get(v.book_id, "unlocked")
        if lock == "mismatch":
            entry = {
                "code": "lock_mismatch",
                "locator": v.book_id,
                "message": f"{v.book_id} 已定稿，但本次生成的 text.txt 与锁定的 hash 不一致",
            }
            (warnings if force else errors).append(entry)

        narrative = [c for c in out.chapters if c["section_type"] == "narrative"]
        lengths = [c["char_end"] - c["char_start"] for c in narrative]
        median = statistics.median(lengths) if lengths else 0
        flags = []
        for c in out.chapters:
            n = c["char_end"] - c["char_start"]
            if c["section_type"] == "narrative" and median and (n < 0.25 * median or n > 3 * median):
                flags.append({"chapter_idx": c["chapter_idx"], "title": c["title"], "chars": n, "reason": "长度异常"})
            if c["section_type"] == "narrative" and c["pov"] is None:
                flags.append({"chapter_idx": c["chapter_idx"], "title": c["title"], "chars": n, "reason": "无 POV"})

        residuals = _residuals(out.text, sample_size)
        if residuals["note_marker"]["count"]:
            warnings.append(
                {
                    "code": "residual_note_marker",
                    "locator": v.book_id,
                    "message": f"{v.book_id} 正文中残留 {residuals['note_marker']['count']} 处注释标记样式的文本",
                }
            )

        volumes.append(
            {
                "book_id": v.book_id,
                "number": v.number,
                "number_source": v.number_source,
                "title": v.title,
                "title_source": v.title_source,
                "raw_part_titles": v.raw_titles,
                "text": {
                    "sha256": out.text_sha256,
                    "chars": len(out.text),
                    "lines": out.text.count("\n"),
                    "previous_sha256": previous_sha.get(v.book_id),
                    "changed": previous_sha.get(v.book_id) not in (None, out.text_sha256),
                    "lock": lock,
                },
                "counts": out.manifest["counts"],
                "reading_order": out.manifest["reading_order"],
                "chapters": [
                    {k: c[k] for k in ("chapter_idx", "title", "pov", "section_type", "reading_order")}
                    | {"chars": c["char_end"] - c["char_start"], "paragraphs": c["n_paragraphs"]}
                    for c in out.chapters
                ],
                "chapter_flags": flags,
                "others": [
                    {
                        "title": o.title,
                        "section_type": o.section_type,
                        "chars": o.chars,
                        "images": o.images,
                        "part": o.part_title,
                        "locator": o.locator,
                    }
                    for o in v.others
                ],
                "normalization": {
                    "hits": dict(sorted(stats.hits.items())),
                    "decisions": stats.decisions,
                    "samples": stats.samples,
                },
                "residuals": residuals,
                "rare_chars": _rare_chars(out.text),
                "spot_check": _spot_check(out),
            }
        )

    return {
        "status": "fail" if errors else "pass",
        "series_id": series_id,
        "source": {
            "name": doc.source_name,
            "sha256": doc.source_sha256,
            "format": doc.format,
            "parser_version": doc.parser_version,
            "metadata": doc.metadata,
        },
        "book_config": book_config_source,
        "container": doc.stats.get("container"),
        "toc": doc.stats.get("toc"),
        "parser_stats": {
            k: doc.stats[k] for k in ("empty_paragraphs_dropped", "stray_placeholder_chars") if k in doc.stats
        },
        "errors": errors,
        "warnings": warnings,
        "warning_counts": dict(Counter(w["code"] for w in warnings)),
        "parts": [
            {
                "index": i,
                "title": p.title,
                "sections": len(p.sections),
                "chapters": sum(1 for s in p.sections if s.title is not None),
            }
            for i, p in enumerate(doc.parts)
        ],
        "notes_by_file": doc.stats.get("notes_by_file", {}),
        "spine": doc.stats.get("spine", []),
        "volumes": volumes,
    }


def _spot_check(out: VolumeOutput, width: int = 40) -> list[dict[str, Any]]:
    """每卷前三章与最后一章的开头、结尾，便于人工确认章节边界与正文首尾完整。"""
    picks = out.chapters[:3] + out.chapters[-1:] if len(out.chapters) > 3 else out.chapters
    rows = []
    for c in picks:
        body = out.text[c["title_end"] + 1 : c["char_end"]]
        rows.append(
            {
                "chapter_idx": c["chapter_idx"],
                "title": c["title"],
                "head": body[:width].replace("\n", "⏎"),
                "tail": body[-width:].replace("\n", "⏎"),
            }
        )
    return rows


# ---------------------------------------------------------------- Markdown 摘要


def _row(*cells: object) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header: list[str]) -> list[str]:
    return [_row(*header), _row(*["---"] * len(header))]


def render_markdown(r: dict[str, Any]) -> str:
    L: list[str] = []
    a = L.append
    a(f"# 导入质检报告：{r['series_id']}")
    a("")
    verdict = "通过" if r["status"] == "pass" else "失败"
    a(f"- 结论：**{verdict}**（error {len(r['errors'])}，warning {len(r['warnings'])}）")
    a(f"- 来源：{r['source']['name']}")
    a(f"- sha256：`{r['source']['sha256']}`；解析器 {r['source']['parser_version']}")
    c = r.get("container") or {}
    a(
        f"- 容器：{c.get('files')} 个文件，解压后 {c.get('uncompressed_bytes')} 字节，EPUB {c.get('epub_version')}；"
        f"目录 {r['toc']}"
    )
    a(f"- 书籍配置：{r['book_config'] or '无'}")
    a(f"- 解析统计：{r['parser_stats']}")
    a("")
    if r["errors"]:
        a("## Errors")
        for e in r["errors"]:
            a(f"- `{e['code']}` {e['message']} {e['locator']}")
        a("")
    if r["warnings"]:
        a("## Warnings")
        a(f"按类型计数：{r['warning_counts']}")
        a("")
        for w in r["warnings"][:50]:
            a(f"- `{w['code']}` {w['message']} {w['locator']}")
        a("")

    a("## 卷合并")
    L.extend(
        _table(
            [
                "卷号",
                "卷号来源",
                "卷名",
                "卷名来源",
                "分册",
                "章节",
                "正文/附录",
                "注释",
                "插图锚点",
                "字数",
                "reading_order",
                "定稿",
                "与上次相比",
            ]
        )
    )
    for v in r["volumes"]:
        t, n = v["text"], v["counts"]
        change = "首次" if t["previous_sha256"] is None else ("**变化**" if t["changed"] else "不变")
        a(
            _row(
                v["number"],
                v["number_source"],
                v["title"],
                v["title_source"],
                " / ".join(v["raw_part_titles"]),
                n["chapters"],
                f"{n['narrative']}/{n['appendix']}",
                n["notes"],
                n["image_anchors"],
                t["chars"],
                v["reading_order"],
                t["lock"],
                change,
            )
        )
    a("")
    a("## 注释配对（按文件）")
    L.extend(_table(["文件", "标记", "注释段落", "配对", "未配对标记", "孤立注释"]))
    for path, s in r["notes_by_file"].items():
        a(_row(path, s["markers"], s["notes"], s["paired"], s["unpaired_markers"], s["orphan_notes"]))
    a("")

    for v in r["volumes"]:
        a(f"## {v['book_id']} {v['title']}")
        a(f"text sha256：`{v['text']['sha256']}`")
        a("")
        a("### 规范化命中")
        L.extend(_table(["规则", "次数"]))
        for k, cnt in v["normalization"]["hits"].items():
            a(_row(k, cnt))
        a("")
        for d in v["normalization"]["decisions"]:
            a(f"- {d}")
        a("")
        a("### 残留检查")
        for k, res in v["residuals"].items():
            line = f"- {k}：{res['count']}"
            if res["samples"]:
                line += "；例：" + " / ".join(res["samples"][:3])
            a(line)
        a("")
        if v["chapter_flags"]:
            a("### 章节标记")
            for f in v["chapter_flags"]:
                a(f"- 第 {f['chapter_idx']} 章「{f['title']}」{f['chars']} 字：{f['reason']}")
            a("")
        a("### 未写入 text.txt 的内容")
        for o in v["others"]:
            where = o["part"] or "目录前"
            a(f"- [{o['section_type']}] {o['title'] or '（无标题）'}（{where}）{o['chars']} 字，{o['images']} 图")
        a("")
        a("### 章节表")
        L.extend(_table(["章", "标题", "POV", "类型", "reading_order", "字数", "段落"]))
        for ch in v["chapters"]:
            a(
                _row(
                    ch["chapter_idx"],
                    ch["title"],
                    ch["pov"] or "",
                    ch["section_type"],
                    ch["reading_order"],
                    ch["chars"],
                    ch["paragraphs"],
                )
            )
        a("")

    a("## 人工抽查清单")
    a("")
    a("### 1. 半角 ? 的处理样例（修复为间隔号 / 统一为全角问号）")
    for v in r["volumes"]:
        for rule, to in (("qmark_repair", "·"), ("qmark_fullwidth", "？")):
            samples = v["normalization"]["samples"].get(rule, [])
            if samples:
                a(f"**{v['book_id']} {rule}**（共 {v['normalization']['hits'].get(rule, 0)} 处）")
                for s in samples:
                    a(f"- {s['before']}【{s['match']}→{to}】{s['after']}")
    a("")
    a("### 2. 间隔号统一与其他规则的样例")
    for v in r["volumes"][:1]:
        for rule, samples in v["normalization"]["samples"].items():
            if rule in ("qmark_repair", "qmark_fullwidth"):
                continue
            a(f"- {rule}：" + " / ".join(f"{s['before']}【{s['match']!r}】{s['after']}" for s in samples[:4]))
    a("")
    a("### 3. 章节首尾")
    for v in r["volumes"]:
        a(f"**{v['book_id']}**")
        for s in v["spot_check"]:
            a(f"- 第 {s['chapter_idx']} 章「{s['title']}」开头：{s['head']} …… 结尾：{s['tail']}")
    a("")
    a("### 4. 未配对的注释")
    unpaired = [w for w in r["warnings"] if w["code"] in ("unpaired_marker", "orphan_note")]
    a("无" if not unpaired else "\n".join(f"- {w['message']} {w['locator']}" for w in unpaired))
    a("")
    a("### 5. 罕见字符（前 20）")
    for v in r["volumes"]:
        a(
            f"**{v['book_id']}**："
            + "，".join(f"{x['char']}({x['codepoint']})×{x['count']}" for x in v["rare_chars"][:20])
        )
    a("")
    return "\n".join(L) + "\n"
