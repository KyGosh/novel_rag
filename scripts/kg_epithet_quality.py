"""称呼归属质量（P3.1 验收 7）：导出抽查表 → 金标（Claude 预标、用户核对）→ 打分。

用法：
    uv run python scripts/kg_epithet_quality.py export --kg <版本>   # 写 epithet_sheet.md
    uv run python scripts/kg_epithet_quality.py score --kg <版本>    # 读 epithet_gold.jsonl
审阅表与金标在 data/reports/kg_quality/{kg}/，结果写 eval/results/{时间}_kg_epithet/。

样本分三层（种子固定）：
  - 别名：作为别名并入真实人物的称呼（status=merged），随机 N_ALIAS 个；
  - 挂靠：泛称按章挂靠（status=linked），按"称呼 × 章"随机 N_LINK 条（旧图谱没有这一层）；
  - 未归属：其余（无目标、冲突、支持度不足、"某人的某某"、泛称一章也没挂上），随机 N_OTHER 个。
金标 epithet_gold.jsonl：{"key", "stratum", "gold", "first_ok", "note"}
  - gold：按抽查表给出的段落，这个称呼指的是谁——实体 ID；看不出来为"无"；不同段落指不同的人为"多人"；
    只指一个特定的人、但这个人没有名字（或名字没在段落里出现）为"单人无名"。
  - 别名层用严格口径：称呼只要有一段指别人，就是"多人"（泛称不该成为任何人的别名）。
  - first_ok：只对别名层归属正确的填写——别名开始可见的那一段，文字本身能否看出指的是谁（防剧透）。
打分：别名精确率、挂靠精确率、漏并率（未归属层里其实能确定唯一目标的比例）、
     误删率（被当作泛称删掉、其实只指一个人的比例）、可见起点合格率。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from novel_rag.config import get_settings
from novel_rag.eval.runner import git_info
from novel_rag.ingestion.chunking import chunk_version, load_books, series_book_ids
from novel_rag.kg.common import extract_dir, kg_root
from novel_rag.kg.epithet import find_raw
from novel_rag.kg.names import normalize_name

SEED = 20261112
N_ALIAS, N_LINK, N_OTHER = 30, 20, 20
WINDOW, MAX_PASSAGES = 150, 6
OTHER_STATUSES = ("no_target", "conflict", "low_support", "possessive", "generic_unlinked")


def sample_items(atts: list[dict[str, Any]], legacy: bool = False) -> list[dict[str, Any]]:
    """抽查条目：{key, stratum, epithet_id, name, target_id, chapter, att}，按层排列。
    legacy：fd032ef1 的两层抽样（归属成功 40 + 未归属 20），用来复现它的分数。"""
    rng = random.Random(SEED)
    if legacy:
        mg = sorted((a for a in atts if a["status"] == "merged"), key=lambda a: a["epithet_id"])
        ot = sorted(
            (a for a in atts if a["status"] in ("no_target", "conflict", "low_support")), key=lambda a: a["epithet_id"]
        )
        picked = [(a, "alias") for a in rng.sample(mg, min(40, len(mg)))]
        picked += [(a, "other") for a in rng.sample(ot, min(20, len(ot)))]
        return [
            {"key": a["epithet_id"], "stratum": st, "target_id": a["target_id"] if st == "alias" else None,
             "chapter": None, "att": a, "epithet_id": a["epithet_id"], "name": a["name"]}
            for a, st in picked
        ]  # fmt: skip
    merged = sorted((a for a in atts if a["status"] == "merged"), key=lambda a: a["epithet_id"])
    other = sorted((a for a in atts if a["status"] in OTHER_STATUSES), key=lambda a: a["epithet_id"])
    links = sorted(
        ((a, tuple(x)) for a in atts if a["status"] == "linked" for x in a.get("links", [])),
        key=lambda t: (t[0]["epithet_id"], t[1][0], t[1][1]),
    )
    out = []
    for a in rng.sample(merged, min(N_ALIAS, len(merged))):
        out.append({"key": a["epithet_id"], "stratum": "alias", "target_id": a["target_id"], "chapter": None, "att": a})
    for a, (b, c, tid, _) in rng.sample(links, min(N_LINK, len(links))):
        key = f"{a['epithet_id']}@{b}:{c:04d}"
        out.append({"key": key, "stratum": "link", "target_id": tid, "chapter": (b, c), "att": a})
    for a in rng.sample(other, min(N_OTHER, len(other))):
        out.append({"key": a["epithet_id"], "stratum": "other", "target_id": None, "chapter": None, "att": a})
    for it in out:
        it["epithet_id"], it["name"] = it["att"]["epithet_id"], it["att"]["name"]
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["export", "score"])
    parser.add_argument("--kg", required=True, help="图谱版本")
    args = parser.parse_args()

    s = get_settings()
    kgdir = kg_root(s.paths.data_dir) / args.kg
    atts = [json.loads(line) for line in (kgdir / "internal" / "epithets.jsonl").open(encoding="utf-8")]
    work = s.paths.reports_dir / "kg_quality" / args.kg
    work.mkdir(parents=True, exist_ok=True)
    gold_path = work / "epithet_gold.jsonl"
    gold_rows = [json.loads(line) for line in gold_path.open(encoding="utf-8")] if gold_path.is_file() else []
    # 旧格式金标（以 epithet_id 为键、没有分层）：按旧的两层抽样复现
    legacy = bool(gold_rows) and "key" not in gold_rows[0]
    sample = sample_items(atts, legacy=legacy)

    if args.action == "export":
        manifest = json.loads((kgdir / "manifest.json").read_text(encoding="utf-8"))
        xver = manifest["version_parts"]["extract_version"]
        cver = chunk_version(s.chunk)
        _, chunks = load_books(s.paths.books_dir, series_book_ids(s.paths.books_dir, cver), cver)
        text = {c["chunk_id"]: c["text"] for c in chunks}
        where = {c["chunk_id"]: f"{c['chapter_title']}（ro {c['reading_order']}）" for c in chunks}
        m2e = json.loads((kgdir / "internal" / "mention_entity.json").read_text(encoding="utf-8"))
        ents = {e["entity_id"]: e for e in map(json.loads, (kgdir / "entities.jsonl").open(encoding="utf-8"))}
        # 称呼原来的提及：写法是这个称呼的，按层限定现在挂在哪个实体上
        names = {it["key"]: normalize_name(it["name"]) for it in sample}
        found: dict[str, list[tuple[int, str]]] = defaultdict(list)
        records = map(json.loads, (extract_dir(s.paths.data_dir, xver) / "all.jsonl").open(encoding="utf-8"))
        for rec in records:
            for m in rec["mentions"]:
                nm = normalize_name(m["name"])
                eid = m2e.get(f"{rec['chunk_id']}#{m['m']}")
                for it in sample:
                    if nm != names[it["key"]]:
                        continue
                    if it["stratum"] == "alias":
                        ok = eid in (it["epithet_id"], it["target_id"])
                    elif it["stratum"] == "link":
                        ok = eid == it["target_id"] and (rec["book_id"], rec["chapter_idx"]) == it["chapter"]
                    else:
                        ok = eid in (it["epithet_id"], None)
                    if ok:
                        found[it["key"]].append((rec["reading_order"], rec["chunk_id"]))
        title = {"alias": "别名（专属称呼并入）", "link": "挂靠（泛称按章）", "other": "未归属"}
        lines = [
            f"# 称呼归属抽查表（{len(sample)} 条，种子 {SEED}）",
            "",
            "> 每条给出称呼的几段原文（【】标出）。判断：按段落文字，这个称呼指的是谁。",
            "> 别名层用严格口径：只要有一段指别人，就算“多人”（泛称不该成为任何人的别名）。",
            "> 挂靠层只看这一章：这一章里的这个称呼指的是不是目标。",
            "",
        ]
        cur = None
        for i, it in enumerate(sample, 1):
            if it["stratum"] != cur:
                cur = it["stratum"]
                lines += [f"# {title[cur]}", ""]
            a = it["att"]
            tgt = ents.get(it["target_id"] or "", {})
            votes = "、".join(f"{k.split(':', 2)[-1]}×{v}" for k, v in a["votes"].items()) or "无"
            extra = f"　章：{it['chapter'][0]}:{it['chapter'][1]}" if it["chapter"] else ""
            vis = f"　可见起点：{a['visible_from']}" if it["stratum"] == "alias" else ""
            lines += [
                f"## P{i:02d}　{it['name']}",
                f"`{it['key']}`　状态：{a['status']}　模型判：{a.get('kind') or '—'}　目标：{tgt.get('name', '无')}"
                f"{'（`' + it['target_id'] + '`）' if it['target_id'] else ''}{extra}"
                f"　票数：{votes}（判定 {a['n_passages']} 段）{vis}　依据：{'；'.join(a['basis']) or '无'}",
                "",
            ]
            seen: set[str] = set()
            for _ro, cid in sorted(set(found[it["key"]])):
                if cid in seen or len(seen) >= MAX_PASSAGES:
                    continue
                t = text[cid]
                span = find_raw(t, names[it["key"]])
                if span is None:
                    continue
                seen.add(cid)
                j, k = span
                snip = (t[max(0, j - WINDOW) : j] + "【" + t[j:k] + "】" + t[k : k + WINDOW]).replace("\n", " ")
                mark = "　← 可见起点" if it["stratum"] == "alias" and cid == a["first_chunk"] else ""
                lines += [f"- [{where[cid]}]{mark} {snip}", ""]
        (work / "epithet_sheet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"抽查表：{work / 'epithet_sheet.md'}（{dict(Counter(it['stratum'] for it in sample))}）")
        return 0

    gold = {r.get("key", r.get("epithet_id")): r for r in gold_rows}
    missing = [it["key"] for it in sample if it["key"] not in gold]
    if missing:
        print(f"金标缺 {len(missing)} 个：{missing[:5]}")
        return 1
    layer = {st: [it for it in sample if it["stratum"] == st] for st in ("alias", "link", "other")}

    def g(it: dict[str, Any]) -> str:
        return gold[it["key"]]["gold"]

    def rate(n: int, d: int) -> float | None:
        return round(n / d, 3) if d else None

    correct = [it for it in layer["alias"] if g(it) == it["target_id"]]
    link_ok = [it for it in layer["link"] if g(it) == it["target_id"]]
    missed = [it for it in layer["other"] if ":" in g(it)]  # 金标是实体 ID
    # 被删掉的泛称：一章也没挂上、且有证据指向不同的人（multi）
    dropped = [it for it in layer["other"] if it["att"]["status"] == "generic_unlinked" and it["att"].get("multi")]
    wrong_drop = [it for it in dropped if g(it) not in ("无", "多人")]
    first = [gold[it["key"]].get("first_ok") for it in correct]
    out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "kg_version": args.kg,
        "seed": SEED,
        "legacy_sampling": legacy,
        "population": dict(Counter(a["status"] for a in atts)),
        "n_links": sum(len(a.get("links", [])) for a in atts),
        "sample": {k: len(v) for k, v in layer.items()},
        "alias_precision": rate(len(correct), len(layer["alias"])),
        "link_precision": rate(len(link_ok), len(layer["link"])),
        "missed_rate": rate(len(missed), len(layer["other"])),
        "generic_dropped_in_sample": len(dropped),
        "wrong_drop_rate": rate(len(wrong_drop), len(dropped)),
        "visible_from_ok": rate(sum(1 for f in first if f), len(first)),
        "errors": [
            {
                "key": it["key"],
                "stratum": it["stratum"],
                "system": it["target_id"] or it["att"]["status"],
                "gold": g(it),
                "note": gold[it["key"]].get("note", ""),
            }
            for it in sample
            if (it["stratum"] in ("alias", "link") and g(it) != it["target_id"]) or it in missed or it in wrong_drop
        ],  # fmt: skip
    }
    review = work / "epithet_review_status.txt"
    out["review"] = review.read_text(encoding="utf-8").strip() if review.is_file() else "Claude 预标（待用户核对）"
    res = s.paths.results_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_kg_epithet"
    res.mkdir(parents=True)
    (res / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "git"}, ensure_ascii=False, indent=2))
    print(f"结果：{res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
