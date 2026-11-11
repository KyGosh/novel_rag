"""人物类别判定质量（P3.1 验收 2）：导出审阅表 → 金标（Claude 预标、用户核对）→ 打分。

用法：
    uv run python scripts/kg_category_quality.py export --kg 7e9c8e49   # 写 category_sheet.md
    uv run python scripts/kg_category_quality.py score --kg 7e9c8e49    # 读 category_gold.jsonl
审阅表与金标在 data/reports/kg_quality/{kg}/，结果写 eval/results/{时间}_kg_category/。

样本：系统判为“动物”和“非角色”的实体全部入样；判为“人”的按提及数分层（1 / 2–4 / 5–19 / ≥20），每层随机 25 个。
金标 category_gold.jsonl：{"entity_id": ..., "gold": "人|动物|非角色", "note": ...}。
打分：各类 P / R（“人”的召回只在分层样本上有意义，按层权重还原成全体估计）、混淆矩阵、判错清单。
审阅表含原文片段，只在 data/ 下；结果目录只有指标与 ID。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime

from novel_rag.config import get_settings
from novel_rag.eval.runner import git_info
from novel_rag.ingestion.chunking import chunk_version, load_books, series_book_ids
from novel_rag.kg.common import kg_root
from novel_rag.kg.schema import ANIMAL, HUMAN, NON_CHARACTER

SEED = 20261111
STRATA = ((1, 1), (2, 4), (5, 19), (20, 10**9))
PER_STRATUM = 25
LABELS = (HUMAN, ANIMAL, NON_CHARACTER)


def stratum(n: int) -> int:
    return next(i for i, (lo, hi) in enumerate(STRATA) if lo <= n <= hi)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["export", "score", "compare"])
    parser.add_argument("--kg", required=True, help="图谱版本")
    parser.add_argument("--gold-kg", help="compare：金标所在的旧图谱版本")
    args = parser.parse_args()

    s = get_settings()
    kgdir = kg_root(s.paths.data_dir) / args.kg
    if args.action == "compare":
        return compare(s, kgdir, args.kg, args.gold_kg)
    cats = [json.loads(line) for line in (kgdir / "internal" / "categories.jsonl").open(encoding="utf-8")]
    work = s.paths.reports_dir / "kg_quality" / args.kg
    work.mkdir(parents=True, exist_ok=True)

    # ---- 抽样（export 与 score 用同一种子，结果一致）----
    rng = random.Random(SEED)
    sample = [c for c in cats if c["category"] != HUMAN]
    humans = defaultdict(list)
    for c in cats:
        if c["category"] == HUMAN:
            humans[stratum(c["n_mentions"])].append(c)
    for k in range(len(STRATA)):
        pool = sorted(humans[k], key=lambda c: c["entity_id"])
        sample += rng.sample(pool, min(PER_STRATUM, len(pool)))

    if args.action == "export":
        cver = chunk_version(s.chunk)
        _, chunks = load_books(s.paths.books_dir, series_book_ids(s.paths.books_dir, cver), cver)
        by_chunk = {c["chunk_id"]: c for c in chunks}
        m2e = json.loads((kgdir / "internal" / "mention_entity.json").read_text(encoding="utf-8"))
        first: dict[str, str] = {}
        for mid in sorted(m2e):  # chunk_id 按阅读顺序排序，取最早的提及
            first.setdefault(m2e[mid], mid.split("#")[0])
        lines = [f"# 人物类别审阅表（{len(sample)} 个实体，种子 {SEED}）", ""]
        for i, c in enumerate(sample, 1):
            names = [c["name"], *c.get("aliases", [])]
            snip = ""
            cid = first.get(c["entity_id"])
            if cid in by_chunk:
                text = by_chunk[cid]["text"]
                j = next((p for p in (text.find(n) for n in names) if p >= 0), -1)
                if j >= 0:
                    n = next(n for n in names if text.find(n) == j)
                    snip = text[max(0, j - 60) : j] + "【" + n + "】" + text[j + len(n) : j + len(n) + 40]
                    snip = snip.replace("\n", " ")
            lines.append(
                f"- C{i:03d}｜{c['entity_id']}｜{c['category']}｜n={c['n_mentions']}｜"
                f"别名：{'、'.join(c.get('aliases', [])) or '无'}｜群体：{'、'.join(c.get('groups', [])) or '无'}｜"
                f"描述：{' / '.join(c['descriptions'][:4])}｜理由：{c.get('reason', '')}｜原文：{snip or '（未定位）'}"
            )
        (work / "category_sheet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"审阅表：{work / 'category_sheet.md'}（{len(sample)} 个）")
        return 0

    gold = {g["entity_id"]: g for g in map(json.loads, (work / "category_gold.jsonl").open(encoding="utf-8"))}
    missing = [c["entity_id"] for c in sample if c["entity_id"] not in gold]
    if missing:
        print(f"金标缺 {len(missing)} 个：{missing[:5]}")
        return 1
    conf = Counter((c["category"], gold[c["entity_id"]]["gold"]) for c in sample)
    # 判为“人”的只抽了样：按层把样本计数放大回总体，用于估计各类召回
    weight = {}
    for k in range(len(STRATA)):
        n_s = sum(1 for c in sample if c["category"] == HUMAN and stratum(c["n_mentions"]) == k)
        weight[k] = len(humans[k]) / n_s if n_s else 0.0
    est = Counter()
    for c in sample:
        w = weight[stratum(c["n_mentions"])] if c["category"] == HUMAN else 1.0
        est[(c["category"], gold[c["entity_id"]]["gold"])] += w

    def pr(label: str) -> dict:
        sys_n = sum(v for (p, _), v in conf.items() if p == label)
        tp = conf[(label, label)]
        gold_est = sum(v for (_, g), v in est.items() if g == label)
        return {
            "system": sys_n,
            "precision": round(tp / sys_n, 3) if sys_n else None,
            "recall_est": round(est[(label, label)] / gold_est, 3) if gold_est else None,
        }

    out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "kg_version": args.kg,
        "seed": SEED,
        "sample": {"total": len(sample), "human_strata": [len(humans[k]) for k in range(len(STRATA))]},
        "per_label": {lb: pr(lb) for lb in LABELS},
        "confusion": {f"{p}->{g}": v for (p, g), v in sorted(conf.items())},
        "errors": [
            {
                "entity_id": c["entity_id"],
                "system": c["category"],
                "gold": gold[c["entity_id"]]["gold"],
                "note": gold[c["entity_id"]].get("note", ""),
            }
            for c in sample
            if gold[c["entity_id"]]["gold"] != c["category"]
        ],  # fmt: skip
    }
    review = work / "category_review_status.txt"
    out["review"] = review.read_text(encoding="utf-8").strip() if review.is_file() else "Claude 预标（待用户核对）"
    res = s.paths.results_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_kg_category"
    res.mkdir(parents=True)
    (res / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("git", "errors")}, ensure_ascii=False, indent=2))
    print(f"判错 {len(out['errors'])} 个；结果：{res}")
    return 0


def compare(s, kgdir, kg: str, gold_kg: str | None) -> int:
    """用旧图谱的类别金标评新图谱（实体 ID 由消歧决定，类别判定前后不变）：
    被称呼归属并进别人的实体，按目标实体的类别计；泛称的提及全部按章挂靠出去的，按挂靠目标计；
    泛称因有证据指向不同的人而被删掉的，计为非角色（和非角色一样不在图谱里）；其余仍在图谱里的称呼按"人"计。
    样本是按旧图谱的系统判定抽的，精确率 / 召回不再有意义，只报告混淆矩阵与各类错误数，并与旧系统判定对照。"""
    if not gold_kg:
        print("compare 需要 --gold-kg")
        return 1
    gold_dir = s.paths.reports_dir / "kg_quality" / gold_kg
    gold = [json.loads(line) for line in (gold_dir / "category_gold.jsonl").open(encoding="utf-8")]
    old = {
        c["entity_id"]: c["category"]
        for c in map(
            json.loads, (kg_root(s.paths.data_dir) / gold_kg / "internal" / "categories.jsonl").open(encoding="utf-8")
        )
    }
    new = {
        c["entity_id"]: c["category"]
        for c in map(json.loads, (kgdir / "internal" / "categories.jsonl").open(encoding="utf-8"))
    }
    ep_path = kgdir / "internal" / "epithets.jsonl"
    merged, generic = {}, set()
    present = {e["entity_id"] for e in map(json.loads, (kgdir / "entities.jsonl").open(encoding="utf-8"))}
    if ep_path.is_file():
        for a in map(json.loads, ep_path.open(encoding="utf-8")):
            eid = a["entity_id" if "entity_id" in a else "epithet_id"]
            if a["status"] == "merged":
                merged[eid] = a["target_id"]
            elif a.get("generic") and eid not in present:
                if a.get("multi"):
                    generic.add(eid)  # 有证据指向不同的人，剩余提及连同实体删掉
                elif a.get("links"):
                    merged[eid] = a["links"][0][2]  # 提及全部按章挂靠出去了，按挂靠目标计

    def label(eid: str) -> str:
        if eid in generic:
            return "非角色"
        if eid in merged:
            eid = merged[eid]
        c = new.get(eid, "缺失")
        return HUMAN if c == "称呼" else c

    conf_new, conf_old = Counter(), Counter()
    changed = []
    for g in gold:
        eid = g["entity_id"]
        n, o = label(eid), old[eid]
        conf_new[(n, g["gold"])] += 1
        conf_old[(o, g["gold"])] += 1
        if (n == g["gold"]) != (o == g["gold"]):
            changed.append(
                {
                    "entity_id": eid,
                    "gold": g["gold"],
                    "old": o,
                    "new": n,
                    "new_raw": new.get(eid),
                    "merged_into": merged.get(eid),
                    "generic": eid in generic,
                }
            )

    def errs(conf: Counter) -> dict:
        return {f"{p}->{g}": v for (p, g), v in sorted(conf.items()) if p != g}

    out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "kg_version": kg,
        "gold_from": gold_kg,
        "n": len(gold),
        "errors_new": sum(v for (p, g), v in conf_new.items() if p != g),
        "errors_old": sum(v for (p, g), v in conf_old.items() if p != g),
        "error_types_new": errs(conf_new),
        "error_types_old": errs(conf_old),
        "epithets_merged_in_sample": sum(1 for g in gold if g["entity_id"] in merged),
        "generic_dropped_in_sample": sum(1 for g in gold if g["entity_id"] in generic),
        "changed": changed,
    }
    review = gold_dir / "category_review_status.txt"
    out["review"] = review.read_text(encoding="utf-8").strip() if review.is_file() else "Claude 预标（待用户核对）"
    res = s.paths.results_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_kg_category_compare"
    res.mkdir(parents=True)
    (res / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "git"}, ensure_ascii=False, indent=2))
    print(f"结果：{res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
