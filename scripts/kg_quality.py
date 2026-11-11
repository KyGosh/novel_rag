"""图谱质量评估（P3 第十二节，内在指标）：导出审阅表 → 金标（Claude 预标、用户核对）→ 打分。

用法：
    uv run python scripts/kg_quality.py export --kg d9c84c78        # 写 data/reports/kg_quality/{kg}/ 下三张审阅表
    uv run python scripts/kg_quality.py score --kg d9c84c78         # 读金标，写 eval/results/{时间}_kg_quality/
审阅表与金标含原文片段，只在 data/ 下；结果目录只有指标与 ID，不含原文。

1. 抽取：分层抽 20 个 chunk（各卷 + 附录），金标写在 extraction_gold.jsonl：
   {"chunk_id": ..., "entities": [{"name": "原文写法", "type": "Person"}],
    "relations": [{"s": "名", "type": "KIN", "role": "father", "o": "名"}]}
   实体按"类型 + 去称谓规范名"匹配（系统的名字或全名任一相同即算）；关系按（主体、类型、角色、客体）匹配，
   对称关系不分方向，father / mother 与 parent 视为同一类亲属。
2. 消歧：按难例名字抽样提及，金标 disamb_gold.jsonl：{"mid": ..., "gold": "身份标签"}；
   与系统的提及 → 实体映射比较，报告成对 P / R / F1 与 B-cubed（只统计双方都有标签的提及）。
3. 关系证据：随机 100 条关系，金标 relations_gold.jsonl：
   {"key": ..., "verdict": "supported|partial|unsupported"}。
score 加 --gold-kg <旧版本>：用旧图谱的已核对金标评新图谱（抽取和消歧金标与图谱版本无关，直接沿用；
关系证据只保留在新图谱中仍然存在的关系键，被改写或删除的剔除，报告剩余条数）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations

from novel_rag.config import get_settings
from novel_rag.eval.runner import git_info
from novel_rag.ingestion.chunking import chunk_version, load_books, series_book_ids
from novel_rag.kg.build import load_export
from novel_rag.kg.common import extract_dir, kg_root, load_series_kg
from novel_rag.kg.extract import extract_version
from novel_rag.kg.names import normalize_name
from novel_rag.kg.schema import KIN_ROLES, REL_BY_NAME, SYMMETRIC_KIN_ROLES
from novel_rag.llm import get_chat_model
from novel_rag.prompts import load_prompt

SEED = 20261107
HARD_NAMES = (
    "琼恩", "布兰登", "瓦德", "伊耿", "劳勃", "奈德", "艾德", "小恶魔", "猎狗", "小指头", "臭佬", "阿莲", "艾莉亚",
    "艾利亚", "凯特琳", "丹妮", "卓戈", "莱莎", "瑟曦", "詹姆", "山姆", "巴隆", "攸伦", "席恩", "弑君者", "魔山",
    "黑鱼", "熊老", "史坦尼斯", "罗柏", "蓝礼", "梅丽珊卓", "戴佛斯", "提利昂", "珊莎", "布兰",
)  # fmt: skip
PER_NAME = 8


def rel_label(t: str, role: str | None) -> str:
    return REL_BY_NAME[t].label + (f"/{KIN_ROLES[role]}" if role else "")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["export", "score"])
    parser.add_argument("--kg", required=True, help="图谱版本")
    parser.add_argument("--extract-tag", default="all")
    parser.add_argument("--gold-kg", help="score：金标所在的旧图谱版本（默认就是 --kg）")
    args = parser.parse_args()

    s = get_settings()
    cver = chunk_version(s.chunk)
    book_ids = series_book_ids(s.paths.books_dir, cver)
    books, chunks = load_books(s.paths.books_dir, book_ids, cver)
    by_chunk = {c["chunk_id"]: c for c in chunks}
    series = load_series_kg(s.paths.books_dir, book_ids)
    xver = extract_version(
        series.schema, load_prompt("kg_extract"), get_chat_model("extraction").cache_identity(), cver
    )
    records = {
        r["chunk_id"]: r
        for r in map(
            json.loads, (extract_dir(s.paths.data_dir, xver) / f"{args.extract_tag}.jsonl").open(encoding="utf-8")
        )
    }
    kgdir = kg_root(s.paths.data_dir) / args.kg
    tables, manifest = load_export(kgdir)
    m2e = json.loads((kgdir / "internal" / "mention_entity.json").read_text(encoding="utf-8"))
    ents = {e["entity_id"]: e for e in tables["entities"]}
    work = s.paths.reports_dir / "kg_quality" / args.kg
    work.mkdir(parents=True, exist_ok=True)
    gold_dir = s.paths.reports_dir / "kg_quality" / (args.gold_kg or args.kg)
    rng = random.Random(SEED)

    # ---- 抽样（export 与 score 用同一种子，结果一致）----
    narrative = defaultdict(list)
    appendix = []
    for c in chunks:
        (appendix if c["section_type"] == "appendix" else narrative[c["book_id"]]).append(c["chunk_id"])
    ext_sample = [cid for b in sorted(narrative) for cid in rng.sample(narrative[b], 4)] + rng.sample(appendix, 2)
    hard_mids = []
    for name in HARD_NAMES:
        pool = sorted(
            f"{cid}#{m['m']}"
            for cid, rec in records.items()
            for m in rec["mentions"]
            if m["type"] == "Person" and normalize_name(m["name"]).startswith(name)
        )
        hard_mids.extend(rng.sample(pool, min(PER_NAME, len(pool))))
    rels = [r for r in tables["relations"] if REL_BY_NAME[r["type"]].source == "chunk"]
    rel_sample = rng.sample(rels, 100)

    def rel_key(r: dict) -> str:
        return f"{r['subject']}|{r['type']}|{r['role'] or ''}|{r['object']}"

    if args.action == "export":
        lines = [f"# 抽取金标审阅表（{len(ext_sample)} 个 chunk，种子 {SEED}）", ""]
        for cid in ext_sample:
            lines += [f"## {cid}（{by_chunk[cid]['section_type']}）", by_chunk[cid]["text"], ""]
        (work / "extraction_sheet.md").write_text("\n".join(lines), encoding="utf-8")
        sysout = []
        for cid in ext_sample:
            rec = records[cid]
            sysout.append(
                {
                    "chunk_id": cid,
                    "entities": [(m["name"], m["full_name"], m["type"]) for m in rec["mentions"]],
                    "relations": [
                        (
                            rec["mentions"][r["s"]]["name"],
                            rel_label(r["type"], r["role"]),
                            rec["mentions"][r["o"]]["name"],
                        )
                        for r in rec["relations"]
                    ],
                }
            )
        (work / "extraction_system.json").write_text(json.dumps(sysout, ensure_ascii=False, indent=1), encoding="utf-8")
        lines = [f"# 消歧金标审阅表（{len(hard_mids)} 个提及）", ""]
        for mid in hard_mids:
            cid, idx = mid.split("#")
            m = records[cid]["mentions"][int(idx)]
            text = by_chunk[cid]["text"]
            i = normalize_name(text).find(m["name"])
            snip = text[max(0, i - 70) : i + len(m["name"]) + 50].replace("\n", " ") if i >= 0 else text[:120]
            where = f"{by_chunk[cid]['chapter_title']}（ro {by_chunk[cid]['reading_order']}）"
            lines.append(f"- {mid}｜{where}｜【{m['name']}】｜{snip}")
        (work / "disamb_sheet.md").write_text("\n".join(lines), encoding="utf-8")
        lines = ["# 关系证据审阅表（100 条）", ""]
        for r in rel_sample:
            ev = r["evidence"][0]
            quote = books[ev["book_id"]].text[ev["start"] : ev["end"]]
            sname, oname = ents[r["subject"]]["name"], ents[r["object"]]["name"]
            fact = f"{sname} —{rel_label(r['type'], r['role'])}→ {oname}"
            lines.append(f"- {rel_key(r)}｜{fact}｜证据 {r['n_evidence']} 条｜「{quote}」")
        (work / "relations_sheet.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"审阅表写在 {work}")
        return 0

    # ---- 打分 ----
    rules = series.rules
    out: dict = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "kg_version": args.kg,
        "gold_from": args.gold_kg or args.kg,
        "extract_version": xver,
        "seed": SEED,
    }

    def canon(n: str) -> str:
        return normalize_name(rules.canonical(n)).replace("·", "")

    gold_path = gold_dir / "extraction_gold.jsonl"
    if gold_path.is_file():
        tp_e = fp_e = fn_e = tp_r = fp_r = fn_r = 0
        per_chunk = []
        for g in map(json.loads, gold_path.open(encoding="utf-8")):
            rec = records[g["chunk_id"]]
            sys_ents = {(m["type"], canon(m["name"])) for m in rec["mentions"]} | {
                (m["type"], canon(m["full_name"])) for m in rec["mentions"] if m["full_name"]
            }
            gold_ents = {(e["type"], canon(e["name"])) for e in g["entities"]}
            matched_gold = {x for x in gold_ents if x in sys_ents}
            tp_e += len(matched_gold)
            fn_e += len(gold_ents - matched_gold)
            fp_e += sum(
                1
                for m in rec["mentions"]
                if (m["type"], canon(m["name"])) not in gold_ents
                and not (m["full_name"] and (m["type"], canon(m["full_name"])) in gold_ents)
            )

            def norm_rel(sn: str, t: str, role: str | None, on: str) -> tuple:
                role = "parent" if role in ("father", "mother", "parent") else role
                a, b = canon(sn), canon(on)
                sym = REL_BY_NAME[t].symmetric or (t == "KIN" and role in SYMMETRIC_KIN_ROLES)
                return (t, role, *(sorted((a, b)) if sym else (a, b)))

            def variants(i: int, ms: list = rec["mentions"]) -> list[str]:  # noqa: B008
                m = ms[i]
                return [m["name"]] + ([m["full_name"]] if m["full_name"] else [])

            sys_rels = []
            for r in rec["relations"]:
                sys_rels.append(
                    {norm_rel(a, r["type"], r["role"], b) for a in variants(r["s"]) for b in variants(r["o"])}
                )
            gold_rels = {norm_rel(x["s"], x["type"], x.get("role"), x["o"]) for x in g["relations"]}
            hit_gold = {gr for gr in gold_rels if any(gr in sr for sr in sys_rels)}
            tp_r += len(hit_gold)
            fn_r += len(gold_rels - hit_gold)
            fp_r += sum(1 for sr in sys_rels if not (sr & gold_rels))
            per_chunk.append(
                {"chunk_id": g["chunk_id"], "gold_entities": len(gold_ents), "gold_relations": len(gold_rels)}
            )
        out["extraction"] = {
            "chunks": len(per_chunk),
            "entity": {"tp": tp_e, "fp": fp_e, "fn": fn_e, "precision": round(tp_e / max(tp_e + fp_e, 1), 3),
                       "recall": round(tp_e / max(tp_e + fn_e, 1), 3)},
            "relation": {"tp": tp_r, "fp": fp_r, "fn": fn_r, "precision": round(tp_r / max(tp_r + fp_r, 1), 3),
                         "recall": round(tp_r / max(tp_r + fn_r, 1), 3)},
        }  # fmt: skip

    gold_path = gold_dir / "disamb_gold.jsonl"
    if gold_path.is_file():
        gold = {g["mid"]: g["gold"] for g in map(json.loads, gold_path.open(encoding="utf-8")) if g.get("gold")}
        mids = [m for m in gold if m in m2e]
        unmapped = [m for m in gold if m not in m2e]
        sys_lab = {m: m2e[m] for m in mids}
        tp = fp = fn = 0
        for a, b in combinations(mids, 2):
            sg, ss = gold[a] == gold[b], sys_lab[a] == sys_lab[b]
            tp += sg and ss
            fp += ss and not sg
            fn += sg and not ss
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        # B-cubed
        bp = br = 0.0
        for m in mids:
            same_sys = [x for x in mids if sys_lab[x] == sys_lab[m]]
            same_gold = [x for x in mids if gold[x] == gold[m]]
            inter = len([x for x in same_sys if gold[x] == gold[m]])
            bp += inter / len(same_sys)
            br += inter / len(same_gold)
        bp, br = bp / max(len(mids), 1), br / max(len(mids), 1)
        out["disambiguation"] = {
            "mentions_labeled": len(gold),
            "mentions_scored": len(mids),
            "mentions_unmapped_by_system": len(unmapped),
            "gold_entities": len(set(gold.values())),
            "pairwise": {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(p + r, 1e-9), 3)},
            "b_cubed": {
                "precision": round(bp, 3),
                "recall": round(br, 3),
                "f1": round(2 * bp * br / max(bp + br, 1e-9), 3),
            },
            "errors": [
                {"mid": a, "gold": gold[a], "system": sys_lab[a]}
                for a in mids
                if Counter(gold[x] for x in mids if sys_lab[x] == sys_lab[a]).most_common(1)[0][0] != gold[a]
            ][:50],
        }

    gold_path = gold_dir / "relations_gold.jsonl"
    if gold_path.is_file():
        nev = {rel_key(r): r["n_evidence"] for r in rels}
        all_rows = list(map(json.loads, gold_path.open(encoding="utf-8")))
        gold_rows = [g for g in all_rows if g["key"] in nev]  # 旧金标里在新图谱中已被改写或删除的关系剔除

        def rates(vs: list[str]) -> dict:
            c = Counter(vs)
            n = max(len(vs), 1)
            return {"n": len(vs), "verdicts": dict(c), "support_rate": round(c["supported"] / n, 3),
                    "support_rate_incl_partial": round((c["supported"] + c["partial"]) / n, 3)}  # fmt: skip

        out["relation_evidence"] = {
            "gold_rows": len(all_rows),
            "dropped_not_in_graph": [g["key"] for g in all_rows if g["key"] not in nev],
            **rates([g["verdict"] for g in gold_rows]),
            "by_evidence_count": {
                "n_evidence=1": rates([g["verdict"] for g in gold_rows if nev[g["key"]] == 1]),
                "n_evidence>=2": rates([g["verdict"] for g in gold_rows if nev[g["key"]] >= 2]),
            },
        }
    review = gold_dir / "review_status.txt"
    out["review"] = review.read_text(encoding="utf-8").strip() if review.is_file() else "Claude 预标（待用户核对）"
    res = s.paths.results_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_kg_quality"
    res.mkdir(parents=True)
    (res / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "git"}, ensure_ascii=False, indent=2))
    print(f"结果：{res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
