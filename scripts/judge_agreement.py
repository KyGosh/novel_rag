"""judge 一致率验证（CLAUDE.md 硬性要求：judge 用 DeepSeek 时要用人工打分验证一致率）。

两步：
    # 1. 从若干次运行的 dev 答案中分层抽样，生成核对表（judge 的判定不写进核对表，避免先入为主）
    uv run python scripts/judge_agreement.py sample --run eval/results/..._p4_closed_dev --run ... --n 50
    #    → eval/judge_review/judge_v1_sample.csv（人工核对）+ judge_v1_key.json（系统判定，核对时不看）
    # 2. 核对完成后计算一致率与 Cohen's κ，写入 eval/results/{时间}_judge_agreement/
    uv run python scripts/judge_agreement.py score --review eval/judge_review/judge_v1_sample.csv

核对表列：编号、题型、问题、标准答案、其他写法、系统答案、预标（Claude 预标，填在 pre_label）、
人工（human_label，核对时确认或改写；留空视为同意预标）、备注。标签：correct / partial / wrong / abstain。
分层：judge 判过的答案按（题型 × judge 判定）分层为主体，另抽少量规则判对和弃答的答案检查整条判分链路。
核对表只含问题和简短答案（与评测集同级别的信息），不含原文。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from novel_rag.config import PROJECT_ROOT, get_settings
from novel_rag.eval.dataset import load_dataset
from novel_rag.eval.runner import git_info

REVIEW_DIR = PROJECT_ROOT / "eval" / "judge_review"
LABELS = ("correct", "partial", "wrong", "abstain")
FIELDS = ["no", "key", "type", "question", "gold", "aliases", "answer", "pre_label", "human_label", "note"]


def _h(x: str) -> str:
    return hashlib.sha256(f"judge-sample:{x}".encode()).hexdigest()


def cmd_sample(args: argparse.Namespace) -> int:
    s = get_settings()
    items = {i.id: i for i in load_dataset(s.paths.datasets_dir / args.dataset, s.paths.books_dir, split="dev")}
    pool = []
    for d in args.run:
        d = Path(d)
        sysname = json.loads((d / "run.json").read_text(encoding="utf-8"))["system"]["name"]
        for line in (d / "per_question.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            pool.append({**r, "system": sysname, "run": d.name, "key": f"{sysname}:{r['id']}"})
    # 同一题同一答案只保留一次
    seen, uniq = set(), []
    for r in sorted(pool, key=lambda r: _h(r["key"])):
        sig = (r["id"], r["answer"].strip())
        if sig not in seen:
            seen.add(sig)
            uniq.append(r)
    judged = [r for r in uniq if r["scored_by"] == "judge"]
    rule = [r for r in uniq if r["scored_by"] == "rule"]
    abst = [r for r in uniq if r["scored_by"] == "abstain"]
    n_rule, n_abst = max(1, args.n // 10), max(1, args.n // 12)
    n_judge = args.n - n_rule - n_abst
    strata: dict[tuple[str, str], list] = defaultdict(list)
    for r in judged:
        strata[(r["type"], r["label"])].append(r)
    # 轮流从各层取，保证每种题型、每种判定都有
    chosen, keys = [], sorted(strata)
    while len(chosen) < n_judge and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(chosen) < n_judge:
                chosen.append(strata[k].pop(0))
    chosen += rule[:n_rule] + abst[:n_abst]
    chosen.sort(key=lambda r: _h(r["key"]))  # 打乱顺序，避免按判定分组
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = REVIEW_DIR / f"{args.tag}_sample.csv"
    keyf = REVIEW_DIR / f"{args.tag}_key.json"
    with out.open("w", encoding="utf-8-sig", newline="") as f:  # 带 BOM，Excel 直接打开不乱码
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for i, r in enumerate(chosen, 1):
            it = items[r["id"]]
            w.writerow(
                {
                    "no": i,
                    "key": r["key"],
                    "type": it.type,
                    "question": it.question,
                    "gold": it.answer,
                    "aliases": "、".join(it.answer_aliases),
                    "answer": r["answer"],
                    "pre_label": "",
                    "human_label": "",
                    "note": "",
                }
            )
    keyf.write_text(
        json.dumps(
            {
                "runs": args.run,
                "created": datetime.now().isoformat(timespec="seconds"),
                "items": {
                    r["key"]: {
                        "label": r["label"],
                        "scored_by": r["scored_by"],
                        "judge_reason": r.get("judge_reason"),
                        "run": r["run"],
                    }
                    for r in chosen
                },
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"抽样 {len(chosen)} 条（judge {len(chosen) - n_rule - n_abst}、规则 {n_rule}、弃答 {n_abst}）→ {out}")
    print("分层：", dict(Counter((r["type"], r["label"], r["scored_by"]) for r in chosen)))
    return 0


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[x] * cb[x] for x in labels) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def cmd_score(args: argparse.Namespace) -> int:
    s = get_settings()
    review = Path(args.review)
    key = json.loads(review.with_name(review.name.replace("_sample.csv", "_key.json")).read_text(encoding="utf-8"))
    rows = list(csv.DictReader(review.open(encoding="utf-8-sig")))
    human, system, by, types = [], [], [], []
    for r in rows:
        h = (r["human_label"] or r["pre_label"]).strip()
        if h not in LABELS:
            print(f"第 {r['no']} 行人工标签无效：{h!r}", file=sys.stderr)
            return 1
        k = key["items"][r["key"]]
        human.append(h)
        system.append(k["label"])
        by.append(k["scored_by"])
        types.append(r["type"])
    pre = [r["pre_label"].strip() for r in rows]
    changed = sum(1 for r in rows if r["human_label"].strip() and r["human_label"].strip() != r["pre_label"].strip())

    def agree(idx: list[int]) -> dict[str, float]:
        if not idx:
            return {"n": 0}
        a, b = [human[i] for i in idx], [system[i] for i in idx]
        # 二值口径：correct vs 其他（主指标"准确率"只看 correct）
        a2, b2 = [x == "correct" for x in a], [x == "correct" for x in b]
        return {
            "n": len(idx),
            "agreement": round(sum(x == y for x, y in zip(a, b, strict=True)) / len(idx), 4),
            "kappa": round(cohen_kappa(a, b), 4),
            "agreement_binary": round(sum(x == y for x, y in zip(a2, b2, strict=True)) / len(idx), 4),
            "kappa_binary": round(cohen_kappa([str(x) for x in a2], [str(x) for x in b2]), 4),
        }

    allidx = list(range(len(rows)))
    judged = [i for i in allidx if by[i] == "judge"]
    res = {
        "review_file": str(review.relative_to(PROJECT_ROOT)),
        "n": len(rows),
        "human_changed_prelabels": changed,
        "prelabel_vs_human": agree([i for i in allidx])
        | {"agreement": round(sum(p == h for p, h in zip(pre, human, strict=True)) / len(rows), 4)},
        "overall": agree(allidx),
        "judge_only": agree(judged),
        "by_type": {t: agree([i for i in judged if types[i] == t]) for t in sorted(set(types))},
        "confusion_judge_only": dict(Counter(f"{system[i]}→人工{human[i]}" for i in judged)),
        "disagreements": [
            {"key": rows[i]["key"], "system": system[i], "human": human[i], "by": by[i]}
            for i in allidx
            if human[i] != system[i]
        ],
    }
    out = s.paths.results_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_judge_agreement"
    out.mkdir(parents=True)
    run = {"git": git_info(), "key": key["runs"], "judge": s.llm.tasks["answer_judge"].model_dump()}
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "disagreements"}, ensure_ascii=False, indent=2))
    print(f"不一致 {len(res['disagreements'])} 条；结果目录：{out}")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sample")
    a.add_argument("--run", action="append", required=True)
    a.add_argument("--n", type=int, default=50)
    a.add_argument("--dataset", default="asoiaf_v1")
    a.add_argument("--tag", default="judge_v1")
    b = sub.add_parser("score")
    b.add_argument("--review", required=True)
    args = ap.parse_args()
    return cmd_sample(args) if args.cmd == "sample" else cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
