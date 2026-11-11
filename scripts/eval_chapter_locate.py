"""章节定位诊断：摘要检索能否把金标证据所在的章节排进前 N？与"chunk 检索前 K 名所在的章节"对比。

指标：章节召回@N = 金标证据章节中出现在前 N 个章节里的比例（按题平均），按题型报告。
chunk 一侧的章节列表按 chunk 排名去重得到（同一章只算第一次出现）。结果写 eval/results/{时间}_{名称}/。
用法：
    uv run python scripts/eval_chapter_locate.py --dataset asoiaf_v0 --name p3a_chapter_locate
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime

from novel_rag.config import get_settings
from novel_rag.corpus import BookText
from novel_rag.eval.dataset import DatasetManifest, load_dataset
from novel_rag.eval.runner import TYPE_NAMES, TYPE_ORDER, git_info
from novel_rag.retrieval.factory import build_retriever, build_summary_retriever
from novel_rag.retrieval.filters import SearchFilter

NS = (1, 3, 5, 10, 20)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--spoiler-filter", action="store_true")
    args = parser.parse_args()

    s = get_settings()
    ds = s.paths.datasets_dir / args.dataset
    items = load_dataset(ds, s.paths.books_dir, split=args.split)
    manifest = DatasetManifest.model_validate_json((ds / "manifest.json").read_text(encoding="utf-8"))
    scope = tuple(manifest.books)
    books = {b: BookText.load(s.paths.books_dir / b) for b in scope}
    summ = build_summary_retriever(s, scope)
    chunks = build_retriever(s, scope, "hybrid")
    rows = []
    for it in items:
        flt = SearchFilter(book_ids=scope)
        if args.spoiler_filter:
            flt = flt.with_max_reading_order(it.max_reading_order)
        gold = {books[e.book_id].chapter(e.chapter_idx).reading_order for e in it.evidence}
        s_chaps = [int(h.fields["reading_order"]) for h in summ.search(it.question, max(NS), flt)]
        c_chaps: list[int] = []
        for h in chunks.search(it.question, 40, flt):
            ro = int(h.fields["reading_order"])
            if ro not in c_chaps:
                c_chaps.append(ro)
        rec = {"id": it.id, "type": it.type, "n_gold_chapters": len(gold), "summary": {}, "chunk": {}}
        for n in NS:
            rec["summary"][n] = len(gold & set(s_chaps[:n])) / len(gold)
            rec["chunk"][n] = len(gold & set(c_chaps[:n])) / len(gold)
        rec["summary_top"] = s_chaps[:10]
        rec["chunk_top"] = c_chaps[:10]
        rows.append(rec)

    def agg(rs: list[dict]) -> dict:
        return {
            src: {f"ch_recall@{n}": round(sum(r[src][n] for r in rs) / len(rs), 3) for n in NS}
            for src in ("summary", "chunk")
        } | {"n": len(rs)}

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    metrics = {"overall": agg(rows), "by_type": {t: agg(by_type[t]) for t in TYPE_ORDER if by_type[t]}}
    out = s.paths.results_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.name}"
    out.mkdir(parents=True)
    run = {
        "name": args.name,
        "git": git_info(),
        "dataset": args.dataset,
        "split": args.split,
        "spoiler_filter": args.spoiler_filter,
        "summary_retriever": summ.describe(),
        "chunk_retriever": chunks.describe(),
        "config": s.snapshot(),
    }
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "per_question.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    lines = [f"# {args.name}：章节召回@N（摘要检索 vs chunk 检索前 40 名所在章节）", ""]
    lines.append(
        "| 分组 | 题数 | " + " | ".join(f"摘要@{n}" for n in NS) + " | " + " | ".join(f"chunk@{n}" for n in NS) + " |"
    )
    lines.append("|---" * (2 + 2 * len(NS)) + "|")
    for label, m in [
        *((TYPE_NAMES[t], metrics["by_type"][t]) for t in metrics["by_type"]),
        ("**总体**", metrics["overall"]),
    ]:
        lines.append(
            f"| {label} | {m['n']} | "
            + " | ".join(f"{m['summary'][f'ch_recall@{n}']:.3f}" for n in NS)
            + " | "
            + " | ".join(f"{m['chunk'][f'ch_recall@{n}']:.3f}" for n in NS)
            + " |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"结果目录：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
