"""评测集 v1（多卷）出题工具：导出 / 查看任务、导入笔记与候选题、独立复核、冻结。

规范：prompts/qgen_guidelines.v2.md。

用法（工作目录 data/eval_work/{dataset}/，含原文，不入库）：
    uv run python scripts/gen_questions_v1.py export --round 1 --dataset asoiaf_v1
    uv run python scripts/gen_questions_v1.py show --round 1 --dataset asoiaf_v1 --start 0 --end 20
    uv run python scripts/gen_questions_v1.py notes --dataset asoiaf_v1     # 人物标注与事实笔记 → notes_r1.jsonl
    uv run python scripts/gen_questions_v1.py import --round 1 --dataset asoiaf_v1   # 第一轮事实题
    uv run python scripts/gen_questions_v1.py export --round 2 --dataset asoiaf_v1   # 按身份配对
    uv run python scripts/gen_questions_v1.py import --round 2 --dataset asoiaf_v1
    uv run python scripts/gen_questions_v1.py review-sheet --dataset asoiaf_v1 --start 0 --end 40
    uv run python scripts/gen_questions_v1.py review-import --dataset asoiaf_v1 --verdicts path.jsonl
    uv run python scripts/gen_questions_v1.py freeze --dataset asoiaf_v1
卷默认取全部已定稿的卷。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from novel_rag.config import get_settings
from novel_rag.corpus import BookText
from novel_rag.eval.qgen import series

REVIEW_METHOD = (
    "自动校验（证据定位、预筛、重复检测、版权片段检查）+ Claude 独立复核（只看问题、答案和证据原文），未经人工复核"
)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "action", choices=["export", "show", "notes", "import", "review-sheet", "review-import", "freeze", "stats"]
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--round", type=int, choices=[1, 2])
    parser.add_argument("--books", help="逗号分隔的 book_id；默认为全部已定稿的卷")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=20)
    parser.add_argument("--with-text", action="store_true", help="第二轮也显示窗口原文")
    parser.add_argument("--verdicts", type=Path)
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()

    s = get_settings()
    cfg = s.evalgen
    if args.books:
        ids = [b.strip() for b in args.books.split(",")]
    else:
        ids = sorted(p.parent.name for p in s.paths.books_dir.glob("*/text.lock"))
    books = {b: BookText.load(s.paths.books_dir / b) for b in ids}
    work = s.paths.eval_work_dir / args.dataset

    if args.action == "export":
        if args.round == 1:
            out = series.export_round1(books, cfg, work, cfg.v1_fact_windows)
        else:
            out = series.export_round2(books, cfg, work, cfg.v1_combo_counts)
        rows = series.read_jsonl(out)
        extra = {}
        if args.round == 2:
            from collections import Counter

            extra = {
                "by_type": dict(Counter(r["type"] for r in rows)),
                "cross_volume": dict(Counter(r["type"] for r in rows if r["cross_volume"])),
            }
        print(json.dumps({"tasks": len(rows), "file": str(out), **extra}, ensure_ascii=False, indent=2))
    elif args.action == "show":
        print(series.render_tasks(work, args.round, args.start, args.end, with_text=args.with_text))
    elif args.action == "notes":
        print(json.dumps(series.import_notes_r1(books, cfg, work), ensure_ascii=False, indent=2))
    elif args.action == "import":
        stats = series.import_round(books, cfg, work, args.round)
        print(json.dumps({k: v for k, v in stats.items() if k != "issues"}, ensure_ascii=False, indent=2))
        for i in stats["issues"]:
            print(f"[issue] {i}")
    elif args.action == "review-sheet":
        print(series.review_sheet(books, work, args.start, args.end))
    elif args.action == "review-import":
        print(json.dumps(series.import_review(work, args.verdicts, "claude-independent"), ensure_ascii=False))
    elif args.action == "stats":
        from novel_rag.eval.qgen.review import ReviewLog, load_candidates, replay, review_stats

        st = replay(load_candidates(work), ReviewLog(work / "review.jsonl").read())
        print(json.dumps(review_stats(st), ensure_ascii=False, indent=2))
    elif args.action == "freeze":
        from novel_rag.eval.qgen.freeze import FreezeError

        try:
            m = series.freeze_series(
                books, cfg, work, s.paths.datasets_dir / args.dataset, args.dataset,
                review_method=REVIEW_METHOD, allow_pending=args.allow_pending,
            )  # fmt: skip
        except FreezeError as e:
            print(f"冻结失败：{e}", file=sys.stderr)
            return 1
        print(json.dumps(m.counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
