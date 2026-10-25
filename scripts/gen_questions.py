"""评测集出题：导出出题任务 / 导入候选题与人工题（定位证据 + 预筛）。

出题者按 src/novel_rag/prompts/qgen_guidelines 写候选题。

用法：
    uv run python scripts/gen_questions.py export --round 1 --book asoiaf-1 --dataset asoiaf_v0
    uv run python scripts/gen_questions.py import --round 1 --book asoiaf-1 --dataset asoiaf_v0
    uv run python scripts/gen_questions.py export --round 2 ...   # 用第一轮记录的人物做共现配对
    uv run python scripts/gen_questions.py import-human --book asoiaf-1 --dataset asoiaf_v0
人工题写在 data/eval_work/{dataset}/human.jsonl，每行一题：
    {"type": "fact", "question": "...", "answer": "...", "evidence": [{"quote": "原文片段", "chapter_idx": 12}]}
chapter_idx 可省略，此时原文片段必须在全书中只出现一次。
中间产物在 data/eval_work/{dataset}/（含原文，不入库）。只允许在已定稿（有 text.lock）的卷上出题。
"""

from __future__ import annotations

import argparse
import json
import sys

from novel_rag.config import get_settings
from novel_rag.corpus import BookText
from novel_rag.eval.qgen.tasks import export_round1, export_round2, import_human, import_round


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["export", "import", "import-human"])
    parser.add_argument("--round", type=int, choices=[1, 2], help="export / import 时必填")
    parser.add_argument("--book", required=True, help="book_id，例如 asoiaf-1")
    parser.add_argument("--dataset", required=True, help="评测集名称，例如 asoiaf_v0")
    args = parser.parse_args()

    s = get_settings()
    book = BookText.load(s.paths.books_dir / args.book)
    work = s.paths.eval_work_dir / args.dataset
    if args.action != "import-human" and args.round is None:
        parser.error("export / import 需要 --round")
    if args.action == "export":
        out = (export_round1 if args.round == 1 else export_round2)(book, s.evalgen, work)
        n = sum(1 for _ in out.open(encoding="utf-8"))
        print(f"已导出 {n} 个出题任务：{out}")
        return 0
    if args.action == "import-human":
        stats = import_human(book, s.evalgen, work)
    else:
        stats = import_round(book, s.evalgen, work, args.round)
    print(json.dumps({k: v for k, v in stats.items() if k != "issues"}, ensure_ascii=False, indent=2))
    for issue in stats.get("issues", []):
        print(f"[issue] {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
