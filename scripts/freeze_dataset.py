"""冻结评测集：从校验通过的候选题中选题、划分 dev / test、校验（区间、text hash、版权），写入 eval/datasets/{name}/。

用法：
    uv run python scripts/freeze_dataset.py --book asoiaf-1 --dataset asoiaf_v0
    uv run python scripts/freeze_dataset.py --book asoiaf-1 --dataset asoiaf_v0 --allow-pending   # 允许有未校验的题
    uv run python scripts/freeze_dataset.py --book asoiaf-1 --dataset asoiaf_v0 --draft   # 草稿：未校验题也收录
草稿写到 data/eval_work/{dataset}/draft/（不入库），只用来在人工校验完成前跑通评测链路。
输出 items.jsonl（只含问题、简短答案和字符区间）、manifest.json、reserve.jsonl（未入选的通过题）。
"""

from __future__ import annotations

import argparse
import json
import sys

from novel_rag.config import get_settings
from novel_rag.corpus import BookText
from novel_rag.eval.qgen.freeze import FreezeError, freeze


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book", required=True)
    parser.add_argument("--dataset", required=True, help="评测集名称，同时是工作目录和输出目录的名字")
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--draft", action="store_true", help="未校验的题也收录，输出到工作目录下的 draft/")
    parser.add_argument("--review-method", help="校验方式说明，写入 manifest（例如谁预审、谁确认）")
    args = parser.parse_args()

    s = get_settings()
    book = BookText.load(s.paths.books_dir / args.book)
    work = s.paths.eval_work_dir / args.dataset
    out = work / "draft" if args.draft else s.paths.datasets_dir / args.dataset
    name = f"{args.dataset}_draft" if args.draft else args.dataset
    try:
        manifest = freeze(
            book,
            s.evalgen,
            work,
            out,
            name,
            allow_pending=args.allow_pending,
            draft=args.draft,
            review_method=args.review_method,
        )
    except FreezeError as e:
        print(f"冻结失败：{e}", file=sys.stderr)
        return 1
    print(json.dumps(manifest.counts, ensure_ascii=False, indent=2))
    print(f"已写入：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
