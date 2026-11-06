"""导入原文：解析、规范化、识别卷与章节，输出每卷的 text.txt / chapters.jsonl / notes.jsonl / manifest.json。

用法：
    uv run python scripts/ingest.py data/raw/xxx.epub --series-id asoiaf             # 解析并写入全部卷
    uv run python scripts/ingest.py data/raw/xxx.epub --series-id asoiaf --dry-run   # 只出质检报告
    uv run python scripts/ingest.py data/raw/xxx.epub --series-id asoiaf --finalize 1  # 写入并定稿卷一
质检报告写到 data/reports/ingest_{series_id}_{时间}.{json,md}；有 error 时不写任何产物。
已定稿（存在 text.lock）的卷，若本次结果的 hash 不同则拒绝覆盖，除非加 --force。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from novel_rag.ingestion.ir import IngestError
from novel_rag.ingestion.pipeline import run_ingest


def _ints(s: str) -> set[int]:
    return {int(x) for x in s.split(",") if x.strip()}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    parser.add_argument("--series-id", help="系列 ID，决定 book_id = {series_id}-{卷号}；默认由文件名生成")
    parser.add_argument("--volumes", type=_ints, help="只处理这些卷，逗号分隔，例如 1,2")
    parser.add_argument("--finalize", type=_ints, default=set(), help="写入后为这些卷生成 text.lock（定稿）")
    parser.add_argument("--force", action="store_true", help="覆盖已定稿但 hash 不同的卷（金标需要重新校验）")
    parser.add_argument("--dry-run", action="store_true", help="只生成质检报告，不写产物")
    args = parser.parse_args()

    try:
        result = run_ingest(
            args.path,
            series_id=args.series_id,
            volumes=args.volumes,
            finalize=args.finalize,
            force=args.force,
            dry_run=args.dry_run,
        )
    except IngestError as e:
        print(f"导入失败：{e}", file=sys.stderr)
        return 2
    r = result.report
    print(f"质检：{'通过' if result.ok else '失败'}（error {len(r['errors'])}，warning {len(r['warnings'])}）")
    for e in r["errors"][:20]:
        print(f"  [error] {e['code']} {e['message']}")
    for v in r["volumes"]:
        t = v["text"]
        print(
            f"  {v['book_id']} {v['title']}：{v['counts']['chapters']} 章，{t['chars']} 字，"
            f"sha256 {t['sha256'][:16]}…，定稿状态 {t['lock']}"
        )
    print(f"已写入：{result.written or '无'}")
    print(f"报告：{result.report_md}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
