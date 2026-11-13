"""Agent 消融表（P5）：从 eval/results/ 的问答运行结果汇总生成，数字全部来自实验记录。

用法：
    uv run python scripts/qa_ablation.py --spec eval/ablations/qa_v1.json
    uv run python scripts/qa_ablation.py --spec eval/ablations/qa_v1.json --allow-missing   # 运行还没跑完时先看已有的行
输出到 eval/results/ablation_{规格名}.md / .json（覆盖旧版）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from novel_rag.config import get_settings
from novel_rag.eval.ablation import find_run
from novel_rag.qa.ablation import build_table, render_markdown


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true", help="跳过还没有全部跑完的变体")
    args = parser.parse_args()

    s = get_settings()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if args.allow_missing:
        kept = []
        for row in spec["rows"]:
            try:
                for n in row["runs"]:
                    find_run(s.paths.results_dir, n)
                kept.append(row)
            except FileNotFoundError:
                print(f"跳过（未跑完）：{row['label']}", file=sys.stderr)
        spec["rows"] = kept
    e = s.eval
    cfg = {"bootstrap_samples": e.bootstrap_samples, "bootstrap_seed": e.bootstrap_seed, "ci_level": e.ci_level}
    table = build_table(spec, s.paths.results_dir, cfg)
    out = s.paths.results_dir / f"ablation_{args.spec.stem}"
    out.with_suffix(".json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    md = render_markdown(table)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"已写入：{out}.md / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
