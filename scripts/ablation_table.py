"""从 eval/results/ 汇总生成消融表（Markdown + JSON），数字全部来自实验记录。

用法：
    uv run python scripts/ablation_table.py --spec eval/ablations/v1.json
输出到 eval/results/ablation_{规格名}.md / .json（覆盖旧版）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from novel_rag.config import get_settings
from novel_rag.eval.ablation import build_table, render_markdown


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()

    s = get_settings()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    e = s.eval
    cfg = {"bootstrap_samples": e.bootstrap_samples, "bootstrap_seed": e.bootstrap_seed, "ci_level": e.ci_level}
    table = build_table(spec, s.paths.results_dir, cfg)
    out = s.paths.results_dir / f"ablation_{args.spec.stem}"
    out.with_suffix(".json").write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = render_markdown(table)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"已写入：{out}.md / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
