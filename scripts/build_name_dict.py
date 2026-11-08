"""从图谱导出 IK 扩展词典 docker/elasticsearch/ik-dict/novel_ext.dic（P3 第十节）。

用法：
    uv run python scripts/build_name_dict.py                    # 最近构建的图谱版本
    uv run python scripts/build_name_dict.py --dry-run          # 只统计不写文件
写完后：docker compose restart elasticsearch，等 ES 就绪，再用 index_es.py --analysis-version es2 新建索引。
IK 默认词典已经整体成词的写法会被排除（用当前 ES 的 ik_smart 判断，所以要在重启前运行）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

from novel_rag.config import PROJECT_ROOT, get_settings
from novel_rag.kg.build import load_export
from novel_rag.kg.common import kg_root, load_series_kg
from novel_rag.kg.name_dict import candidate_words, filter_known, render_dic

DIC = PROJECT_ROOT / "docker" / "elasticsearch" / "ik-dict" / "novel_ext.dic"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", help="图谱版本，默认取最近构建的")
    parser.add_argument("--min-mentions", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    s = get_settings()
    root = kg_root(s.paths.data_dir)
    d = (
        root / args.version
        if args.version
        else max(root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime).parent
    )
    tables, manifest = load_export(d)
    series = load_series_kg(s.paths.books_dir, list(manifest["book_text_sha256"]))
    words = candidate_words(tables["entities"], series.rules, min_mentions=args.min_mentions)

    from elasticsearch import Elasticsearch

    es = Elasticsearch(s.es.url, request_timeout=s.es.request_timeout)

    def single(batch: list[str]) -> list[bool]:
        out = []
        for w in batch:
            toks = es.indices.analyze(analyzer="ik_smart", text=w)["tokens"]
            out.append(len(toks) == 1 and toks[0]["token"] == w)
        return out

    kept = filter_known(words, single)
    text = render_dic(kept)
    report = {
        "kg_version": manifest["kg_version"],
        "candidates": len(words),
        "already_single_token": len(words) - len(kept),
        "written": len(kept),
        "dic_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "sample": list(kept)[:20],
    }
    if not args.dry_run:
        DIC.write_text(text, encoding="utf-8", newline="\n")
        (d / "name_dict.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
