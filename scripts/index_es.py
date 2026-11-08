"""把若干卷的 chunk 导入 Elasticsearch（幂等：重复运行结果不变；只删除所导入各卷中多余的文档）。

用法：
    uv run python scripts/index_es.py --books asoiaf-1
    uv run python scripts/index_es.py --books asoiaf-1,asoiaf-2,asoiaf-3,asoiaf-4,asoiaf-5
    uv run python scripts/index_es.py --books asoiaf-1 --chunk-version 0c94b7e4
    uv run python scripts/index_es.py --books asoiaf-1,...,asoiaf-5 --analysis-version es2 --kg-version 1a2b3c4d
需要 ES 容器已启动（docker compose up -d），以及先运行 scripts/chunk.py。
索引名 chunks_{chunk 版本}_{分析方案版本}；正文同时建 IK 字段与 bigram 子字段，查询时选择用哪些字段。
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from novel_rag.config import get_settings
from novel_rag.corpus import BookText
from novel_rag.ingestion.chunking import chunk_version, chunks_dir, load_chunks
from novel_rag.retrieval.es_store import ESChunkStore, index_name


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--books", required=True, help="逗号分隔的 book_id")
    parser.add_argument("--chunk-version", help="默认取当前 config 对应的分块版本")
    parser.add_argument("--analysis-version", help="默认取 retrieval.es_analysis_version；回灌人名词典后用 es2")
    parser.add_argument("--kg-version", help="给 chunk 加 entity_ids 字段（图谱提及记录；es2 起可用）")
    args = parser.parse_args()

    s = get_settings()
    version = args.chunk_version or chunk_version(s.chunk)
    chunks = []
    for b in [x.strip() for x in args.books.split(",") if x.strip()]:
        book = BookText.load(s.paths.books_dir / b)
        cs, manifest = load_chunks(chunks_dir(s.paths.books_dir, b, version))
        if manifest["text_sha256"] != book.text_sha256:
            print(f"{b} 的 chunks 与当前 text.txt 不一致，请重新运行 scripts/chunk.py", file=sys.stderr)
            return 1
        chunks.extend(cs)

    analysis = args.analysis_version or s.retrieval.es_analysis_version
    if args.kg_version:
        if analysis == "es1":
            print("es1 的 mapping 没有 entity_ids 字段", file=sys.stderr)
            return 1
        from novel_rag.kg.build import load_export
        from novel_rag.kg.common import kg_root

        tables, _ = load_export(kg_root(s.paths.data_dir) / args.kg_version)
        ents: dict[str, set[str]] = {}
        for m in tables["mentions"]:
            ents.setdefault(m["chunk_id"], set()).add(m["entity_id"])
        for c in chunks:
            c["entity_ids"] = sorted(ents.get(c["chunk_id"], ()))
    t0 = time.perf_counter()
    name = index_name(version, analysis)
    store = ESChunkStore.connect(s.es, name, s.retrieval, analysis)
    desc = {"chunk_version": version, "analysis_version": analysis, "kg_version": args.kg_version}
    store.ensure_index(description=json.dumps(desc, ensure_ascii=False))
    result = store.sync(chunks)
    print(
        json.dumps(
            {"index": name, **result, "books": args.books, "seconds": round(time.perf_counter() - t0, 1)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
