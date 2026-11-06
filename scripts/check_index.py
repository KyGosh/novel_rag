"""ES 与 Milvus 的 chunk 一致性检查：以 chunks.jsonl 为准，比对两个库中这些卷的 chunk 数量、ID 与字段。

用法：
    uv run python scripts/check_index.py --books asoiaf-1
    uv run python scripts/check_index.py --books asoiaf-1,asoiaf-2,asoiaf-3,asoiaf-4,asoiaf-5 --template raw
不一致时退出码为 1，并列出缺失、多余、重复和字段不一致的 chunk_id（不输出原文）。
"""

from __future__ import annotations

import argparse
import json
import sys

from novel_rag.config import get_settings
from novel_rag.ingestion.chunking import chunk_version, chunks_dir, load_chunks
from novel_rag.retrieval.consistency import check_consistency
from novel_rag.retrieval.es_store import ESChunkStore, index_name
from novel_rag.retrieval.milvus_store import MilvusChunkStore, collection_name


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--books", required=True, help="逗号分隔的 book_id")
    parser.add_argument("--chunk-version", help="默认取当前 config 对应的分块版本")
    parser.add_argument("--template", choices=["raw", "title_prefix"], help="Milvus collection 的编码模板")
    args = parser.parse_args()

    s = get_settings()
    version = args.chunk_version or chunk_version(s.chunk)
    books = [x.strip() for x in args.books.split(",") if x.strip()]
    chunks = []
    for b in books:
        chunks.extend(load_chunks(chunks_dir(s.paths.books_dir, b, version))[0])
    es = ESChunkStore.connect(s.es, index_name(version), s.retrieval)
    name = collection_name(
        s.embedding.model_name, s.embedding.dim, version, args.template or s.retrieval.embed_text_template
    )
    mv = MilvusChunkStore.connect(s.milvus, name, s.embedding.dim, s.retrieval)
    report = check_consistency(chunks, es.iter_docs(books), mv.iter_rows(books))
    report.update(es_index=es.name, milvus_collection=mv.name, books=books, chunk_version=version)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
