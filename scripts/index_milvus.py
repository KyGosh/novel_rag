"""把一卷的 chunk 编码后导入 Milvus（幂等：重复运行结果不变；向量按文本 hash 缓存）。

用法：
    uv run python scripts/index_milvus.py --book asoiaf-1
    uv run python scripts/index_milvus.py --book asoiaf-1 --template title_prefix      # 编码文本加卷名与章节标题
    uv run python scripts/index_milvus.py --book asoiaf-1 --chunk-version 0c94b7e4     # 指定已生成的分块版本
需要 Milvus 容器已启动（docker compose up -d），以及先运行 scripts/chunk.py。
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from novel_rag.config import get_settings
from novel_rag.corpus import BookText
from novel_rag.ingestion.chunking import chunk_version, chunks_dir, load_chunks
from novel_rag.llm import get_embedder
from novel_rag.retrieval.embed_cache import EmbeddingCache, encode_text, open_embedding_store
from novel_rag.retrieval.milvus_store import MilvusChunkStore, collection_name


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book", required=True)
    parser.add_argument("--chunk-version", help="默认取当前 config 对应的分块版本")
    parser.add_argument("--template", choices=["raw", "title_prefix"], help="覆盖 retrieval.embed_text_template")
    args = parser.parse_args()

    s = get_settings()
    version = args.chunk_version or chunk_version(s.chunk)
    template = args.template or s.retrieval.embed_text_template
    book = BookText.load(s.paths.books_dir / args.book)
    chunks, manifest = load_chunks(chunks_dir(s.paths.books_dir, book.book_id, version))
    if manifest["text_sha256"] != book.text_sha256:
        print("chunks 与当前 text.txt 不一致，请重新运行 scripts/chunk.py", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    embedder = get_embedder()
    cache = EmbeddingCache(embedder, open_embedding_store(s.paths.embedding_cache_dir))
    vectors = cache.embed_documents([encode_text(c, template, book.title) for c in chunks])
    t1 = time.perf_counter()

    name = collection_name(embedder.model_name, embedder.dim, version, template)
    store = MilvusChunkStore.connect(s.milvus, name, embedder.dim, s.retrieval)
    desc = json.dumps(
        {"chunk_version": version, "template": template, "model": embedder.model_name}, ensure_ascii=False
    )
    store.ensure_collection(description=desc)
    result = store.sync(chunks, vectors)
    t2 = time.perf_counter()
    print(
        json.dumps(
            {
                "collection": name,
                **result,
                "embed_cache_hits": cache.hits,
                "embed_cache_misses": cache.misses,
                "embed_seconds": round(t1 - t0, 1),
                "milvus_seconds": round(t2 - t1, 1),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["count"] == len(chunks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
