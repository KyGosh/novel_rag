"""章节摘要索引：把 data/summaries/{摘要版本}/summaries.jsonl 写进 ES 与 Milvus（删掉重建，幂等）。

用法：
    uv run python scripts/index_summaries.py                    # 最近生成的摘要版本
    uv run python scripts/index_summaries.py --version d8139169
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from novel_rag.config import get_settings
from novel_rag.ingestion.summary import load_summaries
from novel_rag.llm import get_embedder
from novel_rag.retrieval.embed_cache import EmbeddingCache, open_embedding_store
from novel_rag.retrieval.summary_store import (
    ESSummaryStore,
    MilvusSummaryStore,
    es_index_name,
    milvus_collection_name,
    summary_text,
)


def latest_summary_version(data_dir) -> str:
    root = data_dir / "summaries"
    cands = sorted(root.glob("*/summaries.jsonl"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError("没有摘要产物，先运行 scripts/summarize.py")
    return cands[-1].parent.name


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", help="摘要版本，默认取最近生成的")
    parser.add_argument("--analysis-version", help="ES 分析方案版本，默认取 retrieval.es_analysis_version")
    args = parser.parse_args()

    s = get_settings()
    version = args.version or latest_summary_version(s.paths.data_dir)
    records = load_summaries(s.paths.data_dir / "summaries" / version)
    desc = json.dumps({"summary_version": version}, ensure_ascii=False)
    t0 = time.perf_counter()
    es = ESSummaryStore.connect(s.es, es_index_name(version, args.analysis_version or s.retrieval.es_analysis_version))
    n_es = es.rebuild(records, description=desc)
    embedder = get_embedder()
    cache = EmbeddingCache(embedder, open_embedding_store(s.paths.embedding_cache_dir))
    vectors = cache.embed_documents([summary_text(r) for r in records])
    mv = MilvusSummaryStore.connect(
        s.milvus, milvus_collection_name(embedder.model_name, embedder.dim, version), embedder.dim, s.retrieval
    )
    n_mv = mv.rebuild(records, vectors, description=desc)
    print(
        json.dumps(
            {
                "summary_version": version,
                "es_index": es.name,
                "es_count": n_es,
                "milvus_collection": mv.name,
                "milvus_count": n_mv,
                "records": len(records),
                "seconds": round(time.perf_counter() - t0, 1),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if n_es == n_mv == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
