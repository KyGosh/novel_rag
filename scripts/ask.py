"""命令行问答：固定流程 RAG（config 中的检索配置，默认混合检索 + rerank）+ 最简生成，便于手动试用。
原文只在本地终端显示。

用法：
    uv run python scripts/ask.py --book asoiaf-1 "问题"
    uv run python scripts/ask.py --book asoiaf-1 --max-reading-order 20 "问题"   # 只看前 20 章（防剧透）
    uv run python scripts/ask.py --book asoiaf-1 --no-generate "问题"            # 只看检索结果
    uv run python scripts/ask.py --book asoiaf-1 --retriever vector "问题"       # 只用向量检索
"""

from __future__ import annotations

import argparse
import sys

from novel_rag.config import get_settings
from novel_rag.corpus import BookText
from novel_rag.generation.simple import answer
from novel_rag.llm import get_chat_model
from novel_rag.retrieval.factory import build_retriever
from novel_rag.retrieval.filters import SearchFilter


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--book", required=True, action="append", help="可重复指定多卷")
    parser.add_argument("--top-k", type=int, help="默认取 retrieval.rerank_top_k")
    parser.add_argument("--retriever", choices=["vector", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--max-reading-order", type=int)
    parser.add_argument("--no-generate", action="store_true")
    args = parser.parse_args()

    s = get_settings()
    book_ids = tuple(args.book)
    retriever = build_retriever(s, book_ids, args.retriever)
    hits = retriever.search(
        args.question,
        args.top_k or s.retrieval.rerank_top_k,
        SearchFilter(book_ids=book_ids, max_reading_order=args.max_reading_order),
    )
    for h in hits:
        f = h.fields
        preview = f.get("text", "").replace("\n", " ")[:60]
        print(
            f"[{h.rank}] {h.score:.4f} {h.chunk_id} 第{f.get('chapter_idx')}章「{f.get('chapter_title')}」 {preview}…"
        )
    print(f"检索耗时：{retriever.last_latency_ms}")
    if args.no_generate:
        return 0
    titles = {b: BookText.load(s.paths.books_dir / b).title for b in book_ids}
    res = answer(args.question, hits, get_chat_model("generation"), book_titles=titles)
    print("\n" + res["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
