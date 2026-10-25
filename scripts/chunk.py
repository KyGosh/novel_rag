"""分块：在已定稿的卷上切句、分块，统计字数与 token 分布，并检查 token 预算。

用法：
    uv run python scripts/chunk.py --book asoiaf-1
    uv run python scripts/chunk.py --book asoiaf-1 --target 300 --max 500   # 消融用的其他参数
产物写到 data/books/{book_id}/chunks/{chunk_version}/（chunks.jsonl + manifest.json）。
"""

from __future__ import annotations

import argparse
import json
import sys

from novel_rag.config import get_settings
from novel_rag.corpus import BookText
from novel_rag.ingestion.chunking import build_chunks, chunks_dir, write_chunks
from novel_rag.llm import get_token_counter


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book", required=True)
    parser.add_argument("--target", type=int, help="覆盖 chunk.target_chars")
    parser.add_argument("--max", type=int, help="覆盖 chunk.max_chars")
    parser.add_argument("--min", type=int, help="覆盖 chunk.min_chars")
    parser.add_argument("--overlap", type=int, help="覆盖 chunk.overlap_chars")
    args = parser.parse_args()

    s = get_settings()
    overrides = {
        k: v
        for k, v in {
            "target_chars": args.target,
            "max_chars": args.max,
            "min_chars": args.min,
            "overlap_chars": args.overlap,
        }.items()
        if v is not None
    }
    cfg = s.chunk.model_copy(update=overrides)
    book = BookText.load(s.paths.books_dir / args.book)
    chunks, manifest = build_chunks(book, cfg, token_counter=get_token_counter(), token_limit=s.reranker.max_length)
    out = chunks_dir(s.paths.books_dir, book.book_id, manifest["chunk_version"])
    write_chunks(out, chunks, manifest)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("chunk_version", "n_chunks", "chars", "tokens", "token_budget", "by_section_type")
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"已写入：{out}")
    return 0 if manifest["token_budget"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
