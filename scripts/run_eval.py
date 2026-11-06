"""检索评测：在评测集上跑检索，按题型输出 Recall@K / Hit@K / MRR、固定 token 预算下的证据覆盖率 cov@B、
bootstrap 置信区间与分阶段延迟，结果落盘到 eval/results/。

用法：
    uv run python scripts/run_eval.py --dataset asoiaf_v0 --name hybrid_rerank   # 默认配置，只跑 dev
    uv run python scripts/run_eval.py --dataset asoiaf_v0 --name vector --retriever vector --no-rerank --expand-window 0
    uv run python scripts/run_eval.py --dataset asoiaf_v0 --name bm25_bigram --retriever bm25 --bm25-scheme ik_bigram
    uv run python scripts/run_eval.py --dataset asoiaf_v0 --name hybrid_k20 --no-rerank --rrf-k 20
    uv run python scripts/run_eval.py --dataset asoiaf_v0 --name final --split test             # 跑 test 要显式指定
    uv run python scripts/run_eval.py --dataset asoiaf_v0 --name stress --scope all             # 设置 B：全部已索引的卷
    uv run python scripts/run_eval.py --dataset asoiaf_v0 --name gen --generate                 # 顺带生成答案供抽查

所有覆盖项都会写回配置快照（run.json 的 config），实验记录与实际运行的参数一致。
检索范围（--scope）默认是评测集涉及的卷（设置 A）；`all` 表示该分块版本下已切分的全部卷（设置 B）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from novel_rag.config import get_settings
from novel_rag.corpus import BookText
from novel_rag.eval.dataset import DatasetManifest, load_dataset
from novel_rag.eval.metrics import Span
from novel_rag.eval.runner import dataset_fingerprint, git_info, run_retrieval_eval, write_run
from novel_rag.generation.simple import answer
from novel_rag.ingestion.chunking import chunk_version, chunks_dir, load_chunks
from novel_rag.llm import get_chat_model, get_token_counter
from novel_rag.retrieval.expand import ChunkIndex, expand_hits
from novel_rag.retrieval.factory import build_retriever
from novel_rag.retrieval.filters import SearchFilter


def _books_with_chunks(books_dir: Path, version: str) -> list[str]:
    return sorted(p.parent.parent.name for p in books_dir.glob(f"*/chunks/{version}/chunks.jsonl"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-dir", type=Path, help="评测集目录，默认 eval/datasets/{dataset}（草稿用）")
    parser.add_argument("--name", required=True, help="实验名称，用于结果目录名")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--chunk-version", help="默认取当前 config 对应的分块版本")
    parser.add_argument("--template", choices=["raw", "title_prefix"])
    parser.add_argument("--retriever", choices=["vector", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--routes", help="hybrid 的检索路，逗号分隔（覆盖 retrieval.routes）")
    parser.add_argument("--bm25-scheme", choices=["ik", "ik_bigram"])
    parser.add_argument("--rrf-k", type=int)
    parser.add_argument("--fusion-top-k", type=int)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rerank-candidates", type=int)
    parser.add_argument("--expand-window", type=int, help="small-to-big 窗口（0 = 不扩展）")
    parser.add_argument("--scope", help="检索范围：逗号分隔的 book_id，或 all；默认为评测集涉及的卷")
    parser.add_argument("--spoiler-filter", action="store_true", help="用每题的 max_reading_order 过滤检索范围")
    parser.add_argument("--generate", action="store_true", help="对每题调用生成模型作答（消耗 LLM 调用，有缓存）")
    parser.add_argument("--limit", type=int, help="只跑前 N 题（调试用）")
    args = parser.parse_args()

    s = get_settings()
    overrides = {
        k: v
        for k, v in {
            "routes": tuple(x.strip() for x in args.routes.split(",")) if args.routes else None,
            "bm25_scheme": args.bm25_scheme,
            "rrf_k": args.rrf_k,
            "fusion_top_k": args.fusion_top_k,
            "rerank": args.rerank,
            "rerank_candidates": args.rerank_candidates,
            "expand_window": args.expand_window,
            "embed_text_template": args.template,
        }.items()
        if v is not None
    }
    s = s.model_copy(update={"retrieval": s.retrieval.model_copy(update=overrides)})
    rc = s.retrieval

    ds_dir = args.dataset_dir or s.paths.datasets_dir / args.dataset
    items = load_dataset(ds_dir, s.paths.books_dir, split=None if args.split == "all" else args.split)
    if args.limit:
        items = items[: args.limit]
    manifest = DatasetManifest.model_validate_json((ds_dir / "manifest.json").read_text(encoding="utf-8"))
    version = args.chunk_version or chunk_version(s.chunk)
    if args.scope == "all":
        scope = tuple(_books_with_chunks(s.paths.books_dir, version))
    elif args.scope:
        scope = tuple(x.strip() for x in args.scope.split(",") if x.strip())
    else:
        scope = tuple(manifest.books)
    missing = set(manifest.books) - set(scope)
    if missing:
        print(f"检索范围没有包含评测集涉及的卷：{sorted(missing)}", file=sys.stderr)
        return 1

    spans: dict[str, Span] = {}
    titles: dict[str, str] = {}
    all_chunks = []
    for b in scope:
        chunks, cm = load_chunks(chunks_dir(s.paths.books_dir, b, version))
        book = BookText.load(s.paths.books_dir / b)
        expected = manifest.books.get(b, book.text_sha256)
        if cm["text_sha256"] != expected:
            print(f"{b} 的 chunks 与评测集标注时（或当前）的文本不一致", file=sys.stderr)
            return 1
        spans.update({c["chunk_id"]: Span(c["book_id"], c["char_start"], c["char_end"]) for c in chunks})
        titles[b] = book.title
        all_chunks.extend(chunks)
    index = ChunkIndex(all_chunks)

    retriever = build_retriever(s, scope, args.retriever, version=version, template=rc.embed_text_template)
    counter = get_token_counter()

    @lru_cache(maxsize=100_000)
    def n_tokens(text: str) -> int:
        return counter.count([text])[0]

    def expander(hits, flt: SearchFilter):
        return expand_hits(hits, index, rc.expand_window, flt)

    generator = None
    if args.generate:
        chat = get_chat_model("generation")
        k = s.eval.generate_top_k
        generator = lambda item, hits: answer(item.question, hits[:k], chat, book_titles=titles)  # noqa: E731

    rows, metrics = run_retrieval_eval(
        items,
        retriever,
        spans,
        s,
        base_filter=SearchFilter(book_ids=scope),
        spoiler_filter=args.spoiler_filter,
        generator=generator,
        expander=expander,
        count_tokens=lambda c: n_tokens(c.text),
    )
    run = {
        "name": args.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "dataset": {
            "name": manifest.name,
            "split": args.split,
            "n_items": len(items),
            "fingerprint": dataset_fingerprint(ds_dir),
            "books": manifest.books,
        },
        "chunk_version": version,
        "embed_text_template": rc.embed_text_template,
        "retriever_kind": args.retriever,
        "retriever": retriever.describe(),
        "scope": list(scope),
        "expand_window": rc.expand_window,
        "token_counter": counter.model_name,
        "spoiler_filter": args.spoiler_filter,
        "relevance": {"min_ratio": s.eval.relevance_min_ratio, "min_chars": s.eval.relevance_min_chars},
        "ks": list(s.eval.ks),
        "generate": args.generate,
        "overrides": overrides,
        "config": s.snapshot(),
    }
    out = write_run(s.paths.results_dir, args.name, run, metrics, rows, s.eval.ks)
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"结果目录：{out}")
    if args.generate:
        print(json.dumps({"generated": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
