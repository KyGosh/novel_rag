"""检索评测：在评测集上跑向量检索，按题型输出 Recall@K / Hit@K / MRR，结果落盘到 eval/results/。

用法：
    uv run python scripts/run_eval.py --dataset asoiaf_v0 --name baseline_vector               # 默认只跑 dev
    uv run python scripts/run_eval.py --dataset asoiaf_v0 --name baseline_vector --split test  # 跑 test 要显式指定
    uv run python scripts/run_eval.py --dataset asoiaf_v0 --name vec_tp --template title_prefix
    uv run python scripts/run_eval.py --dataset asoiaf_v0 --name baseline_gen --generate       # 顺带生成答案供抽查
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from novel_rag.config import get_settings
from novel_rag.corpus import BookText
from novel_rag.eval.dataset import DatasetManifest, load_dataset
from novel_rag.eval.metrics import Span
from novel_rag.eval.runner import dataset_fingerprint, git_info, run_retrieval_eval, write_run
from novel_rag.generation.simple import answer
from novel_rag.ingestion.chunking import chunk_version, chunks_dir, load_chunks
from novel_rag.llm import get_chat_model
from novel_rag.retrieval.factory import build_vector_retriever


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-dir", type=Path, help="评测集目录，默认 eval/datasets/{dataset}（草稿用）")
    parser.add_argument("--name", required=True, help="实验名称，用于结果目录名")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--chunk-version", help="默认取当前 config 对应的分块版本")
    parser.add_argument("--template", choices=["raw", "title_prefix"])
    parser.add_argument("--spoiler-filter", action="store_true", help="用每题的 max_reading_order 过滤检索范围")
    parser.add_argument("--generate", action="store_true", help="对每题调用生成模型作答（消耗 LLM 调用，有缓存）")
    parser.add_argument("--limit", type=int, help="只跑前 N 题（调试用）")
    args = parser.parse_args()

    s = get_settings()
    ds_dir = args.dataset_dir or s.paths.datasets_dir / args.dataset
    items = load_dataset(ds_dir, s.paths.books_dir, split=None if args.split == "all" else args.split)
    if args.limit:
        items = items[: args.limit]
    manifest = DatasetManifest.model_validate_json((ds_dir / "manifest.json").read_text(encoding="utf-8"))
    book_ids = tuple(manifest.books)
    version = args.chunk_version or chunk_version(s.chunk)
    template = args.template or s.retrieval.embed_text_template

    spans: dict[str, Span] = {}
    titles: dict[str, str] = {}
    for b in book_ids:
        chunks, cm = load_chunks(chunks_dir(s.paths.books_dir, b, version))
        if cm["text_sha256"] != manifest.books[b]:
            print(f"{b} 的 chunks 与评测集标注时的文本不一致", file=sys.stderr)
            return 1
        spans.update({c["chunk_id"]: Span(c["book_id"], c["char_start"], c["char_end"]) for c in chunks})
        titles[b] = BookText.load(s.paths.books_dir / b).title

    retriever = build_vector_retriever(s, book_ids, version=version, template=template)
    generator = None
    if args.generate:
        chat = get_chat_model("generation")
        k = s.eval.generate_top_k
        generator = lambda item, hits: answer(item.question, hits[:k], chat, book_titles=titles)  # noqa: E731

    rows, metrics = run_retrieval_eval(
        items, retriever, spans, s, spoiler_filter=args.spoiler_filter, generator=generator
    )
    run = {
        "name": args.name,
        "created": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "dataset": {
            "name": manifest.name,
            "split": args.split,
            "n_items": len(items),
            "fingerprint": dataset_fingerprint(ds_dir),
            "books": manifest.books,
        },
        "chunk_version": version,
        "embed_text_template": template,
        "retriever": retriever.describe(),
        "spoiler_filter": args.spoiler_filter,
        "relevance": {"min_ratio": s.eval.relevance_min_ratio, "min_chars": s.eval.relevance_min_chars},
        "ks": list(s.eval.ks),
        "generate": args.generate,
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
