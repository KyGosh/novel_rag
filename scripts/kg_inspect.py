"""把抽取结果按 chunk 打印成便于人工核对的文本（实体、关系与证据引文、别名、丢弃计数）。含原文片段，只在本地看。

用法：
    uv run python scripts/kg_inspect.py --tag pilot --chunks asoiaf-1:0003:0002,asoiaf-1:0073:0010
    uv run python scripts/kg_inspect.py --tag pilot --sample 5 --seed 1 --with-text
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from novel_rag.config import get_settings
from novel_rag.corpus import BookText
from novel_rag.kg.common import kg_root
from novel_rag.kg.schema import KIN_ROLES, REL_BY_NAME


def render(rec: dict, book: BookText, with_text: bool) -> str:
    lines = [f"=== {rec['chunk_id']}（reading_order {rec['reading_order']}，{rec['section_type']}）"]
    if with_text:
        lines.append(rec.get("_text", ""))
    lines.append("-- 实体")
    for m in rec["mentions"]:
        extra = f" 全名={m['full_name']}" if m["full_name"] else ""
        extra += f" 声称全名={m['claimed_full_name']}" if m.get("claimed_full_name") else ""
        extra += f" 群体={m['group']}" if m["group"] else ""
        extra += f" 类别={m['category']}" if m.get("category") else ""
        extra += " [补]" if m.get("implicit") else ""
        lines.append(f"  [{m['m']}] {m['type']} {m['name']}{extra} —— {m['description']}")
    lines.append("-- 关系")
    for r in rec["relations"]:
        s, o = rec["mentions"][r["s"]]["name"], rec["mentions"][r["o"]]["name"]
        label = REL_BY_NAME[r["type"]].label + (f"/{KIN_ROLES[r['role']]}" if r["role"] else "")
        quote = book.text[r["evidence"][0] : r["evidence"][1]]
        lines.append(f"  {s} -[{label}]-> {o}    「{quote}」({r['match']})")
    if rec["aliases"]:
        lines.append("-- 别名")
        for a in rec["aliases"]:
            quote = book.text[a["evidence"][0] : a["evidence"][1]]
            lines.append(f"  {rec['mentions'][a['m']]['name']} = {a['alias']}    「{quote}」")
    lines.append(f"-- 丢弃 {rec['drops']}")
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--version", help="抽取版本目录，默认取最近修改的")
    parser.add_argument("--chunks", help="逗号分隔的 chunk_id")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-text", action="store_true")
    args = parser.parse_args()

    s = get_settings()
    root = kg_root(s.paths.data_dir) / "extract"
    if args.version:
        path = root / args.version / f"{args.tag}.jsonl"
    else:
        path = max(root.glob(f"*/{args.tag}.jsonl"), key=lambda p: p.stat().st_mtime)
    recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    if args.chunks:
        want = set(args.chunks.split(","))
        recs = [r for r in recs if r["chunk_id"] in want]
    elif args.sample:
        recs = random.Random(args.seed).sample(recs, min(args.sample, len(recs)))
    books: dict[str, BookText] = {}
    for r in recs:
        if r["book_id"] not in books:
            books[r["book_id"]] = BookText.load(s.paths.books_dir / r["book_id"])
        if args.with_text:
            from novel_rag.ingestion.chunking import chunk_version, chunks_dir, load_chunks

            cs, _ = load_chunks(chunks_dir(s.paths.books_dir, r["book_id"], chunk_version(s.chunk)))
            r["_text"] = next(c["text"] for c in cs if c["chunk_id"] == r["chunk_id"])
        print(render(r, books[r["book_id"]], args.with_text))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
