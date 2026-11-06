"""设置 B（检索范围扩大到更多卷）的假漏检抽查：列出排在金标证据之前的"范围外卷"命中，供人工判断。

金标证据只标在评测集涉及的卷里。范围扩大后，别的卷里也可能有能回答问题的段落（例如后文回顾前情），
这些命中按标注算"不相关"，会把指标压低。本脚本抽出最可疑的命中：设置 B 比设置 A 掉分的题，
排在第一条金标证据之前（或金标没进前 K 时排在前 K）的范围外命中，按名次排序取前 n 条。

输出含原文片段，只写到 data/reports/（不入库）。人工在每条后面填"能 / 不能"回答，统计假漏检率。

用法：
    uv run python scripts/spotcheck_scope.py --run-a p2a_rr40 --run-b p2b_rr40 --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from novel_rag.config import get_settings
from novel_rag.eval.ablation import find_run, load_run
from novel_rag.eval.dataset import load_dataset
from novel_rag.ingestion.chunking import chunks_dir, load_chunks

EXCERPT = 160


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-a", required=True, help="设置 A 的实验名（基准）")
    p.add_argument("--run-b", required=True, help="设置 B 的实验名（范围更大）")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--top", type=int, default=5, help="金标没进前 K 时，取前 K 名里的范围外命中")
    p.add_argument("--metric", default="recall@20", help="判断设置 B 是否掉分的指标")
    args = p.parse_args()
    s = get_settings()
    results = Path("eval/results")
    a, b = load_run(find_run(results, args.run_a)), load_run(find_run(results, args.run_b))
    scope_a = set(a["run"]["scope"])
    ds_dir = Path("eval/datasets") / b["run"]["dataset"]["name"]
    items = {i.id: i for i in load_dataset(ds_dir, s.paths.books_dir)}
    ma = {r["id"]: r["metrics"] for r in a["rows"]}

    picks = []
    for row in b["rows"]:
        drop = ma[row["id"]].get(args.metric, 0) - row["metrics"].get(args.metric, 0)
        ranks = [r for r in row["evidence_ranks"] if r]
        first = min(ranks) if ranks else args.top + 1
        for h in row["hits"]:
            if h["rank"] >= first or h["chunk_id"].split(":")[0] in scope_a:
                continue
            picks.append((-drop, h["rank"], row["id"], h["chunk_id"]))
    picks.sort()
    picks = picks[: args.n]

    version = b["run"]["chunk_version"]
    need = {cid.split(":")[0] for *_, cid in picks}
    texts = {}
    for book in need:
        chunks, _ = load_chunks(chunks_dir(s.paths.books_dir, book, version))
        texts.update({c["chunk_id"]: c for c in chunks})

    out = [
        f"# 设置 B 假漏检抽查：{args.run_b} 对照 {args.run_a}",
        "",
        f"共 {len(picks)} 条；排序：设置 B 掉分多的题优先，同题内按名次。每条判断该命中能否回答问题。",
        "",
    ]
    for n, (neg_drop, rank, qid, cid) in enumerate(picks, 1):
        it, c = items[qid], texts[cid]
        out += [
            f"## {n}. {qid}（{it.type}）· 第 {rank} 名 · {cid} · {args.metric} 掉 {-neg_drop:.2f}",
            "",
            f"- 问：{it.question}",
            f"- 答：{it.answer}",
            f"- 命中（{c['chapter_title']}）：{c['text'][:EXCERPT].replace(chr(10), ' ')}…",
            "- 判断：",
            "",
        ]
    dest = s.paths.data_dir / "reports" / f"spotcheck_scope_{datetime.now():%Y%m%d-%H%M%S}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out), encoding="utf-8")
    print(json.dumps({"report": str(dest), "n": len(picks)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
