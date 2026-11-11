"""small-to-big 扩展与上下文装箱：检索用小块，送进 LLM 前把命中块扩展到同章相邻块（seq ± window），合并重叠。

扩展放在 rerank 之后：rerank 看的是小块本身，扩展只决定"送进上下文的范围"。
扩展会让上下文变长，直接比 Recall@K 不公平，所以配套"固定 token 预算下的证据覆盖率"：按排名依次装入上下文块，
放不下的块跳过、继续尝试后面的（贪心装箱），装满预算为止。

防剧透：邻块与命中块同章，reading_order 相同；仍然用过滤条件逐块复核，不满足的邻块不扩展进来。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from novel_rag.retrieval.base import Hit
from novel_rag.retrieval.filters import SearchFilter

ChunkKey = tuple[str, int, int]  # (book_id, chapter_idx, seq)


class ChunkIndex:
    """按 (book_id, chapter_idx, seq) 查 chunk（来自 chunks.jsonl）。"""

    def __init__(self, chunks: Iterable[dict[str, Any]]):
        self._by_key: dict[ChunkKey, dict[str, Any]] = {}
        self._by_id: dict[str, dict[str, Any]] = {}
        for c in chunks:
            self._by_key[(c["book_id"], c["chapter_idx"], c["seq"])] = c
            self._by_id[c["chunk_id"]] = c

    def __len__(self) -> int:
        return len(self._by_id)

    @property
    def chunks(self) -> list[dict[str, Any]]:
        return list(self._by_id.values())

    def by_id(self, chunk_id: str) -> dict[str, Any]:
        return self._by_id[chunk_id]

    def get(self, book_id: str, chapter_idx: int, seq: int) -> dict[str, Any] | None:
        return self._by_key.get((book_id, chapter_idx, seq))


@dataclass
class Context:
    """送进 LLM 的一段连续原文：同一章内若干相邻 chunk 合并而成。rank 取其中最靠前的命中排名。"""

    rank: int
    book_id: str
    chapter_idx: int
    seq_lo: int
    seq_hi: int  # 含
    char_start: int
    char_end: int
    text: str
    chunk_ids: list[str] = field(default_factory=list)
    hit_ids: list[str] = field(default_factory=list)  # 其中由检索直接命中的 chunk


def _stitch(chunks: Sequence[dict[str, Any]]) -> tuple[int, int, str]:
    """把同章、按 seq 排好的相邻 chunk 拼成一段：重叠部分只保留一次；块间未被覆盖的只有换行，按原长度补回。"""
    start, end = chunks[0]["char_start"], chunks[0]["char_end"]
    parts = [chunks[0]["text"]]
    for c in chunks[1:]:
        if c["char_end"] <= end:
            continue
        if c["char_start"] > end:
            parts.append("\n" * (c["char_start"] - end))
            parts.append(c["text"])
        else:
            parts.append(c["text"][end - c["char_start"] :])
        end = c["char_end"]
    return start, end, "".join(parts)


def expand_hits(hits: Sequence[Hit], index: ChunkIndex, window: int, flt: SearchFilter | None = None) -> list[Context]:
    """window=0 时每个命中块单独成段（不合并），即不扩展的基线。"""
    flt = flt or SearchFilter()
    groups: list[dict[str, Any]] = []  # 每组：book、chapter、seq 集合、rank、命中 id
    for h in hits:
        c = index.by_id(h.chunk_id)
        book, chap, seq = c["book_id"], c["chapter_idx"], c["seq"]
        seqs = {seq}
        for d in range(1, window + 1):
            for s in (seq - d, seq + d):
                n = index.get(book, chap, s)
                if n is not None and n["chunk_type"] == c["chunk_type"] and flt.accepts(n):
                    seqs.add(s)
        groups.append({"book": book, "chap": chap, "seqs": seqs, "rank": h.rank, "hits": [h.chunk_id]})
    # 同章内 seq 范围相交或相邻的组合并（保留更靠前的 rank），反复合并直到稳定；window=0 时不合并
    merged = True
    while merged and window > 0:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                if (
                    a["book"] == b["book"]
                    and a["chap"] == b["chap"]
                    and min(b["seqs"]) <= max(a["seqs"]) + 1
                    and max(b["seqs"]) >= min(a["seqs"]) - 1
                ):
                    a["seqs"] |= b["seqs"]
                    a["hits"] += b["hits"]
                    a["rank"] = min(a["rank"], b["rank"])
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    out = []
    for g in sorted(groups, key=lambda g: g["rank"]):
        seqs = sorted(g["seqs"])
        cs = [index.get(g["book"], g["chap"], s) for s in seqs]
        chunks = [c for c in cs if c is not None]
        start, end, text = _stitch(chunks)
        out.append(
            Context(
                g["rank"],
                g["book"],
                g["chap"],
                seqs[0],
                seqs[-1],
                start,
                end,
                text,
                [c["chunk_id"] for c in chunks],
                g["hits"],
            )
        )
    return out


def pack_contexts(contexts: Sequence[Context], budget: int, count_tokens: Callable[[Context], int]) -> list[Context]:
    """按排名贪心装箱：放得下就装，放不下就跳过、继续看后面的，直到用完预算。"""
    used, out = 0, []
    for c in contexts:
        n = count_tokens(c)
        if used + n <= budget:
            out.append(c)
            used += n
    return out
