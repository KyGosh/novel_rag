"""出题窗口采样（第一轮：每章一个）与组合题配对（第二轮：按第一轮记录的人物做共现配对）。全部用固定种子，可复现。"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from novel_rag.config import EvalGenConfig
from novel_rag.corpus import BookText


@dataclass(frozen=True)
class Window:
    window_id: str
    chapter_idx: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ComboTask:
    task_id: str
    type: str  # relation / multihop / plot
    window_ids: tuple[str, ...]
    shared: tuple[str, ...] = field(default_factory=tuple)  # 窗口共同出现的人物（出题提示）


def sample_windows(book: BookText, cfg: EvalGenConfig) -> list[Window]:
    """每个章节取一段连续段落作为出题窗口，长度在 [min, max] 字之间（单段超长时允许超出）。"""
    allowed = {"narrative", "appendix"} if cfg.include_appendix else {"narrative"}
    windows = []
    for ch in book.chapters:
        if ch.section_type not in allowed:
            continue
        paras = book.paragraphs(ch)
        if not paras:
            continue
        rng = random.Random(f"{cfg.seed}:{book.book_id}:{ch.chapter_idx}")
        # 从每个段落起累积到首次达到下限；优先取不超过上限的窗口，没有再接受超长的（单段很长时）
        options = []
        for i in range(len(paras)):
            k = next((k for k in range(i, len(paras)) if paras[k][1] - paras[i][0] >= cfg.window_min_chars), None)
            if k is not None:
                options.append((paras[i][0], paras[k][1]))
        fitting = [o for o in options if o[1] - o[0] <= cfg.window_max_chars]
        pool = fitting or options
        start, end = rng.choice(pool) if pool else (paras[0][0], paras[-1][1])
        windows.append(Window(f"w{len(windows) + 1:04d}", ch.chapter_idx, start, end))
    return windows


def person_key(name: str) -> str:
    """人名归一：统一间隔号、去空白，取第一段（名）作为键，使"某某"与"某某·某某"能够配对。"""
    n = re.sub(r"[\s]", "", name)
    n = re.sub(r"[•‧・･∙⋅]", "·", n)
    return n.split("·")[0]


def pair_windows(windows: list[Window], persons: dict[str, list[str]], cfg: EvalGenConfig) -> list[ComboTask]:
    by_id = {w.window_id: w for w in windows}
    keys = {wid: {person_key(p) for p in names if person_key(p)} for wid, names in persons.items() if wid in by_id}
    ids = sorted(keys)
    tasks: list[ComboTask] = []

    def take(kind: str, candidates: list[tuple[tuple[str, ...], tuple[str, ...]]], n: int) -> None:
        rng = random.Random(f"{cfg.seed}:{kind}")
        candidates = sorted(set(candidates))
        rng.shuffle(candidates)
        uses: dict[str, int] = defaultdict(int)
        picked = 0
        for wids, shared in candidates:
            if picked >= n:
                break
            if any(uses[w] >= cfg.max_window_uses for w in wids):
                continue
            for w in wids:
                uses[w] += 1
            picked += 1
            tasks.append(ComboTask(f"r2-{kind}-{picked:03d}", kind, wids, shared))

    pairs = [
        ((a, b), tuple(sorted(keys[a] & keys[b])))
        for a, b in combinations(ids, 2)
        if by_id[a].chapter_idx != by_id[b].chapter_idx and keys[a] & keys[b]
    ]
    take("relation", [p for p in pairs if len(p[1]) >= 2], cfg.combo_counts.get("relation", 0))
    take("multihop", pairs, cfg.combo_counts.get("multihop", 0))

    # 跨章节：同一人物、相邻窗口至少相隔 plot_min_chapter_gap 章的链，长度 2..plot_windows
    chains = []
    holders: dict[str, list[str]] = defaultdict(list)
    for wid in ids:
        for k in keys[wid]:
            holders[k].append(wid)
    for person, wids in holders.items():
        wids = sorted(wids, key=lambda w: by_id[w].chapter_idx)
        for i, first in enumerate(wids):
            chain = [first]
            for w in wids[i + 1 :]:
                if by_id[w].chapter_idx - by_id[chain[-1]].chapter_idx >= cfg.plot_min_chapter_gap:
                    chain.append(w)
                    if len(chain) == cfg.plot_windows:
                        break
            if len(chain) >= 2:
                chains.append((tuple(chain), (person,)))
    take("plot", chains, cfg.combo_counts.get("plot", 0))
    return tasks
