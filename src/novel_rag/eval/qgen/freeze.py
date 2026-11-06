"""冻结评测集：从校验通过的候选中按题型选题、确定性划分 dev / test、校验后写入 eval/datasets/{name}/。

选题：每个题型最多 per_type_target 道；人工题优先全部入选，其余名额按"新增覆盖章节最多"贪心挑选生成题，
平局按候选 id 的 hash 排序。没入选的通过题写入 reserve.jsonl，留给下一版扩充。
划分：在"题型 × 来源"的每个分组内按 sha256(id) 排序后交替分配，dev 在前。加题不影响已有题的划分规则。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

from novel_rag.config import EvalGenConfig
from novel_rag.corpus import BookText
from novel_rag.eval.dataset import QUESTION_TYPES, DatasetManifest, EvalItem, Evidence, save_dataset, validate_items
from novel_rag.eval.qgen.review import ReviewLog, ReviewState, load_candidates, replay, review_stats
from novel_rag.prompts import load_prompt

GENERATOR = "claude-opus-5-5（会话内按出题规范生成）"


class FreezeError(RuntimeError):
    pass


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def item_id(qtype: str, question: str) -> str:
    return f"{qtype}-{hashlib.sha1(question.encode('utf-8')).hexdigest()[:8]}"


def select(kept: list[ReviewState], target: int) -> tuple[list[ReviewState], list[ReviewState]]:
    """返回（入选, 备选）。kept 应为同一题型、已通过校验的题。"""
    humans = sorted((s for s in kept if s.cand.get("source") == "human"), key=lambda s: _h(s.cand["cand_id"]))
    pool = sorted((s for s in kept if s.cand.get("source") != "human"), key=lambda s: _h(s.cand["cand_id"]))
    chosen = humans[:target]
    reserve = humans[target:]
    covered = {e["chapter_idx"] for s in chosen for e in s.evidence}
    while pool and len(chosen) < target:
        best = max(range(len(pool)), key=lambda i: (len({e["chapter_idx"] for e in pool[i].evidence} - covered), -i))
        s = pool.pop(best)
        chosen.append(s)
        covered |= {e["chapter_idx"] for e in s.evidence}
    return chosen, reserve + pool


def assign_split(items: list[EvalItem]) -> None:
    groups: dict[tuple[str, str], list[EvalItem]] = {}
    for it in items:
        groups.setdefault((it.type, it.source), []).append(it)
    for group in groups.values():
        for i, it in enumerate(sorted(group, key=lambda x: _h(x.id))):
            it.split = "dev" if i % 2 == 0 else "test"


def to_item(st: ReviewState, book: BookText) -> EvalItem:
    evidence = [
        Evidence(book_id=e["book_id"], chapter_idx=e["chapter_idx"], char_start=e["char_start"], char_end=e["char_end"])
        for e in st.evidence
    ]
    return EvalItem(
        id=item_id(st.cand["type"], st.question),
        type=st.cand["type"],
        question=st.question,
        answer=st.answer,
        evidence=evidence,
        source="human" if st.cand.get("source") == "human" else "generated",
        split="dev",  # 由 assign_split 覆盖
        max_reading_order=max(book.chapter(e.chapter_idx).reading_order for e in evidence),
        review="edited" if st.status == "edited" else "accepted",  # 草稿中的未校验题记为 accepted
        difficulty=st.difficulty,
        origin=st.cand["cand_id"],
    )


def freeze(
    book: BookText,
    cfg: EvalGenConfig,
    work: Path,
    out_dir: Path,
    name: str,
    *,
    allow_pending: bool = False,
    draft: bool = False,
    review_method: str | None = None,
) -> DatasetManifest:
    """draft=True：未校验的题也当作通过，用于在人工校验完成前跑通评测链路；草稿不能用于正式数字。"""
    cands = load_candidates(work)
    states = replay(cands, ReviewLog(work / "review.jsonl").read())
    pending = [cid for cid, s in states.items() if s.status == "pending"]
    if pending and not (allow_pending or draft):
        raise FreezeError(f"还有 {len(pending)} 道候选题未校验（例如 {pending[:3]}）；确认要跳过请加 --allow-pending")

    chosen_items: list[EvalItem] = []
    reserve_items: list[EvalItem] = []
    for t in QUESTION_TYPES:
        ok_status = ("accepted", "edited", "pending") if draft else ("accepted", "edited")
        kept = [s for s in states.values() if s.cand["type"] == t and s.status in ok_status]
        chosen, reserve = select(kept, cfg.per_type_target)
        chosen_items.extend(to_item(s, book) for s in chosen)
        reserve_items.extend(to_item(s, book) for s in reserve)
    assign_split(chosen_items)
    assign_split(reserve_items)

    errors = validate_items(
        chosen_items + reserve_items,
        {book.book_id: book},
        answer_max_chars=cfg.answer_max_chars,
        copyright_max_substring=cfg.copyright_max_substring,
    )
    if errors:
        raise FreezeError("评测集校验未通过：\n" + "\n".join(errors[:30]))

    spans = Counter((e.chapter_idx, e.char_start, e.char_end) for it in chosen_items for e in it.evidence)
    imports = {}
    for p in sorted(work.glob("import_*.json")):
        imports[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    manifest = DatasetManifest(
        name=name,
        draft=draft,
        review_method=review_method,
        books={book.book_id: book.text_sha256},
        counts={
            "total": len(chosen_items),
            "by_type": dict(Counter(i.type for i in chosen_items)),
            "by_source": dict(Counter(i.source for i in chosen_items)),
            "by_split": dict(Counter(i.split for i in chosen_items)),
            "by_type_split": {
                f"{t}/{sp}": n for (t, sp), n in sorted(Counter((i.type, i.split) for i in chosen_items).items())
            },
            "reserve": len(reserve_items),
        },
        created=date.today().isoformat(),
        generator=GENERATOR,
        guideline_version=load_prompt("qgen_guidelines").version_id,
        per_type_target=cfg.per_type_target,
        candidates_files={k: v.get("candidates_file_sha256") for k, v in imports.items()},
        import_stats={
            k: {kk: v[kk] for kk in ("questions", "ok", "by_type", "locate", "drop_reasons", "flags") if kk in v}
            for k, v in imports.items()
        },
        review_stats=review_stats(states),
        evidence_spans={
            "total": sum(spans.values()),
            "distinct": len(spans),
            "shared_by_multiple_items": sum(1 for n in spans.values() if n > 1),
        },
        chapter_coverage=len({e.chapter_idx for it in chosen_items for e in it.evidence}),
    )
    save_dataset(out_dir, chosen_items, manifest)
    (out_dir / "reserve.jsonl").write_text(
        "".join(it.model_dump_json() + "\n" for it in reserve_items), encoding="utf-8", newline=""
    )
    return manifest
