"""评测集 v1：多卷出题工具（规范见 prompts/qgen_guidelines.v2.md）。

与 v0 单卷工具的区别：
- 窗口编号带卷号（v{卷}w{序号}），每个窗口记录 book_id 与 reading_order；证据按窗口所在的卷定位。
- 第一轮写结构化人物标注（全名 + 群体 + 一句话身份，同名者加限定语）和事实笔记（一句话 + 原文引文）；
  约 N 个窗口另出事实题。
- 第二轮按人物**完整身份**（full_name 精确匹配，含限定语）配对，修掉"按名字第一段配对"的问题；
  人物关系 / 多跳 / 跨章节三类各自至少 cross_volume_ratio 的任务来自不同卷；跨章节题按 reading_order 计章节距离。
- 只在正文章节采样，不用附录。出题、配对都不使用系统自己的图谱和摘要。

工作目录 data/eval_work/{dataset}/（含原文，不入库）：windows.json、tasks_r{N}.jsonl、candidates_r{N}.jsonl、
notes_r1.jsonl（第一轮笔记，引文已定位）、located_r{N}.jsonl、import_r{N}.json、review.jsonl。
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from novel_rag.config import EvalGenConfig
from novel_rag.corpus import BookText
from novel_rag.eval.dataset import QuestionType
from novel_rag.eval.qgen.locate import locate
from novel_rag.eval.qgen.prefilter import apply_prefilter
from novel_rag.eval.qgen.sampling import sample_windows

Library = dict[str, BookText]


def volume_of(book_id: str) -> int:
    m = re.search(r"(\d+)$", book_id)
    return int(m.group(1)) if m else 0


@dataclass(frozen=True)
class SWindow:
    window_id: str
    book_id: str
    chapter_idx: int
    reading_order: int
    char_start: int
    char_end: int


class PersonNote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    full_name: str
    group: str | None = None
    desc: str = ""


class FactNote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact: str
    quote: str
    persons: list[str] = Field(default_factory=list)


class SEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    window_id: str
    quote: str


class SQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: QuestionType
    question: str
    answer: str
    answer_aliases: list[str] = Field(default_factory=list)
    evidence: list[SEvidence] = Field(min_length=1)
    note: str = ""


class SLine(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    persons: list[PersonNote] = Field(default_factory=list)
    facts: list[FactNote] = Field(default_factory=list)
    questions: list[SQuestion] = Field(default_factory=list)
    note: str = ""


@dataclass(frozen=True)
class SCombo:
    task_id: str
    type: str
    window_ids: tuple[str, ...]
    shared: tuple[str, ...]
    cross_volume: bool


# ---------------------------------------------------------------- 工具


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="")


def identity(full_name: str) -> str:
    """人物身份键：统一间隔号与括号、去空白；限定语保留（同名不同人靠它区分）。"""
    n = re.sub(r"\s+", "", full_name or "")
    n = re.sub(r"[•‧・･∙⋅]", "·", n)
    return n.replace("(", "（").replace(")", "）")


def load_windows(work: Path) -> list[SWindow]:
    return [SWindow(**w) for w in json.loads((work / "windows.json").read_text(encoding="utf-8"))]


def window_payload(books: Library, w: SWindow) -> dict[str, Any]:
    book = books[w.book_id]
    ch = book.chapter(w.chapter_idx)
    return {
        **asdict(w),
        "book_title": book.title,
        "chapter_title": ch.title,
        "pov": ch.pov,
        "text": book.text[w.char_start : w.char_end],
    }


# ---------------------------------------------------------------- 第一轮


def sample_series_windows(books: list[BookText], cfg: EvalGenConfig) -> list[SWindow]:
    if cfg.include_appendix:
        raise ValueError("v1 只在正文章节采样，include_appendix 必须为 False")
    out: list[SWindow] = []
    for book in sorted(books, key=lambda b: volume_of(b.book_id)):
        v = volume_of(book.book_id)
        for i, w in enumerate(sample_windows(book, cfg), 1):
            ro = book.chapter(w.chapter_idx).reading_order
            out.append(SWindow(f"v{v}w{i:04d}", book.book_id, w.chapter_idx, ro, w.char_start, w.char_end))
    return out


def pick_fact_windows(windows: list[SWindow], n: int, seed: int) -> set[str]:
    """按卷分层、固定种子抽 n 个窗口出事实题（各卷按窗口数比例分配名额）。"""
    by_vol: dict[str, list[SWindow]] = defaultdict(list)
    for w in windows:
        by_vol[w.book_id].append(w)
    total = len(windows)
    picked: set[str] = set()
    for b, ws in sorted(by_vol.items()):
        k = round(n * len(ws) / total)
        rng = random.Random(f"{seed}:fact:{b}")
        picked.update(w.window_id for w in rng.sample(ws, min(k, len(ws))))
    return picked


def export_round1(books: Library, cfg: EvalGenConfig, work: Path, n_fact_windows: int) -> Path:
    windows = sample_series_windows(list(books.values()), cfg)
    facts = pick_fact_windows(windows, n_fact_windows, cfg.seed)
    work.mkdir(parents=True, exist_ok=True)
    (work / "windows.json").write_text(
        json.dumps([asdict(w) for w in windows], ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    rows = [
        {
            "task_id": f"r1-{w.window_id}",
            "round": 1,
            "type": "fact",
            "want_questions": w.window_id in facts,
            "windows": [window_payload(books, w)],
        }
        for w in windows
    ]
    out = work / "tasks_r1.jsonl"
    write_jsonl(out, rows)
    return out


def _locate_in_window(books: Library, w: dict[str, Any], quote: str, cfg: EvalGenConfig):  # noqa: ANN202
    book = books[w["book_id"]]
    return locate(book.text, quote.strip(), w["char_start"], w["char_end"], fuzzy_threshold=cfg.fuzzy_threshold)


def import_notes_r1(books: Library, cfg: EvalGenConfig, work: Path) -> dict[str, Any]:
    """第一轮的人物标注与事实笔记 → notes_r1.jsonl（引文定位成区间；定位失败的笔记标记出来，第二轮不展示）。"""
    tasks = {t["task_id"]: t for t in read_jsonl(work / "tasks_r1.jsonl")}
    notes, stats = [], Counter()
    for row in read_jsonl(work / "candidates_r1.jsonl"):
        line = SLine.model_validate(row)
        task = tasks.get(line.task_id)
        if task is None:
            stats["unknown_task"] += 1
            continue
        w = task["windows"][0]
        facts = []
        for f in line.facts:
            loc = _locate_in_window(books, w, f.quote, cfg)
            stats["facts"] += 1
            stats["facts_located" if loc else "facts_unlocated"] += 1
            facts.append({**f.model_dump(), "located": loc is not None, "span": [loc.start, loc.end] if loc else None})
        stats["persons"] += len(line.persons)
        notes.append(
            {
                "window_id": w["window_id"],
                "book_id": w["book_id"],
                "reading_order": w["reading_order"],
                "persons": [p.model_dump() for p in line.persons],
                "facts": facts,
            }
        )
    write_jsonl(work / "notes_r1.jsonl", notes)
    stats["windows_with_notes"] = len(notes)
    stats["windows_total"] = len(tasks)
    return dict(stats)


# ---------------------------------------------------------------- 第二轮：按身份配对


def pair_by_identity(
    windows: list[SWindow], notes: dict[str, list[str]], cfg: EvalGenConfig, counts: dict[str, int]
) -> list[SCombo]:
    """notes：window_id → 该窗口人物的身份键列表。counts：各题型的任务数。"""
    by_id = {w.window_id: w for w in windows}
    keys = {wid: set(ids) for wid, ids in notes.items() if wid in by_id and ids}
    ids = sorted(keys)
    tasks: list[SCombo] = []

    def cross(wids: tuple[str, ...]) -> bool:
        return len({by_id[w].book_id for w in wids}) > 1

    def take(kind: str, candidates: list[tuple[tuple[str, ...], tuple[str, ...]]], n: int) -> None:
        rng = random.Random(f"{cfg.seed}:v1:{kind}")
        candidates = sorted(set(candidates))
        rng.shuffle(candidates)
        uses: dict[str, int] = defaultdict(int)
        chosen: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        n_cross = math.ceil(cfg.cross_volume_ratio * n)

        def try_add(c: tuple[tuple[str, ...], tuple[str, ...]]) -> None:
            if len(chosen) >= n or c in chosen or any(uses[w] >= cfg.max_window_uses for w in c[0]):
                return
            for w in c[0]:
                uses[w] += 1
            chosen.append(c)

        for c in candidates:  # 先满足跨卷配额
            if sum(cross(x[0]) for x in chosen) >= n_cross:
                break
            if cross(c[0]):
                try_add(c)
        for c in candidates:
            try_add(c)
        for i, (wids, shared) in enumerate(chosen, 1):
            tasks.append(SCombo(f"r2-{kind}-{i:03d}", kind, wids, shared, cross(wids)))

    pairs = [
        ((a, b), tuple(sorted(keys[a] & keys[b])))
        for a, b in combinations(ids, 2)
        if by_id[a].reading_order != by_id[b].reading_order and keys[a] & keys[b]
    ]
    take("relation", [p for p in pairs if len(p[1]) >= 2], counts.get("relation", 0))
    take("multihop", pairs, counts.get("multihop", 0))
    chains = []
    holders: dict[str, list[str]] = defaultdict(list)
    for wid in ids:
        for k in keys[wid]:
            holders[k].append(wid)
    for person, wids in holders.items():
        wids = sorted(wids, key=lambda w: by_id[w].reading_order)
        for i, first in enumerate(wids):
            chain = [first]
            for w in wids[i + 1 :]:
                if by_id[w].reading_order - by_id[chain[-1]].reading_order >= cfg.plot_min_chapter_gap:
                    chain.append(w)
                    if len(chain) == cfg.plot_windows:
                        break
            if len(chain) >= 2:
                chains.append((tuple(chain), (person,)))
    take("plot", chains, counts.get("plot", 0))
    return tasks


def export_round2(books: Library, cfg: EvalGenConfig, work: Path, counts: dict[str, int]) -> Path:
    windows = load_windows(work)
    by_id = {w.window_id: w for w in windows}
    notes = {n["window_id"]: n for n in read_jsonl(work / "notes_r1.jsonl")}
    person_keys = {wid: [identity(p["full_name"]) for p in n["persons"]] for wid, n in notes.items()}
    combos = pair_by_identity(windows, person_keys, cfg, counts)
    rows = []
    for t in combos:
        wins = []
        for wid in t.window_ids:
            n = notes.get(wid, {"persons": [], "facts": []})
            wins.append(
                {
                    **window_payload(books, by_id[wid]),
                    "persons": n["persons"],
                    "facts": [f for f in n["facts"] if f["located"]],
                }
            )
        rows.append(
            {
                "task_id": t.task_id,
                "round": 2,
                "type": t.type,
                "shared": list(t.shared),
                "cross_volume": t.cross_volume,
                "windows": wins,
            }
        )
    out = work / "tasks_r2.jsonl"
    write_jsonl(out, rows)
    return out


# ---------------------------------------------------------------- 导入候选题


def import_round(books: Library, cfg: EvalGenConfig, work: Path, round_no: int) -> dict[str, Any]:
    tasks = {t["task_id"]: t for t in read_jsonl(work / f"tasks_r{round_no}.jsonl")}
    cand_path = work / f"candidates_r{round_no}.jsonl"
    raw = cand_path.read_bytes()
    issues: list[str] = []
    located: list[dict[str, Any]] = []
    seen: set[str] = set()
    for n, row in enumerate(read_jsonl(cand_path), 1):
        line = SLine.model_validate(row)
        task = tasks.get(line.task_id)
        if task is None:
            issues.append(f"第 {n} 行：未知的 task_id {line.task_id}")
            continue
        if line.task_id in seen:
            issues.append(f"第 {n} 行：task_id {line.task_id} 重复出现，已忽略")
            continue
        seen.add(line.task_id)
        wins = {w["window_id"]: w for w in task["windows"]}
        for k, q in enumerate(line.questions, 1):
            cand: dict[str, Any] = {
                "cand_id": f"{line.task_id}-q{k}",
                "task_id": line.task_id,
                "round": round_no,
                "type": q.type,
                "question": q.question.strip(),
                "answer": q.answer.strip(),
                "answer_aliases": [a.strip() for a in q.answer_aliases if a.strip()],
                "note": q.note,
                "cross_volume": bool(task.get("cross_volume", False)),
                "evidence": [],
                "status": "ok",
                "drop_reasons": [],
                "flags": [],
            }
            if q.type != task["type"]:
                cand["drop_reasons"].append("type_mismatch")
            for ev in q.evidence:
                w = wins.get(ev.window_id)
                quote = ev.quote.strip()
                if w is None:
                    cand["drop_reasons"].append("unknown_window")
                    continue
                if len(quote) < cfg.quote_min_chars:
                    cand["drop_reasons"].append("quote_too_short")
                    continue
                if len(quote) > cfg.quote_max_chars:
                    cand["drop_reasons"].append("quote_too_long")
                    continue
                loc = _locate_in_window(books, w, quote, cfg)
                if loc is None:
                    cand["drop_reasons"].append("locate_failed")
                    continue
                cand["evidence"].append(
                    {
                        "window_id": w["window_id"],
                        "book_id": w["book_id"],
                        "chapter_idx": w["chapter_idx"],
                        "char_start": loc.start,
                        "char_end": loc.end,
                        "method": loc.method,
                        "score": loc.score,
                    }
                )
            if cand["drop_reasons"]:
                cand["status"] = "dropped"
            located.append(cand)
    earlier = []
    for r in range(1, round_no):
        p = work / f"located_r{r}.jsonl"
        if p.is_file():
            earlier.extend(read_jsonl(p))
    apply_prefilter(located, books, cfg, earlier=earlier)
    write_jsonl(work / f"located_r{round_no}.jsonl", located)
    by_type: dict[str, Counter] = {}
    for c in located:
        by_type.setdefault(c["type"], Counter())[c["status"]] += 1
    stats = {
        "round": round_no,
        "candidates_file_sha256": hashlib.sha256(raw).hexdigest(),
        "tasks_total": len(tasks),
        "tasks_answered": len(seen),
        "questions": len(located),
        "ok": sum(c["status"] == "ok" for c in located),
        "by_type": {k: dict(v) for k, v in sorted(by_type.items())},
        "cross_volume_ok": sum(c["status"] == "ok" and c["cross_volume"] for c in located),
        "locate": dict(Counter(e["method"] for c in located for e in c["evidence"])),
        "drop_reasons": dict(Counter(r for c in located for r in c["drop_reasons"])),
        "flags": dict(Counter(f.split(":")[0] for c in located for f in c["flags"])),
        "issues": issues,
    }
    (work / f"import_r{round_no}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


# ---------------------------------------------------------------- 给出题者看的紧凑视图


def render_tasks(work: Path, round_no: int, start: int, end: int, *, with_text: bool) -> str:
    rows = read_jsonl(work / f"tasks_r{round_no}.jsonl")[start:end]
    out = []
    for t in rows:
        head = f"### {t['task_id']}（{t['type']}"
        head += "，出事实题" if t.get("want_questions") else ""
        head += "，跨卷" if t.get("cross_volume") else ""
        head += f"，共同人物：{'、'.join(t['shared'])}" if t.get("shared") else ""
        out.append(head + "）")
        for w in t["windows"]:
            out.append(f"[{w['window_id']}] 《{w['book_title']}》{w['chapter_title']}（ro {w['reading_order']}）")
            if round_no == 2:
                out.append(
                    "  人物："
                    + "；".join(f"{p['full_name']}〔{p.get('group') or '-'}〕{p['desc']}" for p in w["persons"])
                )
                for f in w["facts"]:
                    out.append(f"  - {f['fact']}　「{f['quote']}」")
            if with_text or round_no == 1:
                out.append(w["text"])
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- 独立复核与冻结

REVIEW_CHECKLIST = (
    "support",  # 证据能否支撑答案
    "unique",  # 是否有唯一答案
    "type_fit",  # 题型是否名副其实（多跳是否真的需要两段证据）
    "standalone",  # 问题措辞能否独立理解
    "disambiguated",  # 同名限定是否充分
)


def review_sheet(books: Library, work: Path, start: int, end: int) -> str:
    """独立复核用的视图：只给问题、答案、可接受写法和证据原文（不给出题笔记和任务上下文）。"""
    from novel_rag.eval.qgen.review import load_candidates

    cands = sorted(load_candidates(work), key=lambda c: c["cand_id"])[start:end]
    out = []
    for c in cands:
        out.append(f"### {c['cand_id']}［{c['type']}］")
        out.append(f"问：{c['question']}")
        alias = f"（亦可：{'、'.join(c.get('answer_aliases', []))}）" if c.get("answer_aliases") else ""
        out.append(f"答：{c['answer']}{alias}")
        for i, e in enumerate(c["evidence"], 1):
            book = books[e["book_id"]]
            ch = book.chapter(e["chapter_idx"])
            text = book.text[e["char_start"] : e["char_end"]]
            out.append(f"  证据{i}（{book.title}·{ch.title}，ro {ch.reading_order}）：{text}")
        out.append("")
    return "\n".join(out)


def import_review(work: Path, verdicts_path: Path, reviewer: str) -> dict[str, Any]:
    """把复核结论（jsonl：cand_id、action=accept/edit/drop、可选 question/answer/answer_aliases/reason/checks/note）
    追加进 review.jsonl。已有结论的题不重复追加。"""
    from novel_rag.eval.qgen.review import DROP_REASONS, ReviewLog, load_candidates, replay

    log = ReviewLog(work / "review.jsonl")
    cands = {c["cand_id"]: c for c in load_candidates(work)}
    states = replay(list(cands.values()), log.read())
    stats: Counter = Counter()
    valid_reasons = set(DROP_REASONS.values())
    for v in read_jsonl(verdicts_path):
        cid = v["cand_id"]
        if cid not in cands:
            stats["unknown"] += 1
            continue
        if states[cid].status != "pending":
            stats["already_reviewed"] += 1
            continue
        extra = {k: v[k] for k in ("checks", "note") if v.get(k)}
        if v["action"] == "edit":
            log.append(
                {"cand_id": cid, "action": "edit", "reviewer": reviewer, **{
                    k: v[k] for k in ("question", "answer", "answer_aliases") if k in v
                }}
            )  # fmt: skip
            log.append({"cand_id": cid, "action": "accept", "reviewer": reviewer, **extra})
        elif v["action"] == "drop":
            reason = v.get("reason") if v.get("reason") in valid_reasons else "other"
            log.append({"cand_id": cid, "action": "drop", "reason": reason, "reviewer": reviewer, **extra})
        elif v["action"] == "accept":
            log.append({"cand_id": cid, "action": "accept", "reviewer": reviewer, **extra})
        else:
            stats["bad_action"] += 1
            continue
        stats[v["action"]] += 1
    return dict(stats)


def freeze_series(
    books: Library,
    cfg: EvalGenConfig,
    work: Path,
    out_dir: Path,
    name: str,
    *,
    review_method: str,
    allow_pending: bool = False,
) -> Any:
    from datetime import date

    from novel_rag.eval.dataset import QUESTION_TYPES, DatasetManifest, EvalItem, Evidence, save_dataset, validate_items
    from novel_rag.eval.qgen.freeze import FreezeError, assign_split, item_id, select
    from novel_rag.eval.qgen.review import ReviewLog, load_candidates, replay, review_stats
    from novel_rag.prompts import load_prompt

    states = replay(load_candidates(work), ReviewLog(work / "review.jsonl").read())
    pending = [cid for cid, s in states.items() if s.status == "pending"]
    if pending and not allow_pending:
        raise FreezeError(f"还有 {len(pending)} 道候选题未复核（例如 {pending[:3]}）")

    def to_item(st: Any) -> EvalItem:
        ev = [
            Evidence(
                book_id=e["book_id"], chapter_idx=e["chapter_idx"], char_start=e["char_start"], char_end=e["char_end"]
            )
            for e in st.evidence
        ]
        return EvalItem(
            id=item_id(st.cand["type"], st.question),
            type=st.cand["type"],
            question=st.question,
            answer=st.answer,
            answer_aliases=st.answer_aliases,
            evidence=ev,
            source="generated",
            split="dev",
            max_reading_order=max(books[e.book_id].chapter(e.chapter_idx).reading_order for e in ev),
            review="edited" if st.status == "edited" else "accepted",
            origin=st.cand["cand_id"],
        )

    chosen, reserve = [], []
    for t in QUESTION_TYPES:
        kept = [s for s in states.values() if s.cand["type"] == t and s.status in ("accepted", "edited")]
        c, r = select(kept, cfg.v1_targets.get(t, 0))
        chosen.extend(to_item(s) for s in c)
        reserve.extend(to_item(s) for s in r)
    assign_split(chosen)
    assign_split(reserve)
    errors = validate_items(
        chosen + reserve,
        books,
        answer_max_chars=cfg.answer_max_chars,
        copyright_max_substring=cfg.copyright_max_substring,
    )
    if errors:
        raise FreezeError("评测集校验未通过：\n" + "\n".join(errors[:30]))
    imports = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(work.glob("import_*.json"))}
    combo = [i for i in chosen if i.type != "fact"]
    cross = [i for i in combo if len({e.book_id for e in i.evidence}) > 1]
    spans = Counter((e.book_id, e.char_start, e.char_end) for it in chosen for e in it.evidence)
    by_type_split = Counter((i.type, i.split) for i in chosen)
    manifest = DatasetManifest(
        name=name,
        books={b: book.text_sha256 for b, book in sorted(books.items())},
        counts={
            "total": len(chosen),
            "by_type": dict(Counter(i.type for i in chosen)),
            "by_split": dict(Counter(i.split for i in chosen)),
            "by_type_split": {f"{t}/{sp}": n for (t, sp), n in sorted(by_type_split.items())},
            "reserve": len(reserve),
            "cross_volume_evidence": {"combo_items": len(combo), "cross_volume": len(cross)},
            "by_last_volume": dict(Counter(max(e.book_id for e in i.evidence) for i in chosen)),
        },
        created=date.today().isoformat(),
        generator="claude-opus-5-5（会话内按出题规范 v2 生成）",
        guideline_version=load_prompt("qgen_guidelines", 2).version_id,
        review_method=review_method,
        human_reviewed=False,
        review_note="自动校验 + 模型复核（Claude 独立复核），未经人工复核；使用 v1 数字的报告都要注明这一点",
        targets=cfg.v1_targets,
        candidates_files={k: v.get("candidates_file_sha256") for k, v in imports.items()},
        import_stats=imports,
        review_stats=review_stats(states),
        evidence_spans={"total": sum(spans.values()), "distinct": len(spans)},
        chapter_coverage=len({(e.book_id, e.chapter_idx) for it in chosen for e in it.evidence}),
    )
    save_dataset(out_dir, chosen, manifest)
    (out_dir / "reserve.jsonl").write_text(
        "".join(it.model_dump_json() + "\n" for it in reserve), encoding="utf-8", newline=""
    )
    return manifest
