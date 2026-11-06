"""出题任务的导出与候选题的导入。

工作目录 data/eval_work/{dataset}/（属于 data/，不入库，因为任务与候选题含原文）：
    windows.json          第一轮采样的窗口
    tasks_r{N}.jsonl      导出的出题任务（含窗口原文，实时读自 text.txt）
    candidates_r{N}.jsonl 出题者按 prompts/qgen_guidelines 写出的候选题（每行对应一个任务）
    located_r{N}.jsonl    导入结果：证据已定位成字符区间，带预筛状态与标记（不含原文）
    import_r{N}.json      导入统计
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from novel_rag.config import EvalGenConfig
from novel_rag.corpus import BookText
from novel_rag.eval.dataset import QuestionType
from novel_rag.eval.qgen.locate import locate
from novel_rag.eval.qgen.prefilter import apply_prefilter
from novel_rag.eval.qgen.sampling import ComboTask, Window, pair_windows, sample_windows


class CandEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    window_id: str
    quote: str


class CandQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: QuestionType
    question: str
    answer: str
    evidence: list[CandEvidence] = Field(min_length=1)
    note: str = ""


class CandidateLine(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    persons: list[str] = Field(default_factory=list)
    questions: list[CandQuestion] = Field(default_factory=list)
    note: str = ""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="")


def _window_payload(book: BookText, w: Window) -> dict[str, Any]:
    ch = book.chapter(w.chapter_idx)
    return {
        "window_id": w.window_id,
        "chapter_idx": w.chapter_idx,
        "chapter_title": ch.title,
        "pov": ch.pov,
        "char_start": w.char_start,
        "char_end": w.char_end,
        "text": book.text[w.char_start : w.char_end],
    }


def load_windows(work: Path) -> list[Window]:
    return [Window(**w) for w in json.loads((work / "windows.json").read_text(encoding="utf-8"))]


def export_round1(book: BookText, cfg: EvalGenConfig, work: Path) -> Path:
    windows = sample_windows(book, cfg)
    work.mkdir(parents=True, exist_ok=True)
    (work / "windows.json").write_text(
        json.dumps([asdict(w) for w in windows], ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    rows = [
        {"task_id": f"r1-{i:04d}", "round": 1, "type": "fact", "shared": [], "windows": [_window_payload(book, w)]}
        for i, w in enumerate(windows, 1)
    ]
    out = work / "tasks_r1.jsonl"
    _write_jsonl(out, rows)
    return out


def round1_persons(work: Path) -> dict[str, list[str]]:
    """第一轮候选题中记录的人物，按窗口汇总。"""
    tasks = {t["task_id"]: t for t in _read_jsonl(work / "tasks_r1.jsonl")}
    persons: dict[str, list[str]] = {}
    for row in _read_jsonl(work / "candidates_r1.jsonl"):
        line = CandidateLine.model_validate(row)
        if line.task_id in tasks:
            persons[tasks[line.task_id]["windows"][0]["window_id"]] = line.persons
    return persons


def export_round2(book: BookText, cfg: EvalGenConfig, work: Path) -> Path:
    windows = load_windows(work)
    by_id = {w.window_id: w for w in windows}
    combos: list[ComboTask] = pair_windows(windows, round1_persons(work), cfg)
    rows = [
        {
            "task_id": t.task_id,
            "round": 2,
            "type": t.type,
            "shared": list(t.shared),
            "windows": [_window_payload(book, by_id[w]) for w in t.window_ids],
        }
        for t in combos
    ]
    out = work / "tasks_r2.jsonl"
    _write_jsonl(out, rows)
    return out


def import_round(book: BookText, cfg: EvalGenConfig, work: Path, round_no: int) -> dict[str, Any]:
    tasks = {t["task_id"]: t for t in _read_jsonl(work / f"tasks_r{round_no}.jsonl")}
    cand_path = work / f"candidates_r{round_no}.jsonl"
    raw = cand_path.read_bytes()
    issues: list[str] = []
    located: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for n, row in enumerate(_read_jsonl(cand_path), 1):
        line = CandidateLine.model_validate(row)
        task = tasks.get(line.task_id)
        if task is None:
            issues.append(f"第 {n} 行：未知的 task_id {line.task_id}")
            continue
        if line.task_id in seen_tasks:
            issues.append(f"第 {n} 行：task_id {line.task_id} 重复出现，已忽略")
            continue
        seen_tasks.add(line.task_id)
        wins = {w["window_id"]: w for w in task["windows"]}
        for k, q in enumerate(line.questions, 1):
            cand: dict[str, Any] = {
                "cand_id": f"{line.task_id}-q{k}",
                "task_id": line.task_id,
                "round": round_no,
                "type": q.type,
                "question": q.question.strip(),
                "answer": q.answer.strip(),
                "note": q.note,
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
                loc = locate(book.text, quote, w["char_start"], w["char_end"], fuzzy_threshold=cfg.fuzzy_threshold)
                if loc is None:
                    cand["drop_reasons"].append("locate_failed")
                    continue
                cand["evidence"].append(
                    {
                        "window_id": w["window_id"],
                        "book_id": book.book_id,
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
            earlier.extend(_read_jsonl(p))
    apply_prefilter(located, book, cfg, earlier=earlier)
    _write_jsonl(work / f"located_r{round_no}.jsonl", located)

    stats = {
        "round": round_no,
        "candidates_file_sha256": hashlib.sha256(raw).hexdigest(),
        "tasks_total": len(tasks),
        "tasks_answered": len(seen_tasks),
        **_import_stats(located),
        "issues": issues,
    }
    (work / f"import_r{round_no}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def _import_stats(located: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, Counter] = {}
    for c in located:
        by_type.setdefault(c["type"], Counter())[c["status"]] += 1
    return {
        "questions": len(located),
        "ok": sum(c["status"] == "ok" for c in located),
        "by_type": {k: dict(v) for k, v in sorted(by_type.items())},
        "locate": dict(Counter(e["method"] for c in located for e in c["evidence"])),
        "drop_reasons": dict(Counter(r for c in located for r in c["drop_reasons"])),
        "flags": dict(Counter(f.split(":")[0] for c in located for f in c["flags"])),
    }


# ---------------------------------------------------------------- 人工题


class HumanEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quote: str
    chapter_idx: int | None = None  # 片段在全书出现多处时用来消歧


class HumanQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: QuestionType
    question: str
    answer: str
    evidence: list[HumanEvidence] = Field(min_length=1)
    note: str = ""


def _locate_human(book: BookText, cfg: EvalGenConfig, ev: HumanEvidence) -> tuple[Any, str | None]:
    """返回（Located 或 None, 失败原因）。给了章节就在该章内三级定位；否则要求原文片段在全书唯一出现。"""
    quote = ev.quote.strip()
    if ev.chapter_idx is not None:
        try:
            ch = book.chapter(ev.chapter_idx)
        except KeyError:
            return None, "unknown_chapter"
        loc = locate(book.text, quote, ch.char_start, ch.char_end, fuzzy_threshold=cfg.fuzzy_threshold)
        return loc, None if loc else "locate_failed"
    first = book.text.find(quote)
    if first != -1 and book.text.find(quote, first + 1) != -1:
        return None, "ambiguous_quote"
    loc = locate(book.text, quote, 0, len(book.text), fuzzy_threshold=cfg.fuzzy_threshold)
    return loc, None if loc else "locate_failed"


def import_human(book: BookText, cfg: EvalGenConfig, work: Path) -> dict[str, Any]:
    """导入人工题 human.jsonl（每行一题，字段见 HumanQuestion），结果写 located_human.jsonl，同样经过预筛与人工校验。"""
    path = work / "human.jsonl"
    raw = path.read_bytes()
    located: list[dict[str, Any]] = []
    for n, row in enumerate(_read_jsonl(path), 1):
        q = HumanQuestion.model_validate(row)
        cand: dict[str, Any] = {
            "cand_id": f"h-{n:04d}",
            "task_id": None,
            "round": 0,
            "source": "human",
            "type": q.type,
            "question": q.question.strip(),
            "answer": q.answer.strip(),
            "note": q.note,
            "evidence": [],
            "status": "ok",
            "drop_reasons": [],
            "flags": [],
        }
        for ev in q.evidence:
            if len(ev.quote.strip()) < cfg.quote_min_chars:
                cand["drop_reasons"].append("quote_too_short")
                continue
            loc, reason = _locate_human(book, cfg, ev)
            if loc is None:
                cand["drop_reasons"].append(reason)
                continue
            ch = book.chapter_at(loc.start)
            assert ch is not None
            cand["evidence"].append(
                {
                    "window_id": f"ch{ch.chapter_idx}",  # 人工题没有出题窗口，用章节代替（预筛按它判断是否跨章）
                    "book_id": book.book_id,
                    "chapter_idx": ch.chapter_idx,
                    "char_start": loc.start,
                    "char_end": loc.end,
                    "method": loc.method,
                    "score": loc.score,
                }
            )
        if cand["drop_reasons"]:
            cand["status"] = "dropped"
        located.append(cand)
    earlier = [c for name in ("located_r1.jsonl", "located_r2.jsonl") if (work / name).is_file()
               for c in _read_jsonl(work / name)]  # fmt: skip
    apply_prefilter(located, book, cfg, earlier=earlier)
    _write_jsonl(work / "located_human.jsonl", located)
    stats = {"candidates_file_sha256": hashlib.sha256(raw).hexdigest(), **_import_stats(located)}
    (work / "import_human.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats
