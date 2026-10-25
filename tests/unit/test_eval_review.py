"""人工校验状态机、人工题导入、冻结（选题、划分、校验）。样例均为自拟。"""

import json

import pytest

from novel_rag.config import EvalGenConfig
from novel_rag.corpus import BookText
from novel_rag.eval.dataset import load_dataset
from novel_rag.eval.qgen.freeze import FreezeError, assign_split, freeze, item_id, select
from novel_rag.eval.qgen.review import ReviewLog, ReviewState, replay, respan_evidence, review_order, review_stats
from novel_rag.eval.qgen.tasks import import_human
from tests.unit.book_factory import make_book, para

CFG = EvalGenConfig(per_type_target=2)


@pytest.fixture
def book(tmp_path):
    chapters = [(f"第{i}章", [f"码头上第{i}艘船载着{i * 3}箱香料靠岸。", para(f"船员{i}", 60)]) for i in range(1, 7)]
    make_book(tmp_path / "books", "demo-1", chapters)
    return BookText.load(tmp_path / "books" / "demo-1")


def _ev(book, ch, offset=0, length=8):
    c = book.chapter(ch)
    s = c.body_start + offset
    return {
        "window_id": f"w{ch}",
        "book_id": book.book_id,
        "chapter_idx": ch,
        "char_start": s,
        "char_end": s + length,
        "method": "exact",
        "score": 100.0,
    }


def _cand(cid, qtype, chapters, book, source=None, question=None):
    c = {
        "cand_id": cid,
        "type": qtype,
        "question": question or f"{cid} 的问题是什么？",
        "answer": "答案",
        "note": "",
        "evidence": [_ev(book, ch) for ch in chapters],
        "status": "ok",
        "drop_reasons": [],
        "flags": [],
    }
    if source:
        c["source"] = source
    return c


# ---------------------------------------------------------------- 状态机


def test_replay_accept_edit_drop_and_undo(book):
    cands = [_cand("c1", "fact", [1], book), _cand("c2", "fact", [2], book), _cand("c3", "plot", [3, 4], book)]
    log = [
        {"cand_id": "c1", "action": "accept", "difficulty": 2},
        {"cand_id": "c2", "action": "edit", "answer": "新答案"},
        {"cand_id": "c3", "action": "drop", "reason": "too_easy"},
        {"cand_id": "c3", "action": "undo"},
    ]
    st = replay(cands, log)
    assert (st["c1"].status, st["c1"].difficulty) == ("accepted", 2)
    assert (st["c2"].status, st["c2"].answer) == ("edited", "新答案")
    assert st["c3"].status == "pending" and st["c3"].drop_reason is None
    # 修改后又改回原样，算作直接通过
    st = replay(cands, log + [{"cand_id": "c2", "action": "edit", "answer": "答案"}])
    assert st["c2"].status == "accepted"


def test_respan_requires_reconfirm_and_counts_as_edit(book):
    cands = [_cand("c1", "fact", [1], book)]
    new_ev = respan_evidence(book, cands[0]["evidence"], 0, "码头上第1艘船", 1, fuzzy_threshold=90)
    assert book.text[new_ev[0]["char_start"] : new_ev[0]["char_end"]] == "码头上第1艘船"
    assert respan_evidence(book, cands[0]["evidence"], None, "完全不存在的句子内容", 1, fuzzy_threshold=90) is None
    log = [{"cand_id": "c1", "action": "respan", "evidence": new_ev}]
    assert replay(cands, log)["c1"].status == "pending"
    st = replay(cands, log + [{"cand_id": "c1", "action": "accept"}])["c1"]
    assert st.status == "edited" and st.evidence == new_ev


def test_review_order_interleaves_types(book):
    cands = [_cand(f"f{i}", "fact", [1], book) for i in range(3)] + [_cand("p0", "plot", [1, 2], book)]
    assert review_order(cands) == ["f0", "p0", "f1", "f2"]


def test_review_stats(book):
    cands = [_cand("c1", "fact", [1], book), _cand("c2", "fact", [2], book), _cand("c3", "fact", [3], book)]
    log = [
        {"cand_id": "c1", "action": "accept"},
        {"cand_id": "c2", "action": "edit", "question": "改过？"},
        {"cand_id": "c3", "action": "drop", "reason": "ambiguous"},
    ]
    s = review_stats(replay(cands, log))["by_type"]["fact"]
    assert (s["reviewed"], s["pass_rate"], s["edit_rate"], s["drop_reasons"]) == (3, 0.667, 0.5, {"ambiguous": 1})


def test_review_log_roundtrip(tmp_path):
    log = ReviewLog(tmp_path / "review.jsonl")
    log.append({"cand_id": "c1", "action": "accept"})
    log.append({"cand_id": "c1", "action": "undo"})
    assert [r["action"] for r in log.read()] == ["accept", "undo"] and "ts" in log.read()[0]


# ---------------------------------------------------------------- 人工题


def test_import_human(book, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    rows = [
        {
            "type": "fact",
            "question": "第二艘船载了几箱香料？",
            "answer": "六箱",
            "evidence": [{"quote": "码头上第2艘船载着6箱香料靠岸。"}],
        },
        {"type": "fact", "question": "歧义题？", "answer": "无", "evidence": [{"quote": "在港口边等待船队归来"}]},
        {
            "type": "plot",
            "question": "船队先后运来什么？",
            "answer": "香料",
            "evidence": [
                {"quote": "在港口边等待船队归来", "chapter_idx": 1},
                {"quote": "第6艘船载着18箱香料", "chapter_idx": 6},
            ],
        },
    ]
    (work / "human.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    stats = import_human(book, CFG, work)
    assert stats["ok"] == 2 and stats["drop_reasons"] == {"ambiguous_quote": 1}
    located = [json.loads(x) for x in (work / "located_human.jsonl").read_text(encoding="utf-8").splitlines()]
    assert located[0]["source"] == "human" and located[0]["evidence"][0]["chapter_idx"] == 2
    assert [e["chapter_idx"] for e in located[2]["evidence"]] == [1, 6]


# ---------------------------------------------------------------- 冻结


def test_select_prefers_humans_then_chapter_coverage(book):
    def st(cid, chs, source=None):
        return ReviewState.fresh(_cand(cid, "fact", chs, book, source))

    kept = [st("g1", [1]), st("g2", [1]), st("g3", [2]), st("h1", [1], "human")]
    chosen, reserve = select(kept, 2)
    assert [s.cand["cand_id"] for s in chosen] == ["h1", "g3"]  # g3 覆盖了新章节
    assert {s.cand["cand_id"] for s in reserve} == {"g1", "g2"}


def test_assign_split_is_deterministic_and_balanced(book):
    from novel_rag.eval.dataset import EvalItem, Evidence

    def item(i, source="generated"):
        return EvalItem(
            id=f"fact-{i:08x}",
            type="fact",
            question="?",
            answer="a",
            source=source,
            split="test",
            evidence=[Evidence(book_id="demo-1", chapter_idx=1, char_start=0, char_end=1)],
        )

    items = [item(i) for i in range(6)] + [item(100 + i, "human") for i in range(2)]
    assign_split(items)
    first = [i.split for i in items]
    assign_split(items)
    assert [i.split for i in items] == first
    assert sum(s == "dev" for s in first[:6]) == 3 and sorted(first[6:]) == ["dev", "test"]


def _write_located(work, name, cands):
    work.mkdir(parents=True, exist_ok=True)
    (work / name).write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cands), encoding="utf-8")


def test_freeze_end_to_end(book, tmp_path):
    work, out = tmp_path / "work", tmp_path / "datasets" / "demo_v0"
    cands = [_cand(f"r1-{i}", "fact", [i], book) for i in range(1, 5)] + [
        _cand("r2-p1", "plot", [2, 5], book),
        _cand("r2-m1", "multihop", [1, 6], book),
        _cand("r2-x", "relation", [3], book),
    ]
    _write_located(work, "located_r1.jsonl", cands[:4])
    _write_located(work, "located_r2.jsonl", cands[4:])
    (work / "import_r1.json").write_text(
        json.dumps({"candidates_file_sha256": "x", "questions": 4, "ok": 4}), encoding="utf-8"
    )
    log = ReviewLog(work / "review.jsonl")
    for c in cands[:-1]:
        log.append({"cand_id": c["cand_id"], "action": "accept"})

    with pytest.raises(FreezeError, match="未校验"):
        freeze(book, CFG, work, out, "demo_v0")
    log.append({"cand_id": "r2-x", "action": "drop", "reason": "too_easy"})
    manifest = freeze(book, CFG, work, out, "demo_v0")

    assert manifest.counts["by_type"] == {"fact": 2, "plot": 1, "multihop": 1}
    assert manifest.counts["reserve"] == 2
    assert manifest.books == {"demo-1": book.text_sha256}
    items = load_dataset(out, tmp_path / "books")
    plot = next(i for i in items if i.type == "plot")
    assert plot.max_reading_order == book.chapter(5).reading_order
    assert plot.id == item_id("plot", plot.question) and plot.origin == "r2-p1"
    assert (out / "reserve.jsonl").is_file()
    assert "码头" not in (out / "items.jsonl").read_text(encoding="utf-8")  # 不含原文


def test_freeze_rejects_copied_question(book, tmp_path):
    work, out = tmp_path / "work", tmp_path / "out"
    copied = "船员1在港口边等待船队归来，风从海面吹来，带着盐和鱼腥的味道，为什么？"
    _write_located(work, "located_r1.jsonl", [_cand("r1-1", "fact", [1], book, question=copied)])
    ReviewLog(work / "review.jsonl").append({"cand_id": "r1-1", "action": "accept"})
    with pytest.raises(FreezeError, match="原文连续"):
        freeze(book, CFG, work, out, "x")


def test_freeze_draft_includes_pending(book, tmp_path):
    work, out = tmp_path / "work", tmp_path / "draft"
    _write_located(work, "located_r1.jsonl", [_cand("r1-1", "fact", [1], book), _cand("r1-2", "fact", [2], book)])
    ReviewLog(work / "review.jsonl").append({"cand_id": "r1-2", "action": "drop", "reason": "too_easy"})
    manifest = freeze(book, CFG, work, out, "demo_draft", draft=True)
    assert manifest.counts["total"] == 1 and manifest.model_dump()["draft"] is True
