"""P4 问答评测：规则判分、弃答、作答环节（预算装箱、引用校验）、失败自动分类。"""

import json

from novel_rag.eval.dataset import EvalItem, Evidence
from novel_rag.eval.metrics import Span
from novel_rag.llm.fake import Say, ScriptedChatModel
from novel_rag.qa.analysis import classify, looping
from novel_rag.qa.answer import EvidenceBlock, answer_question, chunks_context, is_abstain, notes_context
from novel_rag.qa.scoring import normalize, rule_match, score_answer

REL = {"min_ratio": 0.5, "min_chars": 50}


def test_normalize_unifies_middots_and_punctuation():
    assert normalize("琼恩・雪诺！") == normalize("琼恩·雪诺") == normalize("琼恩 雪诺") == "琼恩雪诺"
    assert normalize("《狗熊与美少女》") == "狗熊与美少女"


def test_rule_match_uses_aliases_and_min_length():
    assert rule_match("是莱安娜的死。", "因为莱安娜之死", ["莱安娜的死"])
    assert rule_match("琼恩•雪诺", "琼恩·雪诺", [])
    assert not rule_match("罗柏", "罗柏·史塔克", [])  # 只答了一部分，交给 judge
    assert not rule_match("好", "好", [], min_chars=2)  # 单字不做包含匹配


def test_score_answer_abstain_rule_and_judge():
    assert score_answer(question="q", gold="雪团", aliases=[], answer="原文未提及", abstained=True, judge=None)[
        "label"] == "abstain"  # fmt: skip
    assert score_answer(question="q", gold="雪团", aliases=[], answer="叫雪团", abstained=False, judge=None) == {
        "label": "correct", "by": "rule"}  # fmt: skip
    judge = ScriptedChatModel([Say(json.dumps({"verdict": "partial", "reason": "只答了一半"}, ensure_ascii=False))])
    out = score_answer(question="q", gold="甲和乙", aliases=[], answer="甲", abstained=False, judge=judge)
    assert out["label"] == "partial" and out["by"] == "judge" and out["reason"] == "只答了一半"


def test_is_abstain():
    assert is_abstain("原文未提及。", "原文未提及") and is_abstain("", "原文未提及")
    assert not is_abstain("雪团", "原文未提及")


def test_context_packing_and_citation_validation():
    blocks = [EvidenceBlock(f"b:0000:000{i}", "卷 · 章", "字" * n) for i, n in enumerate((30, 50, 15))]
    ctx = chunks_context(blocks, 50, len)
    assert [b.chunk_id for b in ctx.blocks] == ["b:0000:0000", "b:0000:0002"] and ctx.dropped == ["b:0000:0001"]
    nctx = notes_context("1. 笔记 [b:0000:0001]", blocks, 60, len)
    assert nctx.notes_tokens == len("1. 笔记 [b:0000:0001]") and "证据笔记" in nctx.text
    chat = ScriptedChatModel([Say(json.dumps({"answer": "雪团", "citations": ["b:0000:0000", "x:1:2"]}))])
    out = answer_question("q", ctx, chat, allowed_citations={b.chunk_id for b in ctx.blocks}, abstain="原文未提及")
    assert out["citations"] == ["b:0000:0000"] and out["invalid_citations"] == ["x:1:2"]
    assert "原文证据" in chat.history[0]["messages"][-1]["content"]


def test_closed_book_uses_closed_system_prompt():
    chat = ScriptedChatModel([Say(json.dumps({"answer": "原文未提及"}, ensure_ascii=False))])
    out = answer_question("q", None, chat, allowed_citations=set(), abstain="原文未提及")
    assert out["abstained"] and "没有提供任何原文" in chat.history[0]["messages"][0]["content"]


def _item(type_="multihop"):
    return EvalItem(id="q", type=type_, question="q", answer="a", source="generated", split="dev",
                    evidence=[Evidence(book_id="b", chapter_idx=0, char_start=0, char_end=100),
                              Evidence(book_id="b", chapter_idx=1, char_start=1000, char_end=1100)])  # fmt: skip


SPANS = {"c1": Span("b", 0, 120), "c2": Span("b", 990, 1110), "c3": Span("b", 5000, 5100)}


def _row(label, shown, notes, term="finish", trace=None):
    return {"label": label, "type": "multihop", "termination": term, "citations": notes,
            "evidence_sets": {"shown": shown, "notes": notes}, "agent": {"trace": trace or []}}  # fmt: skip


def test_failure_stages():
    it = _item()
    assert classify(_row("wrong", ["c2"], []), it, SPANS, REL)["stage"] == "bridge_missing"
    assert classify(_row("wrong", ["c3"], [], term="limit_rounds"), _item("plot"), SPANS, REL) == {
        "stage": "retrieval_miss", "flags": ["hit_limit"]}  # fmt: skip
    c = classify(_row("wrong", ["c1", "c2"], ["c1"]), it, SPANS, REL)
    assert c["stage"] == "found_not_noted" and "premature_finish" in c["flags"]
    assert classify(_row("wrong", ["c1"], ["c1"]), it, SPANS, REL)["stage"] == "retrieval_partial"
    assert classify(_row("partial", ["c1", "c2"], ["c1", "c2"]), it, SPANS, REL)["stage"] == "noted_but_wrong"
    assert classify(_row("correct", ["c1", "c3"], ["c3"]), it, SPANS, REL)["flags"] == ["correct_bad_citation"]


def test_looping_detection():
    t = [{"step": "tool", "tool": "search_text", "args": {"query": "艾德 劳勃 和解"}},
         {"step": "tool", "tool": "search_text", "args": {"query": "艾德劳勃和解"}}]  # fmt: skip
    assert looping(t)
    t2 = [{"step": "tool", "tool": "search_text", "args": {"query": "艾德 劳勃 和解"}},
          {"step": "tool", "tool": "search_text", "args": {"query": "莱安娜之死 真相"}}]  # fmt: skip
    assert not looping(t2)
