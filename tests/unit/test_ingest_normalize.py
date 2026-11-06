"""规范化规则：每条规则的正例与反例，以及按分册判定半角问号的用途。样例均为自拟。"""

import pytest

from novel_rag.ingestion.ir import NOTE_MARK
from novel_rag.ingestion.normalize import NormalizeStats, detect_qmark_mode, normalize_text


def norm(text: str, mode: str = "repair") -> tuple[str, dict[str, int]]:
    stats = NormalizeStats()
    return normalize_text(text, stats, qmark_mode=mode), stats.hits


@pytest.mark.parametrize("variant", ["•", "‧", "・", "･", "∙", "⋅"])
def test_interpunct_variants_unified(variant):
    out, hits = norm(f"阿岚{variant}星川")
    assert out == "阿岚·星川" and hits["interpunct"] == 1


def test_isolated_bullet_not_changed_but_counted():
    out, hits = norm("• 第一项")
    assert out == "• 第一项" and hits["interpunct_isolated"] == 1


def test_qmark_repair_only_between_hanzi():
    assert norm("曼达?雷恩")[0] == "曼达·雷恩"
    assert norm("真的吗?”")[0] == "真的吗?”"  # repair 模式下不碰问号位置的 '?'
    assert norm("A?B")[0] == "A?B"
    assert norm("你去吗？我不去")[0] == "你去吗？我不去"  # 全角问号永远不动


def test_qmark_fullwidth_mode():
    out, hits = norm("你去吗?我不去。“真的?”", mode="fullwidth")
    assert out == "你去吗？我不去。“真的？”" and hits["qmark_fullwidth"] == 2
    assert norm("version 2?", mode="fullwidth")[0] == "version 2?"


def test_detect_qmark_mode():
    assert detect_qmark_mode(["甲?乙", "丙?丁"], 0.2)[0] == "repair"
    assert detect_qmark_mode(["真的吗?”", "好吗?”", "甲?乙"], 0.2)[0] == "fullwidth"
    assert detect_qmark_mode(["没有问号"], 0.2) == ("none", 0, 0)


def test_whitespace_rules():
    out, hits = norm("　　他走了。\r\n  然后 停下。 ")
    assert out == "他走了。 然后 停下。"
    assert hits["edge_space"] >= 1 and hits["nbsp"] == 1 and hits["whitespace_run"] >= 1


def test_space_between_hanzi_kept_and_counted():
    """附录表格用空格分隔列，汉字之间的空格只计数不删除。"""
    out, hits = norm("年代 国王 备注")
    assert out == "年代 国王 备注" and hits["cjk_space"] == 2


def test_zero_width_removed():
    assert norm("阿​岚﻿")[0] == "阿岚"


def test_note_placeholder_survives():
    out, _ = norm(f"守望者{NOTE_MARK}站在塔上{NOTE_MARK}")
    assert out.count(NOTE_MARK) == 2


def test_no_nfkc():
    assert norm("（全角括号）！，")[0] == "（全角括号）！，"


def test_samples_record_context():
    stats = NormalizeStats(sample_size=1)
    normalize_text("前面的文字甲•乙后面的文字", stats, "loc")
    s = stats.samples["interpunct"][0]
    assert s["match"] == "•" and s["before"].endswith("甲") and s["after"].startswith("乙") and s["locator"] == "loc"
