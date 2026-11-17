"""P6 版权自查脚本的核心：滑动窗口命中、延伸到最长重合、忽略空白；只用自拟文本。"""

from scripts.copyright_scan import scan_text


def index(text: str, n: int) -> set[int]:
    return {hash(text[i : i + n]) for i in range(len(text) - n + 1)}


SRC = "北岭的雪下了三天三夜，柳月在山口捡到一只白狐，给它取名雪团，从此形影不离。"


def test_hit_is_extended_to_longest_overlap_and_ignores_whitespace():
    n = 8
    doc = "前言。" + SRC[2:20].replace("，", "，\n  ") + "后记"
    hits = scan_text(doc, index(SRC, n), n, SRC)
    assert len(hits) == 1 and hits[0]["length"] == 18 and hits[0]["offset"] == 3


def test_short_overlap_below_threshold_is_not_reported():
    n = 10
    assert scan_text("引用：柳月在山口捡到", index(SRC, n), n, SRC) == []
    assert scan_text("完全无关的一段文字，没有任何重合的内容。", index(SRC, n), n, SRC) == []
