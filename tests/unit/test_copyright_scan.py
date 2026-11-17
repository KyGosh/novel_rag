"""P6 版权自查与结果入库前的处理：滑动窗口命中、延伸到最长重合、忽略空白、只替换重合片段；只用自拟文本。"""

from novel_rag.eval.redact import MARK, RedactIndex
from scripts.copyright_scan import scan_text

SRC = "北岭的雪下了三天三夜，柳月在山口捡到一只白狐，给它取名雪团，从此形影不离。"


def index(text: str, n: int) -> set[int]:
    return {hash(text[i : i + n]) for i in range(len(text) - n + 1)}


def test_hit_is_extended_to_longest_overlap_and_ignores_whitespace():
    n = 8
    doc = "前言。" + SRC[2:20].replace("，", "，\n  ") + "后记"
    hits = scan_text(doc, index(SRC, n), n, SRC)
    assert len(hits) == 1 and hits[0]["length"] == 18 and hits[0]["offset"] == 3


def test_short_overlap_below_threshold_is_not_reported():
    n = 10
    assert scan_text("引用：柳月在山口捡到", index(SRC, n), n, SRC) == []
    assert scan_text("完全无关的一段文字，没有任何重合的内容。", index(SRC, n), n, SRC) == []


def test_redact_replaces_only_the_overlapping_span_in_nested_strings():
    ri = RedactIndex([SRC], n=8)
    q = "检索：" + SRC[11:23] + " 白狐"  # 12 字重合
    obj = {"id": "q1", "agent": {"trace": [{"calls": [{"arguments": q}]}]}, "n": 3, "short": SRC[:5]}
    out = ri.redact_obj(obj)
    assert out["agent"]["trace"][0]["calls"][0]["arguments"] == "检索：" + MARK + " 白狐"
    assert out["id"] == "q1" and out["n"] == 3 and out["short"] == SRC[:5]  # 不足 N 字的不动
    assert ri.count == 1
    assert ri.redact_obj({"a": "毫不相干的一句很长很长的话，足够长了"}) == {"a": "毫不相干的一句很长很长的话，足够长了"}


def test_redact_keeps_whitespace_outside_the_span():
    ri = RedactIndex([SRC], n=6)
    s = "前 缀 " + SRC[0:4] + "\n" + SRC[4:10] + " 后缀"
    assert ri.redact(s) == "前 缀 " + MARK + " 后缀"
