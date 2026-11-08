"""评测集构建：corpus 读取、证据定位、窗口采样与配对、预筛、任务导出 / 候选题导入、数据集校验、prompt 加载。"""

import json

import pytest

from novel_rag.config import EvalGenConfig
from novel_rag.corpus import BookText, CorpusError
from novel_rag.eval.dataset import (
    DatasetError,
    DatasetManifest,
    EvalItem,
    Evidence,
    load_dataset,
    save_dataset,
    validate_items,
)
from novel_rag.eval.qgen.locate import locate
from novel_rag.eval.qgen.prefilter import hard_reasons, jaccard, longest_common_substring
from novel_rag.eval.qgen.sampling import Window, pair_windows, person_key, sample_windows
from novel_rag.eval.qgen.tasks import export_round1, export_round2, import_round
from novel_rag.prompts import load_prompt
from tests.unit.book_factory import make_book, para

CFG = EvalGenConfig(window_min_chars=60, window_max_chars=120)


@pytest.fixture
def book(tmp_path):
    chapters = [
        (
            "阿岚",
            [
                "阿岚在码头上数着归来的船只，一共有七艘。",
                para("阿岚", 50),
                "琪娜·星川把一枚银色的贝壳交给阿岚。",
                para("码头", 40),
            ],
        ),
        ("琪娜", [para("琪娜", 45), "琪娜告诉阿岚，那枚贝壳来自北方的冰海。", para("海鸟", 55)]),
        ("附录", ["地名表 港口 冰海"]),
    ]
    make_book(tmp_path / "books", "demo-1", chapters, section_types={3: "appendix"})
    return BookText.load(tmp_path / "books" / "demo-1")


# ---------------------------------------------------------------- corpus


def test_corpus_requires_lock(tmp_path):
    make_book(tmp_path, "x-1", [("一", ["正文。"])], lock=False)
    with pytest.raises(CorpusError, match="定稿"):
        BookText.load(tmp_path / "x-1")
    assert not BookText.load(tmp_path / "x-1", require_lock=False).locked


def test_corpus_detects_changed_text(tmp_path):
    d = make_book(tmp_path, "x-1", [("一", ["正文。"])])
    (d / "text.txt").write_text("一\n被改过的正文。\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="不一致"):
        BookText.load(d)


def test_corpus_chapter_lookup_and_paragraphs(book):
    ch = book.chapter(1)
    assert book.text[ch.char_start : ch.title_end] == "阿岚"
    paras = book.paragraphs(ch)
    assert book.text[paras[0][0] : paras[0][1]] == "阿岚在码头上数着归来的船只，一共有七艘。"
    assert book.chapter_at(paras[0][0] + 3).chapter_idx == 1
    before, mid, after = book.context(paras[0][0], paras[0][0] + 2, width=5)
    assert mid == "阿岚" and before == "阿岚\n" and len(after) == 5


# ---------------------------------------------------------------- 定位


def test_locate_exact_loose_fuzzy_and_fail():
    text = "开头。琪娜·星川把一枚银色的贝壳交给阿岚，“收好它。”她说。结尾。"
    lo, hi = 0, len(text)
    exact = locate(text, "一枚银色的贝壳", lo, hi, fuzzy_threshold=90)
    assert exact.method == "exact" and text[exact.start : exact.end] == "一枚银色的贝壳"
    loose = locate(text, '琪娜•星川把一枚银色的贝壳交给阿岚,"收好它."', lo, hi, fuzzy_threshold=90)
    assert loose.method == "loose" and text[loose.start : loose.end] == "琪娜·星川把一枚银色的贝壳交给阿岚，“收好它。”"
    fuzzy = locate(text, "琪娜·星川把一枚银白的贝壳交给阿岚", lo, hi, fuzzy_threshold=85)
    assert fuzzy.method == "fuzzy" and "贝壳交给阿岚" in text[fuzzy.start : fuzzy.end]
    assert locate(text, "完全不存在的一句话内容", lo, hi, fuzzy_threshold=90) is None


def test_locate_respects_window_bounds():
    text = "甲乙丙丁。甲乙丙丁。"
    loc = locate(text, "甲乙丙丁", 3, len(text), fuzzy_threshold=90)
    assert loc.start == 5


# ---------------------------------------------------------------- 采样与配对


def test_sample_windows_one_per_narrative_chapter_and_deterministic(book):
    a, b = sample_windows(book, CFG), sample_windows(book, CFG)
    assert a == b
    assert [w.chapter_idx for w in a] == [1, 2]  # 附录默认不参与
    for w in a:
        ch = book.chapter(w.chapter_idx)
        assert ch.body_start <= w.char_start < w.char_end <= ch.char_end
        assert w.char_end - w.char_start >= CFG.window_min_chars
        assert book.text[w.char_start - 1] == "\n"  # 从段落边界开始
    assert [w.chapter_idx for w in sample_windows(book, CFG.model_copy(update={"include_appendix": True}))] == [1, 2, 3]


def test_person_key():
    assert person_key("琪娜•星川") == person_key("琪娜") == "琪娜"


def test_pair_windows_rules():
    chapter_of = {"w0": 1, "w1": 2, "w2": 8, "w3": 9, "w4": 15}
    windows = [Window(w, ch, 0, 1) for w, ch in chapter_of.items()]
    persons = {
        "w0": ["阿岚", "琪娜·星川"],
        "w1": ["阿岚", "琪娜"],
        "w2": ["阿岚"],
        "w3": ["老船长"],
        "w4": ["阿岚", "老船长"],
    }
    cfg = EvalGenConfig(
        combo_counts={"relation": 5, "multihop": 5, "plot": 5}, plot_min_chapter_gap=5, max_window_uses=9
    )
    tasks = pair_windows(windows, persons, cfg)
    rel = [t for t in tasks if t.type == "relation"]
    assert [t.window_ids for t in rel] == [("w0", "w1")] and set(rel[0].shared) == {"阿岚", "琪娜"}
    assert all(len(set(t.window_ids)) == len(t.window_ids) for t in tasks)
    plot = [t for t in tasks if t.type == "plot"]
    assert plot
    for t in plot:
        chs = [chapter_of[w] for w in t.window_ids]
        assert all(b - a >= 5 for a, b in zip(chs, chs[1:], strict=False))
    assert pair_windows(windows, persons, cfg) == tasks  # 确定性


# ---------------------------------------------------------------- 预筛


def test_prefilter_helpers():
    assert longest_common_substring("银色的贝壳交给谁", "把一枚银色的贝壳交给阿岚") == 7
    assert jaccard("阿岚数了几艘船", "阿岚数了几艘船") == 1.0
    base = {"question": "阿岚收到了什么？", "answer": "贝壳", "type": "fact", "evidence": []}
    assert hard_reasons(base) == []
    assert hard_reasons({**base, "question": "阿岚收到的贝壳是谁给的？"}) == ["answer_in_question"]
    multi = {
        **base,
        "type": "multihop",
        "evidence": [{"window_id": "w1", "chapter_idx": 1}, {"window_id": "w1", "chapter_idx": 1}],
    }
    assert hard_reasons(multi) == ["single_window"]
    same_ch = {**multi, "evidence": [{"window_id": "w1", "chapter_idx": 1}, {"window_id": "w2", "chapter_idx": 1}]}
    assert hard_reasons(same_ch) == ["single_chapter"]
    # 跨卷：两卷的章号相同也算两个章节
    cross = {
        **multi,
        "evidence": [
            {"window_id": "v1w1", "book_id": "b1", "chapter_idx": 3},
            {"window_id": "v2w1", "book_id": "b2", "chapter_idx": 3},
        ],
    }
    assert hard_reasons(cross) == []


# ---------------------------------------------------------------- 导出与导入


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_export_import_roundtrip(book, tmp_path):
    work = tmp_path / "work"
    tasks_path = export_round1(book, CFG, work)
    tasks = [json.loads(x) for x in tasks_path.read_text(encoding="utf-8").splitlines()]
    assert [t["task_id"] for t in tasks] == ["r1-0001", "r1-0002"]
    w1, w2 = tasks[0]["windows"][0], tasks[1]["windows"][0]
    assert w1["text"] == book.text[w1["char_start"] : w1["char_end"]]

    # 让测试中的证据一定落在窗口内
    q1 = book.text[w1["char_start"] : w1["char_start"] + 12]
    q2 = book.text[w2["char_start"] : w2["char_start"] + 12]
    _write(
        work / "candidates_r1.jsonl",
        [
            {"task_id": "r1-0001", "persons": ["阿岚", "琪娜·星川"],
             "questions": [{"type": "fact", "question": "码头上的人在等什么？", "answer": "船队",
                            "evidence": [{"window_id": w1["window_id"], "quote": q1}]},
                           {"type": "fact", "question": "无法定位的题？", "answer": "无",
                            "evidence": [{"window_id": w1["window_id"], "quote": "窗口里根本没有这句话的内容"}]}]},
            {"task_id": "r1-0002", "persons": ["琪娜", "阿岚"],
             "questions": [{"type": "fact", "question": "海边的风带着什么味道？", "answer": "盐和鱼腥",
                            "evidence": [{"window_id": w2["window_id"], "quote": q2}]}]},
            {"task_id": "r1-9999", "questions": []},
        ],
    )  # fmt: skip
    stats = import_round(book, CFG, work, 1)
    assert stats["questions"] == 3 and stats["ok"] == 2
    assert stats["drop_reasons"] == {"locate_failed": 1}
    assert stats["issues"] and "r1-9999" in stats["issues"][0]
    located = [json.loads(x) for x in (work / "located_r1.jsonl").read_text(encoding="utf-8").splitlines()]
    ev = located[0]["evidence"][0]
    assert book.text[ev["char_start"] : ev["char_end"]] == q1 and ev["method"] == "exact"
    assert "quote" not in json.dumps(located, ensure_ascii=False)  # 导入结果不含原文片段

    combo_cfg = CFG.model_copy(update={"combo_counts": {"relation": 3, "multihop": 3, "plot": 0}})
    r2 = [json.loads(x) for x in export_round2(book, combo_cfg, work).read_text(encoding="utf-8").splitlines()]
    assert {t["type"] for t in r2} == {"relation", "multihop"}
    assert all(len(t["windows"]) == 2 for t in r2)


# ---------------------------------------------------------------- 数据集


def _item(**kw):
    base = dict(
        id="fact-0001", type="fact", question="阿岚数了几艘船？", answer="七艘", source="generated", split="dev",
        evidence=[Evidence(book_id="demo-1", chapter_idx=1, char_start=3, char_end=10)], max_reading_order=0,
    )  # fmt: skip
    return EvalItem(**{**base, **kw})


def test_validate_items(book):
    books = {"demo-1": book}
    assert validate_items([_item()], books, answer_max_chars=40, copyright_max_substring=20) == []
    bad_span = _item(evidence=[Evidence(book_id="demo-1", chapter_idx=1, char_start=3, char_end=9999)])
    copied = _item(id="fact-0002", question="阿岚在码头上数着归来的船只，一共有七艘。这是真的吗？")
    errors = validate_items([bad_span, copied], books, answer_max_chars=40, copyright_max_substring=20)
    assert any("不在第 1 章内" in e for e in errors) and any("原文连续" in e for e in errors)
    late = _item(evidence=[Evidence(book_id="demo-1", chapter_idx=2, char_start=book.chapter(2).body_start,
                                    char_end=book.chapter(2).body_start + 5)], max_reading_order=0)  # fmt: skip
    assert any("max_reading_order" in e for e in validate_items([late], books, answer_max_chars=40,
                                                                 copyright_max_substring=20))  # fmt: skip


def test_dataset_roundtrip_and_hash_guard(book, tmp_path):
    d = tmp_path / "ds"
    save_dataset(d, [_item(), _item(id="fact-0002", split="test")],
                 DatasetManifest(name="demo_v0", books={"demo-1": book.text_sha256}))  # fmt: skip
    books_dir = tmp_path / "books"
    assert [i.id for i in load_dataset(d, books_dir, split="test")] == ["fact-0002"]
    save_dataset(d, [_item()], DatasetManifest(name="demo_v0", books={"demo-1": "0" * 64}))
    with pytest.raises(DatasetError):
        load_dataset(d, books_dir)


# ---------------------------------------------------------------- prompt


def test_load_prompt_picks_highest_version(tmp_path):
    (tmp_path / "demo.v1.md").write_text("一", encoding="utf-8")
    (tmp_path / "demo.v2.md").write_text("二", encoding="utf-8")
    assert load_prompt("demo", directory=tmp_path).version_id == "demo.v2"
    assert load_prompt("demo", 1, directory=tmp_path).text == "一"
    with pytest.raises(FileNotFoundError):
        load_prompt("missing", directory=tmp_path)
    assert load_prompt("qgen_guidelines").version >= 1
