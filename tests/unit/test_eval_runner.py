"""评测指标与运行器：相关性判定、Recall / Hit / MRR、按题型汇总、防剧透开关、实验记录不含原文；
最简生成的 prompt 组装。
"""

import json

from novel_rag.config import PathsConfig, Settings
from novel_rag.eval.dataset import EvalItem, Evidence
from novel_rag.eval.metrics import Span, aggregate, is_relevant, score_ranking
from novel_rag.eval.runner import run_retrieval_eval, write_run
from novel_rag.generation.simple import answer, build_messages
from novel_rag.llm.fake import FakeChatModel
from novel_rag.prompts import load_prompt
from novel_rag.retrieval.base import Hit, Retriever

KS = (5, 10, 20)


def ev(s, e, book="b"):
    return Evidence(book_id=book, chapter_idx=1, char_start=s, char_end=e)


def test_relevance_threshold():
    short = ev(100, 120)  # 20 字：需要重叠 ≥ 10
    assert is_relevant(Span("b", 110, 500), short, min_ratio=0.5, min_chars=50)
    assert not is_relevant(Span("b", 111, 500), short, min_ratio=0.5, min_chars=50)
    long = ev(0, 400)  # 400 字：需要重叠 ≥ 50（上限）
    assert is_relevant(Span("b", 350, 900), long, min_ratio=0.5, min_chars=50)
    assert not is_relevant(Span("b", 351, 900), long, min_ratio=0.5, min_chars=50)
    assert not is_relevant(Span("other", 0, 400), long, min_ratio=0.5, min_chars=50)


def test_score_ranking_multi_evidence():
    ranked = [Span("b", 0, 10)] * 2 + [Span("b", 100, 200)] + [Span("b", 0, 10)] * 8 + [Span("b", 1000, 1100)]
    m = score_ranking(ranked, [ev(120, 140), ev(1010, 1030)], KS, min_ratio=0.5, min_chars=50)
    assert m["evidence_ranks"] == [3, 12]
    assert (m["recall@5"], m["recall@10"], m["recall@20"]) == (0.5, 0.5, 1.0)
    assert (m["hit@5"], m["mrr"]) == (1.0, 1 / 3)
    miss = score_ranking([Span("b", 0, 10)], [ev(500, 520)], KS, min_ratio=0.5, min_chars=50)
    assert miss["mrr"] == 0.0 and miss["hit@20"] == 0.0


def test_aggregate_groups():
    rows = [
        {"type": "fact", "source": "generated", "metrics": {"recall@5": 1.0, "hit@5": 1.0, "mrr": 1.0}},
        {"type": "fact", "source": "human", "metrics": {"recall@5": 0.0, "hit@5": 0.0, "mrr": 0.0}},
        {"type": "plot", "source": "generated", "metrics": {"recall@5": 0.5, "hit@5": 1.0, "mrr": 0.5}},
    ]
    agg = aggregate(rows, (5,))
    assert agg["by_type"]["fact"] == {"n": 2, "recall@5": 0.5, "hit@5": 0.5, "mrr": 0.5}
    assert agg["by_source"]["human"]["n"] == 1 and agg["overall"]["n"] == 3


class FakeRetriever(Retriever):
    """按 chunk 在 corpus 中的顺序返回满足过滤条件的结果。"""

    name = "fake"

    def __init__(self, corpus):
        self.corpus = corpus
        self.filters = []

    def search(self, query, top_k, flt=None):
        self.filters.append(flt)
        ok = [c for c in self.corpus if flt is None or flt.accepts(c)]
        return [Hit(i, c["chunk_id"], 1.0 - i / 100, c) for i, c in enumerate(ok[:top_k], 1)]


def _item(i, qtype, s, e, mro):
    return EvalItem(
        id=f"{qtype}-{i}",
        type=qtype,
        question="问题？",
        answer="答案",
        source="generated",
        split="dev",
        evidence=[ev(s, e)],
        max_reading_order=mro,
    )


def test_runner_spoiler_switch_and_record(tmp_path):
    corpus = [
        {"chunk_id": "b:0001:0000", "book_id": "b", "reading_order": 0, "text": "原文甲"},
        {"chunk_id": "b:0002:0000", "book_id": "b", "reading_order": 5, "text": "原文乙"},
    ]
    spans = {"b:0001:0000": Span("b", 0, 100), "b:0002:0000": Span("b", 100, 200)}
    items = [_item(1, "fact", 10, 30, 0), _item(2, "plot", 120, 140, 5)]
    settings = Settings(_env_file=None, paths=PathsConfig(results_dir=tmp_path / "results"))
    retr = FakeRetriever(corpus)

    rows, metrics = run_retrieval_eval(items, retr, spans, settings)
    assert all(f is not None and f.max_reading_order is None for f in retr.filters)  # 默认不加防剧透过滤
    assert metrics["overall"]["hit@5"] == 1.0 and rows[1]["evidence_ranks"] == [2]

    retr.filters.clear()
    rows, _ = run_retrieval_eval(items, retr, spans, settings, spoiler_filter=True)
    assert [f.max_reading_order for f in retr.filters] == [0, 5]
    assert [h["chunk_id"] for h in rows[0]["hits"]] == ["b:0001:0000"]  # 晚于进度的 chunk 被过滤

    run = {
        "name": "t",
        "created": "now",
        "git": {"commit": "abc", "dirty": False},
        "dataset": {"name": "d", "split": "dev"},
        "retriever": {"name": "fake"},
        "chunk_version": "v",
        "spoiler_filter": True,
        "relevance": {"min_ratio": 0.5, "min_chars": 50},
    }
    out = write_run(settings.paths.results_dir, "t", run, metrics, rows, KS)
    assert {p.name for p in out.iterdir()} == {"run.json", "metrics.json", "per_question.jsonl", "summary.md"}
    dump = "".join(p.read_text(encoding="utf-8") for p in out.iterdir())
    assert "原文甲" not in dump and "原文乙" not in dump
    assert "跨章节情节" in (out / "summary.md").read_text(encoding="utf-8")
    assert json.loads((out / "metrics.json").read_text(encoding="utf-8"))["overall"]["n"] == 2


def test_generation_messages_and_answer():
    hits = [Hit(1, "b:0001:0000", 0.9, {"book_id": "b", "chapter_title": "第一章", "text": "片段内容"})]
    prompt = load_prompt("answer")
    msgs = build_messages("谁来了？", hits, prompt, {"b": "某书"})
    assert msgs[0]["role"] == "system" and "片段" in msgs[0]["content"]
    assert "谁来了？" in msgs[1]["content"] and "[1] 《某书 第一章》 b:0001:0000\n片段内容" in msgs[1]["content"]
    res = answer("谁来了？", hits, FakeChatModel(lambda m: "某人 [1]"), book_titles={"b": "某书"})
    assert res["answer"] == "某人 [1]" and res["prompt_version"] == "answer.v1" and res["chunk_ids"] == ["b:0001:0000"]


# ---------------------------------------------------------------- P2：预算覆盖率、置信区间、分阶段延迟


def test_coverage_metric():
    from novel_rag.eval.metrics import coverage

    evs = [ev(10, 30), ev(500, 520)]
    assert coverage([Span("b", 0, 100)], evs, min_ratio=0.5, min_chars=50) == 0.5
    assert coverage([Span("b", 0, 100), Span("b", 480, 600)], evs, min_ratio=0.5, min_chars=50) == 1.0
    assert coverage([], evs, min_ratio=0.5, min_chars=50) == 0.0


def test_bootstrap_and_paired_diff():
    from novel_rag.eval.stats import bootstrap_ci, paired_diff

    lo, hi = bootstrap_ci([1.0] * 10, n_samples=200, seed=1, level=0.95)
    assert lo == hi == 1.0
    lo, hi = bootstrap_ci([0.0, 1.0] * 20, n_samples=2000, seed=1, level=0.95)
    assert lo < 0.5 < hi and hi - lo < 0.5
    assert bootstrap_ci([0.0, 1.0] * 20, n_samples=500, seed=7, level=0.9) == bootstrap_ci(
        [0.0, 1.0] * 20, n_samples=500, seed=7, level=0.9
    )  # 固定种子可复现
    d = paired_diff([0.0] * 20, [1.0] * 10 + [0.0] * 10, n_samples=1000, seed=1, level=0.95)
    assert d["diff"] == 0.5 and 0 < d["lo"] <= 0.5 <= d["hi"] and d["p_le0"] == 0.0
    import pytest

    with pytest.raises(ValueError):
        paired_diff([0.0], [0.0, 1.0], n_samples=10, seed=1, level=0.95)


def test_runner_budget_coverage_ci_and_stage_latency():
    from novel_rag.retrieval.expand import ChunkIndex, expand_hits

    corpus = [
        {"chunk_id": f"b:0001:{i:04d}", "book_id": "b", "chapter_idx": 1, "seq": i, "reading_order": 1,
         "chunk_type": "text", "char_start": i * 100, "char_end": i * 100 + 100, "text": "x" * 100}
        for i in range(3)
    ]  # fmt: skip
    spans = {c["chunk_id"]: Span("b", c["char_start"], c["char_end"]) for c in corpus}
    # 证据在第 2 块（seq 1），检索只命中第 1 块：不扩展时预算内覆盖不到，扩展 ±1 后能覆盖
    items = [_item(1, "multihop", 120, 160, 1)]

    class StagedRetriever(FakeRetriever):
        def search(self, query, top_k, flt=None):
            self.last_latency_ms = {"bm25.total": 3.0, "rerank": 10.0, "total": 15.0}
            return super().search(query, top_k, flt)[:1]

    settings = Settings(_env_file=None)
    idx = ChunkIndex(corpus)
    for window, expected in ((0, 0.0), (1, 1.0)):
        rows, metrics = run_retrieval_eval(
            items, StagedRetriever(corpus), spans, settings,
            expander=lambda hits, flt, w=window: expand_hits(hits, idx, w, flt),
            count_tokens=lambda c: len(c.text),
        )  # fmt: skip
        assert rows[0]["metrics"]["cov@4000"] == expected
        assert rows[0]["contexts"][0]["budget"] == 2000
    assert metrics["latency_ms"] == {"p50": 15.0, "p95": 15.0}
    assert metrics["latency_stages_ms"]["rerank"] == {"p50": 10.0, "p95": 10.0}
    assert metrics["ci"]["overall"]["cov@4000"] == [1.0, 1.0] and "type:multihop" in metrics["ci"]
