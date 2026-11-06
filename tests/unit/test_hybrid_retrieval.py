"""P2 检索链路：过滤条件在 Milvus / ES / 内存三处语义一致（含防剧透）、ES 索引同步与 BM25 查询（mock 客户端）、
RRF 融合、混合检索 + rerank、small-to-big 扩展与按预算装箱。"""

import itertools

import pytest

from novel_rag.config import RetrievalConfig
from novel_rag.llm.fake import FakeReranker
from novel_rag.retrieval.base import Hit, Retriever
from novel_rag.retrieval.es_store import BM25_SCHEMES, BM25Retriever, ESChunkStore, index_body, index_name
from novel_rag.retrieval.expand import ChunkIndex, expand_hits, pack_contexts
from novel_rag.retrieval.filters import SearchFilter
from novel_rag.retrieval.fusion import rrf_fuse
from novel_rag.retrieval.hybrid import FilterViolation, HybridRetriever

# ---------------------------------------------------------------- 过滤条件：同一份用例，三种实现


def _es_eval(clauses: list[dict], doc: dict) -> bool:
    """按 ES 语义解释 filter 子句（只覆盖 to_es_filter 会生成的 terms / range）。"""
    for c in clauses:
        if "terms" in c:
            ((f, vals),) = c["terms"].items()
            if doc.get(f) not in vals:
                return False
        elif "range" in c:
            ((f, cond),) = c["range"].items()
            if not doc.get(f, 0) <= cond["lte"]:
                return False
        else:
            raise AssertionError(f"未知子句 {c}")
    return True


def _milvus_eval(expr: str, doc: dict) -> bool:
    """把生成的 Milvus 表达式转成 Python 表达式求值（只覆盖 in / <= / and）。"""
    if not expr:
        return True
    return bool(eval(expr.replace(" and ", " and "), {}, dict(doc)))  # noqa: S307 - 表达式由被测代码生成


FILTERS = [
    SearchFilter(),
    SearchFilter(book_ids=("demo-1",)),
    SearchFilter(max_reading_order=3),
    SearchFilter(book_ids=("demo-1", "demo-2"), max_reading_order=0),
    SearchFilter(section_types=("narrative",), chunk_types=("text",), max_reading_order=5),
]
DOCS = [
    {"book_id": b, "reading_order": ro, "section_type": st, "chunk_type": ct}
    for b, ro, st, ct in itertools.product(
        ("demo-1", "demo-2", "demo-3"), (0, 3, 4, 9), ("narrative", "appendix"), ("text", "note")
    )
]


@pytest.mark.parametrize("flt", FILTERS)
def test_filter_semantics_agree_across_backends(flt):
    for doc in DOCS:
        expected = flt.accepts(doc)
        assert _es_eval(flt.to_es_filter(), doc) == expected, (flt, doc)
        assert _milvus_eval(flt.to_milvus_expr(), doc) == expected, (flt, doc)


def test_es_filter_clauses():
    assert SearchFilter().to_es_filter() == []
    f = SearchFilter(book_ids=("demo-1",), max_reading_order=12)
    assert f.to_es_filter() == [{"terms": {"book_id": ["demo-1"]}}, {"range": {"reading_order": {"lte": 12}}}]


# ---------------------------------------------------------------- ES 索引与 BM25（mock 客户端）


def _chunk(book, chap, seq, ro=None, text="t", start=None):
    start = seq * 10 if start is None else start
    return {
        "chunk_id": f"{book}:{chap:04d}:{seq:04d}",
        "book_id": book,
        "chapter_idx": chap,
        "seq": seq,
        "char_start": start,
        "char_end": start + len(text),
        "reading_order": chap if ro is None else ro,
        "section_type": "narrative",
        "chunk_type": "text",
        "pov": None,
        "chapter_title": f"第{chap}章",
        "text": text,
    }


class FakeES:
    """模仿 Elasticsearch 客户端的最小子集；BM25 打分用"查询字符在正文中出现的个数"。"""

    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.last_query = None
        outer = self

        class Indices:
            def exists(self, index):
                return True

            def create(self, index, settings, mappings):
                outer.created = (settings, mappings)

            def refresh(self, index): ...

        self.indices = Indices()

    def search(self, index, query, size, sort=None, source=True, search_after=None):
        self.last_query = query
        if "terms" in query:  # _scan
            ids = sorted(i for i, d in self.docs.items() if d["book_id"] in query["terms"]["book_id"])
            if search_after is not None:
                ids = [i for i in ids if i > search_after[0]]
            ids = ids[:size]
            return {"hits": {"hits": [{"_id": i, "_source": self.docs[i], "sort": [i]} for i in ids]}}
        q = query["bool"]["should"][0]["match"]["text"]["query"]
        docs = [d for d in self.docs.values() if _es_eval(query["bool"]["filter"], d)]
        scored = sorted(
            ((sum(d["text"].count(ch) for ch in q), d) for d in docs), key=lambda x: (-x[0], x[1]["chunk_id"])
        )
        return {
            "hits": {"hits": [{"_id": d["chunk_id"], "_score": s, "_source": d} for s, d in scored[:size] if s > 0]}
        }

    def count(self, index, query):
        if "match_all" in query:
            return {"count": len(self.docs)}
        return {"count": sum(_es_eval(query["bool"]["filter"], d) for d in self.docs.values())}


class FakeESStore(ESChunkStore):
    def _bulk(self, actions):
        n = 0
        for a in actions:
            if a["_op_type"] == "index":
                self.client.docs[a["_id"]] = a["_source"]
            else:
                self.client.docs.pop(a["_id"], None)
            n += 1
        return n, []


def test_index_body_uses_ik_and_bigram():
    body = index_body()
    text = body["mappings"]["properties"]["text"]
    assert (text["analyzer"], text["search_analyzer"]) == ("ik_max_word", "ik_smart")
    assert text["fields"]["bigram"]["analyzer"] == "cjk_bigram_std"
    assert "cjk_bigram" in body["settings"]["analysis"]["analyzer"]["cjk_bigram_std"]["filter"]
    assert index_name("abcd1234") == "chunks_abcd1234_es1"
    assert BM25_SCHEMES["ik_bigram"] == ("text", "text.bigram")


def test_es_sync_idempotent_and_scoped_to_books():
    es = FakeES()
    store = FakeESStore(es, "idx", RetrievalConfig())
    other = _chunk("demo-2", 1, 0)
    store.sync([other])
    a = [_chunk("demo-1", 1, 0), _chunk("demo-1", 1, 1), _chunk("demo-1", 2, 0)]
    assert store.sync(a) == {"indexed": 3, "deleted": 0, "count": 4}
    assert store.sync(a[:2]) == {"indexed": 2, "deleted": 1, "count": 3}  # 删除本卷多余，不动 demo-2
    assert store.sync(a[:2])["count"] == 3
    assert other["chunk_id"] in es.docs and es.docs[a[0]["chunk_id"]]["pov"] == ""
    assert store.count(SearchFilter(book_ids=("demo-1",))) == 2


def test_bm25_retriever_applies_spoiler_filter_and_fields():
    es = FakeES()
    store = FakeESStore(es, "idx", RetrievalConfig())
    store.sync([_chunk("demo-1", c, 0, text="琼恩在长城") for c in range(4)])
    r = BM25Retriever(store, "ik_bigram", SearchFilter(book_ids=("demo-1",)))
    hits = r.search("琼恩", 10, SearchFilter(book_ids=("demo-1",), max_reading_order=1))
    assert [h.fields["reading_order"] for h in hits] == [0, 1] and [h.rank for h in hits] == [1, 2]
    assert {"range": {"reading_order": {"lte": 1}}} in es.last_query["bool"]["filter"]
    assert [list(s["match"])[0] for s in es.last_query["bool"]["should"]] == ["text", "text.bigram"]
    assert r.describe()["fields"] == ["text", "text.bigram"] and "total" in r.last_latency_ms
    with pytest.raises(ValueError):
        BM25Retriever(store, "jieba")


# ---------------------------------------------------------------- RRF


def _hits(ids, fields=None):
    return [Hit(i, cid, 10.0 - i, dict(fields or {}, chunk_id=cid)) for i, cid in enumerate(ids, 1)]


def test_rrf_formula_dedup_and_debug():
    fused = rrf_fuse({"bm25": _hits(["a", "b", "c"]), "vector": _hits(["b", "d", "a"])}, k=60)
    score = {h.chunk_id: h.score for h in fused}
    assert score["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert score["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert [h.chunk_id for h in fused] == ["b", "a", "d", "c"]  # d 单路第 2 名 > c 单路第 3 名
    assert [h.rank for h in fused] == [1, 2, 3, 4]
    assert fused[0].fields["routes"] == {"bm25": {"rank": 2, "score": 8.0}, "vector": {"rank": 1, "score": 9.0}}
    assert fused[0].fields["chunk_id"] == "b" and fused[0].fields["rrf"] == fused[0].score


def test_rrf_ties_deterministic_and_top_k():
    # c 与 d 只在一路中各排第 3：同分，按最好单路排名、再按路的顺序打破平局
    fused = rrf_fuse({"x": _hits(["a", "b", "c"]), "y": _hits(["a", "b", "d"])}, k=1, top_k=3)
    assert [h.chunk_id for h in fused] == ["a", "b", "c"]
    dup = rrf_fuse({"x": _hits(["a", "a", "b"])}, k=60)  # 同一路内重复只算第一次
    assert dup[0].score == pytest.approx(1 / 61)
    with pytest.raises(ValueError):
        rrf_fuse({"x": []}, k=-1)


def test_rrf_k_changes_weight_of_top_ranks():
    routes = {"x": _hits(["a", "b", "c", "d"]), "y": _hits(["d", "c", "e", "f"])}
    # k 很小时单路第一名权重大，a、d 并列靠前；k 很大时接近"出现次数"，两路都有的 c、d 胜出
    small = [h.chunk_id for h in rrf_fuse(routes, k=0)]
    large = [h.chunk_id for h in rrf_fuse(routes, k=1000)]
    assert small[:2] == ["d", "a"] and set(large[:2]) == {"c", "d"}


# ---------------------------------------------------------------- 混合检索 + rerank


class ListRetriever(Retriever):
    """按给定顺序返回满足过滤条件的 chunk；leak=True 时故意忽略过滤条件（模拟某一路漏掉防剧透）。"""

    def __init__(self, name, chunks, leak=False):
        self.name = name
        self.chunks = chunks
        self.leak = leak
        self.calls: list[SearchFilter | None] = []

    def search(self, query, top_k, flt=None):
        self.calls.append(flt)
        ok = [c for c in self.chunks if self.leak or flt is None or flt.accepts(c)]
        return [Hit(i, c["chunk_id"], 1.0 / i, dict(c)) for i, c in enumerate(ok[:top_k], 1)]


def _corpus():
    texts = ["无关内容", "琼恩站在长城", "长城很冷", "琼恩", "其他"]
    return [_chunk("demo-1", i, 0, text=t) for i, t in enumerate(texts)]


def test_hybrid_rerank_reorders_head_and_keeps_tail():
    cs = _corpus()
    routes = {"bm25": ListRetriever("bm25", cs), "vector": ListRetriever("vector", list(reversed(cs)))}
    h = HybridRetriever(
        routes,
        route_top_k={"bm25": 5, "vector": 5},
        rrf_k=60,
        fusion_top_k=5,
        reranker=FakeReranker(),
        rerank_candidates=3,
        parallel=False,
    )
    fused = [x.chunk_id for x in rrf_fuse({n: r.search("q", 5) for n, r in routes.items()}, k=60)]
    hits = h.search("琼恩长城", 5)
    head = [x.chunk_id for x in hits[:3]]
    assert sorted(head) == sorted(fused[:3]) and [x.chunk_id for x in hits[3:]] == fused[3:]
    texts = {c["chunk_id"]: c["text"] for c in cs}
    scores = FakeReranker().score("琼恩长城", [texts[c] for c in head])
    assert scores == sorted(scores, reverse=True)
    assert all("rerank" in x.fields for x in hits[:3]) and all("rerank" not in x.fields for x in hits[3:])
    assert [x.rank for x in hits] == [1, 2, 3, 4, 5]
    assert {"routes", "fusion", "rerank", "total", "bm25.total", "vector.total"} <= set(h.last_latency_ms)
    assert h.describe()["name"] == "bm25+vector+rerank"


def test_hybrid_passes_spoiler_filter_to_every_route():
    cs = _corpus()
    routes = {"bm25": ListRetriever("bm25", cs), "vector": ListRetriever("vector", cs)}
    h = HybridRetriever(
        routes, route_top_k={"bm25": 5, "vector": 5}, rrf_k=60, fusion_top_k=5, reranker=FakeReranker(),
        rerank_candidates=5, base_filter=SearchFilter(book_ids=("demo-1",)),
    )  # fmt: skip
    flt = SearchFilter(book_ids=("demo-1",), max_reading_order=2)
    hits = h.search("琼恩", 10, flt)
    assert {x.fields["reading_order"] for x in hits} <= {0, 1, 2} and len(hits) == 3
    assert all(r.calls[-1] == flt for r in routes.values())
    h.search("琼恩", 10)
    assert all(r.calls[-1] == SearchFilter(book_ids=("demo-1",)) for r in routes.values())


def test_hybrid_detects_route_that_ignores_spoiler_filter():
    cs = _corpus()
    routes = {"bm25": ListRetriever("bm25", cs), "vector": ListRetriever("vector", cs, leak=True)}
    h = HybridRetriever(routes, route_top_k={"bm25": 5, "vector": 5}, rrf_k=60, fusion_top_k=5, parallel=False)
    with pytest.raises(FilterViolation):
        h.search("琼恩", 5, SearchFilter(max_reading_order=1))


def test_hybrid_without_reranker_returns_fusion_order():
    cs = _corpus()
    routes = {"bm25": ListRetriever("bm25", cs)}
    h = HybridRetriever(routes, route_top_k={"bm25": 5}, rrf_k=60, fusion_top_k=3)
    assert [x.chunk_id for x in h.search("q", 10)] == [c["chunk_id"] for c in cs[:3]]
    assert "rerank" not in h.last_latency_ms and h.describe()["reranker"] is None


# ---------------------------------------------------------------- small-to-big


def _chapter_chunks():
    """两章：第 1 章 4 块（块间有重叠，第 2、3 块之间隔一个换行），第 2 章 2 块。"""
    text1 = "AAAAABBBBBCCCCC\nDDDDDEEEEE"
    spans = [(0, 10), (5, 15), (16, 21), (19, 26)]  # 重叠 5 字；第 2、3 块之间是换行
    ch1 = [_chunk("demo-1", 1, i, ro=1, text=text1[s:e], start=s) for i, (s, e) in enumerate(spans)]
    ch2 = [_chunk("demo-1", 2, i, ro=2, text=t, start=100 + 5 * i) for i, t in enumerate(["xxxxx", "yyyyy"])]
    return text1, ch1, ch2


def _hit(c, rank):
    return Hit(rank, c["chunk_id"], 1.0, dict(c))


def test_expand_window_zero_is_identity():
    _, ch1, ch2 = _chapter_chunks()
    idx = ChunkIndex(ch1 + ch2)
    ctx = expand_hits([_hit(ch1[1], 1), _hit(ch1[0], 2)], idx, window=0)
    assert [c.chunk_ids for c in ctx] == [[ch1[1]["chunk_id"]], [ch1[0]["chunk_id"]]]
    assert ctx[0].text == ch1[1]["text"]


def test_expand_merges_neighbours_within_chapter_only():
    text1, ch1, ch2 = _chapter_chunks()
    idx = ChunkIndex(ch1 + ch2)
    hits = [_hit(ch2[0], 1), _hit(ch1[1], 2), _hit(ch1[3], 3)]
    ctx = expand_hits(hits, idx, window=1)
    assert len(ctx) == 2  # 第 1 章的两个命中扩展后相邻，合并成一段；不跨到第 2 章
    assert ctx[0].chapter_idx == 2 and ctx[0].seq_lo == 0 and ctx[0].seq_hi == 1 and ctx[0].text == "xxxxxyyyyy"
    c = ctx[1]
    assert (c.rank, c.seq_lo, c.seq_hi) == (2, 0, 3)
    assert c.text == text1 and (c.char_start, c.char_end) == (0, len(text1))  # 重叠只保留一次、换行补回
    assert set(c.hit_ids) == {ch1[1]["chunk_id"], ch1[3]["chunk_id"]}


def test_expand_respects_filter():
    _, ch1, _ = _chapter_chunks()
    blocked = dict(ch1[0], chunk_type="note")  # 类型不同的邻块不扩展
    idx = ChunkIndex([blocked, *ch1[1:]])
    ctx = expand_hits([_hit(ch1[1], 1)], idx, window=1)
    assert ctx[0].seq_lo == 1 and ctx[0].seq_hi == 2
    ctx = expand_hits([_hit(ch1[1], 1)], ChunkIndex(ch1), window=1, flt=SearchFilter(max_reading_order=0))
    assert ctx[0].chunk_ids == [ch1[1]["chunk_id"]]


def test_pack_contexts_greedy_skip():
    _, ch1, ch2 = _chapter_chunks()
    ctx = expand_hits([_hit(ch1[0], 1), _hit(ch2[0], 2), _hit(ch1[3], 3)], ChunkIndex(ch1 + ch2), window=0)
    sizes = {c.chunk_ids[0]: n for c, n in zip(ctx, (10, 30, 5), strict=True)}
    packed = pack_contexts(ctx, 16, lambda c: sizes[c.chunk_ids[0]])
    assert [c.rank for c in packed] == [1, 3]  # 第 2 段放不下就跳过，继续装第 3 段
    assert pack_contexts(ctx, 0, lambda c: 1) == []


# ---------------------------------------------------------------- ES / Milvus 一致性检查


def test_consistency_check_reports_missing_extra_and_field_mismatch():
    from novel_rag.retrieval.consistency import check_consistency

    chunks = [_chunk("demo-1", 1, i) for i in range(3)]
    es_docs = [dict(c, pov="") for c in chunks]
    mv_rows = [{k: v for k, v in c.items() if k not in ("char_start", "char_end")} for c in chunks]
    assert check_consistency(chunks, es_docs, mv_rows)["ok"]

    bad_mv = [mv_rows[0], dict(mv_rows[1], reading_order=99), dict(mv_rows[2], chunk_id="demo-1:0009:0000")]
    rep = check_consistency(chunks, es_docs + [es_docs[0]], bad_mv)
    assert not rep["ok"] and rep["es"]["duplicated"] == [chunks[0]["chunk_id"]]
    assert rep["milvus"]["mismatched"] == {chunks[1]["chunk_id"]: ["reading_order"]}
    assert rep["milvus"]["missing"] == [chunks[2]["chunk_id"]] and rep["milvus"]["extra"] == ["demo-1:0009:0000"]
