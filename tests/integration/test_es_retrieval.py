"""集成测试：真实 ES（IK + bigram）上的 chunk 同步、BM25 检索与防剧透过滤；混合检索在真实 ES + Milvus 上的防剧透。

运行：uv run pytest -m integration
"""

import time

import pytest

from novel_rag.config import get_settings
from novel_rag.llm.fake import FakeEmbedder, FakeReranker
from novel_rag.retrieval.es_store import BM25Retriever, ESChunkStore
from novel_rag.retrieval.filters import SearchFilter
from novel_rag.retrieval.hybrid import HybridRetriever
from novel_rag.retrieval.milvus_store import MilvusChunkStore, VectorRetriever

pytestmark = pytest.mark.integration

TEXTS = [
    "守夜人在长城上巡逻。",
    "琼恩·雪诺加入了守夜人。",
    "临冬城下起了雪。",
    "雪诺站在长城顶上眺望。",
    "龙石岛的彗星。",
]


def _chunk(i, book="it-1"):
    return {
        "chunk_id": f"{book}:{i:04d}:0000",
        "book_id": book,
        "chapter_idx": i,
        "seq": 0,
        "char_start": i * 100,
        "char_end": i * 100 + len(TEXTS[i]),
        "reading_order": i,
        "section_type": "narrative",
        "chunk_type": "text",
        "pov": None,
        "chapter_title": f"第{i}章",
        "text": TEXTS[i],
    }


@pytest.fixture
def es_store():
    s = get_settings()
    store = ESChunkStore.connect(s.es, f"it_chunks_{int(time.time() * 1000)}", s.retrieval)
    store.ensure_index()
    yield store
    store.client.indices.delete(index=store.name, ignore_unavailable=True)


def test_es_sync_bm25_and_spoiler_filter(es_store):
    chunks = [_chunk(i) for i in range(5)]
    other = [{**_chunk(0, "it-2"), "chunk_id": "it-2:0000:0000"}]
    assert es_store.sync(other)["count"] == 1
    assert es_store.sync(chunks) == {"indexed": 5, "deleted": 0, "count": 6}
    assert es_store.sync(chunks[:4]) == {"indexed": 4, "deleted": 1, "count": 5}  # 幂等 + 只删本卷多余

    flt = SearchFilter(book_ids=("it-1",))
    ik = BM25Retriever(es_store, "ik", flt)
    bg = BM25Retriever(es_store, "ik_bigram", flt)
    assert {h.chunk_id for h in ik.search("长城", 10)} == {"it-1:0000:0000", "it-1:0003:0000"}
    # IK 把"雪诺"切成单字；bigram 子字段能把整词匹配到的块排在前面
    assert bg.search("雪诺", 1)[0].chunk_id in {"it-1:0001:0000", "it-1:0003:0000"}
    hits = bg.search("雪诺 长城", 10, SearchFilter(book_ids=("it-1",), max_reading_order=1))
    assert hits and all(h.fields["reading_order"] <= 1 for h in hits)
    assert es_store.count(SearchFilter(book_ids=("it-1",), max_reading_order=2)) == 3


def test_hybrid_spoiler_filter_on_real_stores(es_store):
    s = get_settings()
    chunks = [_chunk(i) for i in range(5)]
    es_store.sync(chunks)
    emb = FakeEmbedder(dim=8)
    mv = MilvusChunkStore.connect(s.milvus, f"it_hybrid_{int(time.time())}", 8, s.retrieval)
    try:
        mv.ensure_collection()
        mv.sync(chunks, emb.embed_documents([c["text"] for c in chunks]))
        base = SearchFilter(book_ids=("it-1",))
        h = HybridRetriever(
            {"bm25": BM25Retriever(es_store, "ik_bigram", base), "vector": VectorRetriever(mv, emb, base)},
            route_top_k={"bm25": 10, "vector": 10}, rrf_k=60, fusion_top_k=10,
            reranker=FakeReranker(), rerank_candidates=10, base_filter=base,
        )  # fmt: skip
        for mro in (0, 2, 4):
            hits = h.search("雪诺在长城", 10, SearchFilter(book_ids=("it-1",), max_reading_order=mro))
            assert hits and max(x.fields["reading_order"] for x in hits) <= mro
    finally:
        if mv.exists():
            mv.client.drop_collection(mv.name)
