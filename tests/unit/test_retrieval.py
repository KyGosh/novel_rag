"""检索公共层：过滤表达式（含防剧透）、向量缓存、编码文本、collection 命名、向量检索器与 Milvus 同步（mock 客户端）。"""

import numpy as np

from novel_rag.config import RetrievalConfig
from novel_rag.llm.fake import FakeEmbedder
from novel_rag.retrieval.embed_cache import EmbeddingCache, encode_text
from novel_rag.retrieval.filters import SearchFilter
from novel_rag.retrieval.milvus_store import MilvusChunkStore, VectorRetriever, collection_name


def test_filter_expressions():
    assert SearchFilter().to_milvus_expr() == ""
    f = SearchFilter(book_ids=("demo-1",), max_reading_order=12, section_types=("narrative",))
    assert f.to_milvus_expr() == 'book_id in ["demo-1"] and reading_order <= 12 and section_type in ["narrative"]'
    assert f.with_max_reading_order(3).to_milvus_expr().count("reading_order <= 3") == 1


def test_spoiler_filter_accepts():
    f = SearchFilter(max_reading_order=5)
    assert f.accepts({"reading_order": 5}) and not f.accepts({"reading_order": 6})
    assert SearchFilter(book_ids=("a",)).accepts({"book_id": "a"}) and not SearchFilter(book_ids=("a",)).accepts(
        {"book_id": "b"}
    )


class CountingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__(dim=4)
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return super().embed_documents(texts)


def test_embedding_cache_hits():
    emb = CountingEmbedder()
    cache = EmbeddingCache(emb, {})
    a = cache.embed_documents(["甲", "乙"])
    b = cache.embed_documents(["乙", "丙"])
    assert emb.calls == 3 and cache.hits == 1
    assert np.allclose(a[1], b[0])


def test_encode_text_and_collection_name():
    chunk = {"text": "正文", "chapter_title": "第一章"}
    assert encode_text(chunk, "raw") == "正文"
    assert encode_text(chunk, "title_prefix", "某书") == "某书 第一章\n正文"
    assert collection_name("BAAI/bge-m3", 1024, "abcd1234", "raw") == "chunks_bgem3_1024_abcd1234"
    assert collection_name("BAAI/bge-m3", 1024, "abcd1234", "title_prefix").endswith("_tp")


class FakeClient:
    """模仿 MilvusClient 的最小子集，记录 upsert / delete / search 的调用。"""

    def __init__(self, existing_ids=()):
        self.rows = {i: {"chunk_id": i} for i in existing_ids}
        self.last_filter = None

    def upsert(self, name, rows):
        self.rows.update({r["chunk_id"]: r for r in rows})

    def delete(self, name, ids):
        for i in ids:
            self.rows.pop(i, None)

    def flush(self, name): ...

    def load_collection(self, name): ...

    def query(self, name, filter, output_fields):
        return [{"count(*)": len(self.rows)}]

    def query_iterator(self, name, batch_size, filter, output_fields):
        batches = [[{"chunk_id": i} for i in self.rows], []]

        class It:
            def next(self):
                return batches.pop(0)

            def close(self): ...

        return It()

    def search(self, name, data, limit, filter, output_fields, search_params):
        self.last_filter = filter
        rows = sorted(self.rows.values(), key=lambda r: r["chunk_id"])[:limit]
        return [
            [
                {
                    "id": r["chunk_id"],
                    "distance": 0.9 - 0.1 * i,
                    "entity": {k: v for k, v in r.items() if k != "vector"},
                }
                for i, r in enumerate(rows)
            ]
        ]


def _chunk(cid, ro):
    return {
        "chunk_id": cid,
        "text": "t",
        "book_id": "demo-1",
        "chapter_idx": ro,
        "seq": 0,
        "reading_order": ro,
        "section_type": "narrative",
        "chunk_type": "text",
        "pov": None,
        "chapter_title": "章",
    }


def test_sync_is_idempotent_and_removes_stale():
    client = FakeClient(existing_ids=["old"])
    store = MilvusChunkStore(client, "c", 4, RetrievalConfig())
    chunks = [_chunk("a", 1), _chunk("b", 2)]
    res = store.sync(chunks, np.zeros((2, 4), dtype=np.float32))
    assert res == {"upserted": 2, "deleted": 1, "count": 2}
    assert store.sync(chunks, np.zeros((2, 4), dtype=np.float32))["count"] == 2
    assert client.rows["a"]["pov"] == ""


def test_vector_retriever_passes_spoiler_filter():
    client = FakeClient()
    store = MilvusChunkStore(client, "c", 4, RetrievalConfig())
    store.sync([_chunk("a", 1), _chunk("b", 2)], np.zeros((2, 4), dtype=np.float32))
    r = VectorRetriever(store, FakeEmbedder(dim=4), SearchFilter(book_ids=("demo-1",)))
    hits = r.search("问题", 5, SearchFilter(book_ids=("demo-1",), max_reading_order=1))
    assert client.last_filter == 'book_id in ["demo-1"] and reading_order <= 1'
    assert [h.rank for h in hits] == [1, 2] and hits[0].chunk_id == "a" and "total" in r.last_latency_ms
    r.search("问题", 5)
    assert client.last_filter == 'book_id in ["demo-1"]'
