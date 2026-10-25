"""集成测试：真实 Milvus 上的 chunk 同步与防剧透过滤；本地 reranker tokenizer 计数。

运行：uv run pytest -m integration
"""

import time

import numpy as np
import pytest

from novel_rag.config import get_settings
from novel_rag.llm.fake import FakeEmbedder
from novel_rag.retrieval.filters import SearchFilter
from novel_rag.retrieval.milvus_store import MilvusChunkStore, VectorRetriever

pytestmark = pytest.mark.integration


def _chunk(i):
    return {
        "chunk_id": f"it-1:{i:04d}:0000",
        "text": f"第{i}章的内容",
        "book_id": "it-1",
        "chapter_idx": i,
        "seq": 0,
        "reading_order": i,
        "section_type": "narrative",
        "chunk_type": "text",
        "pov": None,
        "chapter_title": f"第{i}章",
    }


def test_sync_and_spoiler_filter_on_real_milvus():
    s = get_settings()
    emb = FakeEmbedder(dim=8)
    store = MilvusChunkStore.connect(s.milvus, f"it_chunks_{int(time.time())}", 8, s.retrieval)
    try:
        store.ensure_collection()
        chunks = [_chunk(i) for i in range(5)]
        vecs = emb.embed_documents([c["text"] for c in chunks])
        assert store.sync(chunks, vecs)["count"] == 5
        assert store.sync(chunks[:4], vecs[:4]) == {"upserted": 4, "deleted": 1, "count": 4}  # 幂等 + 删除多余
        retr = VectorRetriever(store, emb, SearchFilter(book_ids=("it-1",)))
        assert retr.search("第3章的内容", 1)[0].chunk_id == "it-1:0003:0000"
        hits = retr.search("第3章的内容", 10, SearchFilter(book_ids=("it-1",), max_reading_order=1))
        assert {h.fields["reading_order"] for h in hits} <= {0, 1} and len(hits) == 2
    finally:
        if store.exists():
            store.client.drop_collection(store.name)


def test_reranker_token_counter():
    from novel_rag.llm import get_token_counter

    counts = get_token_counter().count(["守夜人站在长城上。", ""])
    assert counts[0] > 0 and counts[1] == 0
    assert np.isfinite(counts[0])
