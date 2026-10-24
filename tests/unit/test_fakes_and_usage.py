import numpy as np
import pytest

from novel_rag.llm.fake import FakeEmbedder, FakeReranker
from novel_rag.llm.usage import UsageRecord, UsageTracker


def test_fake_embedder_shapes_and_norms():
    e = FakeEmbedder(dim=16)
    docs = e.embed_documents(["甲", "乙", "甲"])
    assert docs.shape == (3, 16) and docs.dtype == np.float32
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-5)
    assert np.allclose(docs[0], docs[2])
    q = e.embed_query("甲")
    assert q.shape == (16,) and np.allclose(q, docs[0])
    assert e.embed_documents([]).shape == (0, 16)


def test_fake_reranker_scores_align_with_candidates():
    r = FakeReranker()
    scores = r.score("北境之王", ["北境的国王", "南方的葡萄酒"])
    assert len(scores) == 2 and scores[0] > scores[1]


def test_usage_summary_excludes_cached_tokens():
    t = UsageTracker()
    t.record(UsageRecord("gen", "m", 100, 20, 1.0, cached=False, attempts=2))
    t.record(UsageRecord("gen", "m", 0, 0, 0.01, cached=True))
    s = t.summary()["gen/m"]
    assert s == {
        "calls": 2,
        "cache_hits": 1,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "latency_s": pytest.approx(1.01),
        "retries": 1,
    }
