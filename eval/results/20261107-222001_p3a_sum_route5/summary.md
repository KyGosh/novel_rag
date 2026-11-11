# p3a_sum_route5

- 时间：2026-11-07T22:20:01；commit `a5fdbea507`（有未提交改动）
- 评测集：asoiaf_v0，划分 dev，60 题
- 检索器：{"name": "bm25+vector+summary_locate+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es1", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}, "summary_locate": {"name": "summary_locate", "locator": {"name": "summary", "es_index": "summaries_d8139169_es1", "milvus_collection": "summaries_bgem3_1024_d8139169", "bm25_top_k": 30, "vector_top_k": 30, "rrf_k": 60}, "n_chapters": 5, "route_top_k": 50, "inner": ["bm25", "vector"]}}, "route_top_k": {"bm25": 50, "vector": 50, "summary_locate": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}
- 分块版本：0c94b7e4；防剧透过滤：关
- 相关性判定：重叠 ≥ min(区间 × 0.5, 50 字)
- 检索延迟（ms）：{'p50': 719.8, 'p95': 791.2}；分阶段：{"bm25.search": {"p50": 9.5, "p95": 13.3}, "bm25.total": {"p50": 9.5, "p95": 13.3}, "fusion": {"p50": 0.5, "p95": 0.8}, "rerank": {"p50": 559.2, "p95": 611.4}, "routes": {"p50": 154.5, "p95": 195.1}, "summary_locate.locate": {"p50": 91.6, "p95": 109.8}, "summary_locate.total": {"p50": 154.0, "p95": 194.5}, "vector.embed": {"p50": 78.0, "p95": 99.2}, "vector.search": {"p50": 13.1, "p95": 15.6}, "vector.total": {"p50": 91.5, "p95": 111.9}}
- 范围：['asoiaf-1']；small-to-big 窗口：0

| 分组 | 题数 | recall@5 | recall@10 | recall@20 | hit@5 | hit@10 | hit@20 | mrr | cov@2000 | cov@4000 | cov@8000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 15 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.756 | 0.800 | 0.800 | 0.800 |
| 人物关系 | 15 | 0.467 | 0.567 | 0.667 | 0.533 | 0.600 | 0.733 | 0.431 | 0.433 | 0.567 | 0.667 |
| 多跳 | 15 | 0.500 | 0.633 | 0.633 | 0.733 | 0.800 | 0.800 | 0.667 | 0.500 | 0.600 | 0.633 |
| 跨章节情节 | 15 | 0.178 | 0.300 | 0.322 | 0.400 | 0.667 | 0.667 | 0.256 | 0.144 | 0.256 | 0.322 |
| 来源：generated | 60 | 0.486 | 0.575 | 0.606 | 0.617 | 0.717 | 0.750 | 0.527 | 0.469 | 0.556 | 0.606 |
| **总体** | 60 | 0.486 | 0.575 | 0.606 | 0.617 | 0.717 | 0.750 | 0.527 | 0.469 | 0.556 | 0.606 |
