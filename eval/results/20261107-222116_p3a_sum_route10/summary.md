# p3a_sum_route10

- 时间：2026-11-07T22:21:15；commit `a5fdbea507`（有未提交改动）
- 评测集：asoiaf_v0，划分 dev，60 题
- 检索器：{"name": "bm25+vector+summary_locate+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es1", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}, "summary_locate": {"name": "summary_locate", "locator": {"name": "summary", "es_index": "summaries_d8139169_es1", "milvus_collection": "summaries_bgem3_1024_d8139169", "bm25_top_k": 30, "vector_top_k": 30, "rrf_k": 60}, "n_chapters": 10, "route_top_k": 50, "inner": ["bm25", "vector"]}}, "route_top_k": {"bm25": 50, "vector": 50, "summary_locate": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}
- 分块版本：0c94b7e4；防剧透过滤：关
- 相关性判定：重叠 ≥ min(区间 × 0.5, 50 字)
- 检索延迟（ms）：{'p50': 715.7, 'p95': 815.8}；分阶段：{"bm25.search": {"p50": 8.0, "p95": 10.3}, "bm25.total": {"p50": 8.0, "p95": 10.3}, "fusion": {"p50": 0.5, "p95": 0.8}, "rerank": {"p50": 558.6, "p95": 642.9}, "routes": {"p50": 151.4, "p95": 185.0}, "summary_locate.locate": {"p50": 87.6, "p95": 107.0}, "summary_locate.total": {"p50": 150.7, "p95": 184.4}, "vector.embed": {"p50": 78.0, "p95": 94.2}, "vector.search": {"p50": 11.4, "p95": 14.1}, "vector.total": {"p50": 89.9, "p95": 107.3}}
- 范围：['asoiaf-1']；small-to-big 窗口：0

| 分组 | 题数 | recall@5 | recall@10 | recall@20 | hit@5 | hit@10 | hit@20 | mrr | cov@2000 | cov@4000 | cov@8000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 15 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.872 | 1.000 | 1.000 | 1.000 |
| 人物关系 | 15 | 0.533 | 0.700 | 0.733 | 0.600 | 0.733 | 0.800 | 0.498 | 0.467 | 0.633 | 0.733 |
| 多跳 | 15 | 0.600 | 0.700 | 0.733 | 0.867 | 0.933 | 0.933 | 0.733 | 0.600 | 0.700 | 0.733 |
| 跨章节情节 | 15 | 0.144 | 0.233 | 0.344 | 0.267 | 0.400 | 0.600 | 0.223 | 0.122 | 0.200 | 0.311 |
| 来源：generated | 60 | 0.569 | 0.658 | 0.703 | 0.683 | 0.767 | 0.833 | 0.582 | 0.547 | 0.633 | 0.694 |
| **总体** | 60 | 0.569 | 0.658 | 0.703 | 0.683 | 0.767 | 0.833 | 0.582 | 0.547 | 0.633 | 0.694 |
