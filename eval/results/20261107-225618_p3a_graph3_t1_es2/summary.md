# p3a_graph3_t1_es2

- 时间：2026-11-07T22:56:18；commit `a5fdbea507`（有未提交改动）
- 评测集：asoiaf_v0，划分 dev，60 题
- 检索器：{"name": "bm25+vector+graph+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}, "graph": {"name": "graph", "max_entities": 4, "per_entity_relations": 10, "per_entity_mentions": 10, "path_limit": 5, "max_tier": 1}}, "route_top_k": {"bm25": 50, "vector": 50, "graph": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}
- 分块版本：0c94b7e4；防剧透过滤：关
- 相关性判定：重叠 ≥ min(区间 × 0.5, 50 字)
- 检索延迟（ms）：{'p50': 693.0, 'p95': 773.0}；分阶段：{"bm25.search": {"p50": 9.5, "p95": 13.3}, "bm25.total": {"p50": 9.6, "p95": 13.4}, "fusion": {"p50": 0.4, "p95": 0.6}, "graph.total": {"p50": 36.8, "p95": 81.8}, "rerank": {"p50": 623.2, "p95": 691.5}, "routes": {"p50": 69.0, "p95": 93.0}, "vector.embed": {"p50": 54.7, "p95": 77.7}, "vector.search": {"p50": 12.5, "p95": 17.1}, "vector.total": {"p50": 68.4, "p95": 92.2}}
- 范围：['asoiaf-1']；small-to-big 窗口：0

| 分组 | 题数 | recall@5 | recall@10 | recall@20 | hit@5 | hit@10 | hit@20 | mrr | cov@2000 | cov@4000 | cov@8000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 15 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.872 | 1.000 | 1.000 | 1.000 |
| 人物关系 | 15 | 0.467 | 0.667 | 0.800 | 0.533 | 0.733 | 0.867 | 0.448 | 0.400 | 0.600 | 0.800 |
| 多跳 | 15 | 0.633 | 0.700 | 0.833 | 0.800 | 0.933 | 0.933 | 0.693 | 0.633 | 0.700 | 0.800 |
| 跨章节情节 | 15 | 0.144 | 0.322 | 0.489 | 0.333 | 0.600 | 0.800 | 0.257 | 0.167 | 0.289 | 0.489 |
| 来源：generated | 60 | 0.561 | 0.672 | 0.781 | 0.667 | 0.817 | 0.900 | 0.568 | 0.550 | 0.647 | 0.772 |
| **总体** | 60 | 0.561 | 0.672 | 0.781 | 0.667 | 0.817 | 0.900 | 0.568 | 0.550 | 0.647 | 0.772 |
