# p3a_graph3_sum

- 时间：2026-11-07T22:50:35；commit `a5fdbea507`（有未提交改动）
- 评测集：asoiaf_v0，划分 dev，60 题
- 检索器：{"name": "bm25+vector+graph+summary_locate+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es1", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}, "graph": {"name": "graph", "max_entities": 4, "per_entity_relations": 10, "per_entity_mentions": 10, "path_limit": 5}, "summary_locate": {"name": "summary_locate", "locator": {"name": "summary", "es_index": "summaries_d8139169_es1", "milvus_collection": "summaries_bgem3_1024_d8139169", "bm25_top_k": 30, "vector_top_k": 30, "rrf_k": 60}, "n_chapters": 5, "route_top_k": 50, "inner": ["bm25", "vector"]}}, "route_top_k": {"bm25": 50, "vector": 50, "graph": 50, "summary_locate": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}
- 分块版本：0c94b7e4；防剧透过滤：关
- 相关性判定：重叠 ≥ min(区间 × 0.5, 50 字)
- 检索延迟（ms）：{'p50': 767.0, 'p95': 872.3}；分阶段：{"bm25.search": {"p50": 7.1, "p95": 8.8}, "bm25.total": {"p50": 7.1, "p95": 8.9}, "fusion": {"p50": 0.7, "p95": 1.1}, "graph.total": {"p50": 78.3, "p95": 147.3}, "rerank": {"p50": 588.6, "p95": 683.3}, "routes": {"p50": 170.4, "p95": 216.5}, "summary_locate.locate": {"p50": 110.6, "p95": 132.0}, "summary_locate.total": {"p50": 169.1, "p95": 215.2}, "vector.embed": {"p50": 100.2, "p95": 120.9}, "vector.search": {"p50": 11.6, "p95": 17.5}, "vector.total": {"p50": 113.1, "p95": 136.3}}
- 范围：['asoiaf-1']；small-to-big 窗口：0

| 分组 | 题数 | recall@5 | recall@10 | recall@20 | hit@5 | hit@10 | hit@20 | mrr | cov@2000 | cov@4000 | cov@8000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 15 | 0.933 | 0.933 | 0.933 | 0.933 | 0.933 | 0.933 | 0.806 | 0.933 | 0.933 | 0.933 |
| 人物关系 | 15 | 0.500 | 0.700 | 0.800 | 0.600 | 0.733 | 0.867 | 0.504 | 0.467 | 0.700 | 0.733 |
| 多跳 | 15 | 0.567 | 0.700 | 0.733 | 0.867 | 0.933 | 0.933 | 0.724 | 0.533 | 0.700 | 0.733 |
| 跨章节情节 | 15 | 0.144 | 0.233 | 0.311 | 0.267 | 0.467 | 0.600 | 0.227 | 0.122 | 0.233 | 0.311 |
| 来源：generated | 60 | 0.536 | 0.642 | 0.694 | 0.667 | 0.767 | 0.833 | 0.565 | 0.514 | 0.642 | 0.678 |
| **总体** | 60 | 0.536 | 0.642 | 0.694 | 0.667 | 0.767 | 0.833 | 0.565 | 0.514 | 0.642 | 0.678 |
