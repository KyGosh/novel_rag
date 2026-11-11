# p3a_graph3

- 时间：2026-11-07T22:49:18；commit `a5fdbea507`（有未提交改动）
- 评测集：asoiaf_v0，划分 dev，60 题
- 检索器：{"name": "bm25+vector+graph+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es1", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}, "graph": {"name": "graph", "max_entities": 4, "per_entity_relations": 10, "per_entity_mentions": 10, "path_limit": 5}}, "route_top_k": {"bm25": 50, "vector": 50, "graph": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}
- 分块版本：0c94b7e4；防剧透过滤：关
- 相关性判定：重叠 ≥ min(区间 × 0.5, 50 字)
- 检索延迟（ms）：{'p50': 647.1, 'p95': 735.4}；分阶段：{"bm25.search": {"p50": 6.4, "p95": 8.8}, "bm25.total": {"p50": 6.4, "p95": 8.9}, "fusion": {"p50": 0.5, "p95": 0.7}, "graph.total": {"p50": 65.9, "p95": 127.8}, "rerank": {"p50": 575.4, "p95": 635.2}, "routes": {"p50": 72.8, "p95": 128.3}, "vector.embed": {"p50": 57.3, "p95": 67.8}, "vector.search": {"p50": 11.4, "p95": 14.7}, "vector.total": {"p50": 69.1, "p95": 80.4}}
- 范围：['asoiaf-1']；small-to-big 窗口：0

| 分组 | 题数 | recall@5 | recall@10 | recall@20 | hit@5 | hit@10 | hit@20 | mrr | cov@2000 | cov@4000 | cov@8000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 15 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.872 | 1.000 | 1.000 | 1.000 |
| 人物关系 | 15 | 0.400 | 0.567 | 0.633 | 0.467 | 0.600 | 0.667 | 0.425 | 0.400 | 0.567 | 0.633 |
| 多跳 | 15 | 0.633 | 0.700 | 0.800 | 0.800 | 0.933 | 0.933 | 0.690 | 0.633 | 0.700 | 0.800 |
| 跨章节情节 | 15 | 0.144 | 0.344 | 0.478 | 0.333 | 0.600 | 0.733 | 0.287 | 0.167 | 0.311 | 0.444 |
| 来源：generated | 60 | 0.544 | 0.653 | 0.728 | 0.650 | 0.783 | 0.833 | 0.569 | 0.550 | 0.644 | 0.719 |
| **总体** | 60 | 0.544 | 0.653 | 0.728 | 0.650 | 0.783 | 0.833 | 0.569 | 0.550 | 0.644 | 0.719 |
