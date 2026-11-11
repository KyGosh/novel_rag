# p31_v1_g_no_event

- 时间：2026-11-11T23:21:54；commit `7c9d06cb37`（有未提交改动）
- 评测集：asoiaf_v1，划分 dev，141 题
- 检索器：{"name": "bm25+vector+graph+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}, "graph": {"name": "graph", "max_entities": 4, "per_entity_relations": 10, "per_entity_mentions": 10, "path_limit": 5, "max_tier": 1, "entity_types": null, "path_node_types": null, "path_exclude_node_types": ["Event"], "min_entity_mentions": 1, "kg_version": "56be6611"}}, "route_top_k": {"bm25": 50, "vector": 50, "graph": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}
- 分块版本：0c94b7e4；防剧透过滤：开
- 相关性判定：重叠 ≥ min(区间 × 0.5, 50 字)
- 检索延迟（ms）：{'p50': 826.7, 'p95': 926.3}；分阶段：{"bm25.search": {"p50": 11.6, "p95": 14.9}, "bm25.total": {"p50": 11.7, "p95": 14.9}, "fusion": {"p50": 0.5, "p95": 0.9}, "graph.total": {"p50": 36.4, "p95": 98.4}, "rerank": {"p50": 742.6, "p95": 826.9}, "routes": {"p50": 79.5, "p95": 104.9}, "vector.embed": {"p50": 64.2, "p95": 81.2}, "vector.search": {"p50": 14.1, "p95": 16.5}, "vector.total": {"p50": 78.9, "p95": 95.1}}
- 范围：['asoiaf-1', 'asoiaf-2', 'asoiaf-3', 'asoiaf-4', 'asoiaf-5']；small-to-big 窗口：0

| 分组 | 题数 | recall@5 | recall@10 | recall@20 | hit@5 | hit@10 | hit@20 | mrr | cov@2000 | cov@4000 | cov@8000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.944 | 1.000 | 1.000 | 1.000 |
| 人物关系 | 33 | 0.939 | 0.970 | 0.970 | 0.939 | 0.970 | 0.970 | 0.759 | 0.894 | 0.970 | 0.970 |
| 多跳 | 44 | 0.515 | 0.648 | 0.689 | 0.818 | 0.932 | 0.932 | 0.618 | 0.481 | 0.629 | 0.689 |
| 跨章节情节 | 34 | 0.471 | 0.544 | 0.706 | 0.765 | 0.824 | 0.882 | 0.659 | 0.441 | 0.529 | 0.691 |
| 来源：generated | 141 | 0.707 | 0.773 | 0.825 | 0.872 | 0.929 | 0.943 | 0.731 | 0.678 | 0.764 | 0.822 |
| **总体** | 141 | 0.707 | 0.773 | 0.825 | 0.872 | 0.929 | 0.943 | 0.731 | 0.678 | 0.764 | 0.822 |
