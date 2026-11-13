# p5_test_graphrag

- 时间：2026-11-13T19:57:28；commit `3fcb8a787e`（有未提交改动）
- 系统：{"name": "graphrag", "retriever": {"name": "bm25+vector+graph+rerank", "routes": {"bm25": {"name": "bm25_ik", "index": "chunks_0c94b7e4_es2", "fields": ["text"], "boosts": {"text": 1.0}}, "vector": {"name": "vector", "collection": "chunks_bgem3_1024_0c94b7e4", "embedding_model": "BAAI/bge-m3", "metric": "COSINE", "hnsw_ef": 128}, "graph": {"name": "graph", "max_entities": 4, "per_entity_relations": 10, "per_entity_mentions": 10, "path_limit": 5, "max_tier": 1, "entity_types": null, "path_node_types": null, "path_exclude_node_types": null, "min_entity_mentions": 1, "kg_version": "56be6611"}}, "route_top_k": {"bm25": 50, "vector": 50, "graph": 50}, "rrf_k": 60, "fusion_top_k": 40, "reranker": "BAAI/bge-reranker-v2-m3", "rerank_candidates": 40}, "top_k": 8, "pool": 40, "budget": 4000, "answer_model": "workbuddy/deepseek-v4.1-flash"}
- 评测集：asoiaf_v1，划分 test，141 题；采样编号 0；并发 8
- 终止原因：{'answered': 141}；判分方式：{'rule': 82, 'judge': 50, 'abstain': 9}；无效引用 9 个；出错 0 题
- 延迟（并发运行，仅供参考，正式延迟见单独的延迟测量）：{'p50': 5370.9, 'p95': 7237.2}

| 分组 | 题数 | acc | partial | acc_lenient | abstain | process_cov | cite_recall | cite_precision | ctx_recall | recall@5 | recall@20 | mrr | cov@4000 | llm_calls | tool_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 事实 | 30 | 0.967 | 0.000 | 0.967 | 0.000 | 0.967 | 0.933 | 0.813 | 0.967 | 0.933 | 0.983 | 0.887 | 0.967 | 1.0 | 1.0 | 3461.9 |
| 人物关系 | 33 | 0.939 | 0.000 | 0.939 | 0.030 | 0.788 | 0.742 | 0.656 | 0.788 | 0.727 | 0.924 | 0.644 | 0.818 | 1.0 | 1.0 | 3445.8 |
| 多跳 | 44 | 0.750 | 0.023 | 0.773 | 0.114 | 0.627 | 0.316 | 0.627 | 0.627 | 0.570 | 0.695 | 0.594 | 0.627 | 1.0 | 1.0 | 3427.2 |
| 跨章节情节 | 34 | 0.559 | 0.176 | 0.735 | 0.088 | 0.583 | 0.500 | 0.581 | 0.583 | 0.537 | 0.721 | 0.687 | 0.613 | 1.0 | 1.0 | 3442.9 |
| **总体** | 141 | 0.794 | 0.050 | 0.844 | 0.064 | 0.726 | 0.592 | 0.666 | 0.726 | 0.676 | 0.816 | 0.691 | 0.741 | 1.0 | 1.0 | 3442.7 |

总体准确率 0.794，95% CI [0.7305, 0.8582]
